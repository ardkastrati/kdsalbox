"""
Trainer
-------

Trains a hypernetwork and mainnetwork on just one task
and reports the progress and metrics to wandb.

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from backend.datasets import TrainDataManager, RunDataManager
from backend.parameters import ParameterMap
from backend.image_processing import process
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel

class PreTrainerOneTask(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        train_conf = conf["pretrain_one_task"]

        self._batch_size = train_conf["batch_size"]

        self._export_path = "export/"
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._auto_checkpoint_steps = train_conf["auto_checkpoint_steps"]
        self._input_saliencies = train_conf["input_saliencies"]
        self._train_img_path = train_conf["input_images_train"]
        self._input_images_run = train_conf["input_images_run"]
        self._val_img_path = train_conf["input_images_val"]

        self._tasks = conf["tasks"]
        self._task_cnt = conf["model"]["hnet"]["task_cnt"]
        self._selected_task = self._tasks[0]

        self._loss_fn = train_conf["loss"]
        self._epochs = train_conf["epochs"]
        self._lr = train_conf["lr"]
        self._lr_decay = train_conf["lr_decay"]
        self._freeze_encoder_steps = train_conf["freeze_encoder_steps"]
        self._decay_epochs = train_conf["decay_epochs"]

        self._log_freq = train_conf["log_freq"]

        wandb_conf = train_conf["wandb"]
        self._wandb_watch_log = wandb_conf["watch"]["log"]
        self._wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]
        self._save_checkpoints_to_wandb = wandb_conf["save_checkpoints_to_wandb"]

        # convert parameter dicts to parametermap such that it can be used in process()
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])
    
    # train or evaluate one epoch for all models (mode in [train, val])
    # return loss, model
    def _train_one_epoch(self, model, dataloaders, criterion, optimizer, mode):
        if mode == "train": model.train()
        elif mode == "val": model.eval()

        all_loss = []

        task_id = model.task_to_id(self._selected_task)
        dataloader = dataloaders[mode]
        for i,(X,y) in enumerate(dataloader):
            optimizer.zero_grad()

            # put data on GPU (if cuda)
            X = X.to(self._device)
            y = y.to(self._device)

            pred = model(task_id, X)
            loss = criterion(pred, y)

            # training
            if mode == "train":
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                all_loss.append(loss.item())

                # logging
                if i%self._log_freq == 0:
                    print(f"Batch {i}/{len(dataloader)}: current accumulated loss {np.mean(all_loss)}", flush=True)
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del pred
                del loss
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss)

    # tracks and reports some metrics of the model that is being trained
    def track_progress(self, epoch : int, model : HyperModel):
        with torch.no_grad():
            cols = ["Model"]
            cols.extend([os.path.basename(output_path[0]) for (_, _, output_path) in self._run_dataloader])
            data = []
            for task in self._tasks:
                row = [task]
                task_id = model.task_to_id(task)
                for (image, _, _) in self._run_dataloader:
                    image = image.to(self._device)
                    saliency_map = model.compute_saliency(image, task_id)
                    post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocess_parameter_map)*255).astype(np.uint8), 0, 255)
                    img = wandb.Image(post_processed_image)
                    row.append(img)

                    if torch.cuda.is_available(): # avoid GPU out of mem
                        del image
                        del saliency_map
                        torch.cuda.empty_cache()
                
                data.append(row)

            table = wandb.Table(data=data, columns=cols)
            wandb.log({f"{self.name} - Progress Epoch {epoch}": table})

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Trainer expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # setup logging
        self._export_dir = os.path.join(self._logging_dir, self._export_path)
        os.makedirs(self._logging_dir, exist_ok=True)
        os.makedirs(self._export_dir, exist_ok=True)

        # data loading
        sal_path = os.path.join(self._input_saliencies, self._selected_task)

        train_dataset = TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map)
        val_dataset = TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map)

        self._dataloaders = {
            "train": DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4),
        }

        self._run_dataloader = DataLoader(RunDataManager(self._input_images_run, "", verbose=False, recursive=False), batch_size=1)

        # sanity checks
        assert self._task_cnt == len(self._tasks)


    def execute(self):
        super().execute()

        export_path_best = os.path.join(self._export_dir, "best.pth")
        export_path_final = os.path.join(self._export_dir, "final.pth")

        model = self._model
        model.build()
        model.to(self._device)

        lr = self._lr
        lr_decay = self._lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = {
            "BCELoss": torch.nn.BCELoss(),
            "L1Loss": torch.nn.L1Loss()
        }
        loss = losses[self._loss_fn]
        
        # report to wandb
        wandb.watch((model.hnet, model.mnet), loss, log=self._wandb_watch_log, log_freq=self._wandb_watch_log_freq)
        
        all_epochs = []
        smallest_loss = None

        model.mnet.freeze_encoder()
        if self._verbose: print("Encoder frozen...")

        # evaluate how the model performs initially
        self.track_progress(-1, model)

        # training loop
        for epoch in range(0, self._epochs):
            # decrease learning rate over time
            if epoch in self._decay_epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_decay
                lr = lr * lr_decay

            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                model.mnet.unfreeze_encoder()

            # train the networks
            loss_train = self._train_one_epoch(model, self._dataloaders, loss, optimizer, "train")
            if self._verbose: self.pretty_print_epoch(epoch, "train", loss_train, lr)

            # validate the networks
            loss_val = self._train_one_epoch(model, self._dataloaders, loss, optimizer, "val")
            if self._verbose: self.pretty_print_epoch(epoch, "val", loss_val, lr)

            ### REPORTING / STATS ###
            all_epochs.append([epoch, loss_train, loss_val]) 

            # if better performance than all previous => save weights as checkpoint
            is_best_model = smallest_loss is None or loss_val < smallest_loss
            if epoch % self._auto_checkpoint_steps == 0 or is_best_model:
                checkpoint_dir = os.path.join(self._logging_dir, f"checkpoint_in_epoch_{epoch}/")
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = f"{checkpoint_dir}/{epoch}_{loss_val:f}.pth"

                self.save(path, model, self._save_checkpoints_to_wandb)
                
                # overwrite the best model
                if is_best_model:
                    smallest_loss = loss_val
                    self.save(export_path_best, model, self._save_checkpoints_to_wandb)
                
                self.track_progress(epoch, model)
            
            # save/overwrite results at the end of each epoch
            stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "all_results").replace("\\", "/")
            table = wandb.Table(data=all_epochs, columns=["Epoch", "Loss-Train", "Loss-Val"])
            wandb.log({
                    f"{self.name} epoch": epoch,
                    f"{self.name} loss train": loss_train,
                    f"{self.name} loss val": loss_val,
                    f"{self.name} learning rate": lr,
                    stats_file:table
                })
        
        # save the final model
        self.save(export_path_final, model, save_to_wandb=True)

        return model
    
    def cleanup(self):
        super().cleanup()

        del self._dataloaders

    # saves the model to disk & wandb
    def save(self, path : str, model : HyperModel, save_to_wandb : bool = True):
        model.save(path)

        if save_to_wandb:
            wandb.save(path, base_path=wandb.run.dir)

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")
