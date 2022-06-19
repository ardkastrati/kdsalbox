"""
Trainer
-------

Trains a hypernetwork and mainnetwork on multiple tasks at the same time
and reports the progress and metrics to wandb.

TODO: add a lot more documentation here about all the parameters

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

class Trainer(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        train_conf = conf["train"]

        self._batch_size = train_conf["batch_size"]
        self._imgs_per_task_train = train_conf["imgs_per_task_train"]
        self._imgs_per_task_val = train_conf["imgs_per_task_val"]

        self._export_path = "export/"
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._auto_checkpoint_steps = train_conf["auto_checkpoint_steps"]
        self._input_saliencies = train_conf["input_saliencies"]
        self._train_img_path = train_conf["input_images_train"]
        self._input_images_run = train_conf["input_images_run"]
        self._val_img_path = train_conf["input_images_val"]

        self._tasks = conf["tasks"]
        self._task_cnt = conf["model"]["hnet"]["task_cnt"]
        self._batches_per_task_train = self._imgs_per_task_train // self._batch_size
        self._batches_per_task_val = self._imgs_per_task_val // self._batch_size

        self._loss_fn = train_conf["loss"]
        self._epochs = train_conf["epochs"]
        self._consecutive_batches_per_task = train_conf["consecutive_batches_per_task"]
        self._lr = train_conf["lr"]
        self._lr_decay = train_conf["lr_decay"]
        self._freeze_encoder_steps = train_conf["freeze_encoder_steps"]
        self._decay_epochs = train_conf["decay_epochs"]


        wandb_conf = train_conf["wandb"]
        self.wandb_watch_log = wandb_conf["watch"]["log"]
        self.wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]
        self.save_checkpoints_to_wandb = wandb_conf["save_checkpoints_to_wandb"]

        # convert parameter dicts to parametermap such that it can be used in process()
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])
    
    # train or evaluate one epoch for all models (mode in [train, val])
    # return loss, model
    def _train_one_epoch(self, model, dataloaders, criterion, optimizer, mode):
        if mode == "train": model.train()
        elif mode == "val": model.eval()

        all_loss = []

        # defines which batch will be loaded from which task/model
        if mode == "train":
            limit = self._batches_per_task_train // self._consecutive_batches_per_task
            all_batches = np.concatenate([np.repeat(model.task_to_id(task), limit) for task in self._tasks])
            np.random.shuffle(all_batches)
            all_batches = np.repeat(all_batches, self._consecutive_batches_per_task)
        else:
            all_batches = np.concatenate([np.repeat(model.task_to_id(task), self._batches_per_task_val) for task in self._tasks])

        # for each model
        data_iters = [iter(d) for d in dataloaders[mode].values()] # Note: DataLoader shuffles when iterator is created
        for (i,task_id) in enumerate(all_batches):
            X,y = next(data_iters[task_id])

            optimizer.zero_grad()

            # put data on GPU (if cuda)
            X = X.to(self._device)
            y = y.to(self._device)

            pred = model(task_id.item(), X)
            loss = criterion(pred, y)

            # training
            if mode == "train":
                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())

            # validation
            elif mode == "val":
                with torch.no_grad():
                    all_loss.append(loss.item())

            # logging
            if i%100 == 0:
                print(f"Batch {i}/{len(all_batches)}: current accumulated loss {np.mean(all_loss)}", flush=True)
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss)

    # tracks and reports some metrics of the model that is being trained
    def track_progress(self, epoch : int, model : HyperModel):
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
            data.append(row)

        table = wandb.Table(data=data, columns=cols)
        wandb.log({f"Progress Epoch {epoch}": table})

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
        sal_folders = [os.path.join(self._input_saliencies, task) for task in self._tasks] # path to saliency folder for all models

        train_datasets = [TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]
        val_datasets = [TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        self._dataloaders = {
            "train": {task:DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) for (task,ds) in zip(self._tasks, train_datasets)},
            "val": {task:DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) for (task,ds) in zip(self._tasks, val_datasets)},
        }

        self._run_dataloader = DataLoader(RunDataManager(self._input_images_run, "", verbose=False, recursive=False), batch_size=1)

        # sanity checks
        assert self._task_cnt == len(self._tasks)
        assert self._imgs_per_task_train <= min([len(ds) for ds in train_datasets])
        assert self._imgs_per_task_val <= min([len(ds) for ds in val_datasets])


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
        wandb.watch((model.hnet, model.mnet), loss, log=self.wandb_watch_log, log_freq=self.wandb_watch_log_freq)
        
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

                self.save(path, model, self.save_checkpoints_to_wandb)
                
                # overwrite the best model
                if is_best_model:
                    smallest_loss = loss_val
                    self.save(export_path_best, model, self.save_checkpoints_to_wandb)
                
                self.track_progress(epoch, model)
            
            # save/overwrite results at the end of each epoch
            stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "all_results").replace("\\", "/")
            table = wandb.Table(data=all_epochs, columns=["Epoch", "Loss-Train", "Loss-Val"])
            wandb.log({
                    "epoch": epoch,
                    "loss train": loss_train,
                    "loss val": loss_val,
                    "learning rate": lr,
                    "encoder_frozen": float(epoch < self._freeze_encoder_steps),
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
