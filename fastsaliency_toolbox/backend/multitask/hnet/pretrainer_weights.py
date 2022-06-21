"""
PreTrainer
----------

Trains a hypernetwork to output some specific weights (e.g. to learn pretrained weights before the actual training begins)

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

import backend.student as stud
from backend.multitask.hnet.hyper_model import HyperModel
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.datasets import WeightDataset

class PreTrainerWeights(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        pretrain_conf = conf["pretrain_weights"]

        self._export_path = "export/"
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._target_model_weights = pretrain_conf["target_model_weights"]
        self._auto_checkpoint_steps = pretrain_conf["auto_checkpoint_steps"]

        self._tasks = conf["tasks"]
        self._task_cnt = conf["model"]["hnet"]["task_cnt"]

        self._loss_fn = pretrain_conf["loss"]
        self._epochs = pretrain_conf["epochs"]
        self._lr = pretrain_conf["lr"]
        self._lr_decay = pretrain_conf["lr_decay"]
        self._decay_epochs = pretrain_conf["decay_epochs"]
        self._batch_size = pretrain_conf["batch_size"]


        wandb_conf = pretrain_conf["wandb"]
        self._wandb_watch_log = wandb_conf["watch"]["log"]
        self._wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]
        self._save_checkpoints_to_wandb = wandb_conf["save_checkpoints_to_wandb"]
    
    # evaluate how different the layers of the models are
    def _evaluate_model_differences(self, target_model_weights):
        model_p = [(task, os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        model_weights = []
        for t,p in model_p:
            model = stud.Student()
            state_dict = torch.load(p, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict['student_model'])
            
            model_weights.append((t,{n:p.data.flatten() for n,p in model.named_parameters()}))
            

        for i in range(len(model_weights)):
            for j in range(i+1, len(model_weights)):
                t1,ws1 = model_weights[i]
                t2,ws2 = model_weights[j]

                # compare the two different models
                print(f"{t1} vs {t2}")

                total_dist = 0.0
                total_dist_enc = 0.0
                total_enc = 0
                total_dist_dec = 0.0
                total_dec = 0

                all_dist = []
                for n in ws1.keys():
                    w1 = ws1[n]
                    w2 = ws2[n]
                    dist = torch.sum(torch.abs(w1 - w2)) / len(w1)
                    total_dist += dist

                    if "encoder" in n:
                        total_dist_enc += dist
                        total_enc += 1
                    elif "decoder" in n:
                        total_dist_dec += dist
                        total_dec += 1
                    
                    all_dist.append((n, dist))
                

                print(f"Average {total_dist / len(ws1.keys())}")
                print(f"Average Encoder {total_dist_enc / total_enc}")
                print(f"Average Decoder {total_dist_dec / total_dec}")

                # all_dist = sorted(all_dist, key=(lambda t : t[1]), reverse=True)
                # for n,d in all_dist:
                #     if d < 0.5: continue
                #     print(f"\t{n}: {d:.3}")

    def map_old_to_new_weights(self, named_weights, model, verbose=False):
        decoder_selection = [
            "conv7_3.weight",   "conv7_3.bias",     "bn7_3.weight",     "bn7_3.bias",
            "conv8_1.weight",   "conv8_1.bias",     "bn8_1.weight",     "bn8_1.bias",
            "conv8_2.weight",   "conv8_2.bias",     "bn8_2.weight",     "bn8_2.bias", 
            "conv9_1.weight",   "conv9_1.bias",     "bn9_1.weight",     "bn9_1.bias",
            "conv9_2.weight",   "conv9_2.bias",     "bn9_2.weight",     "bn9_2.bias",
            "conv10_1.weight",  "conv10_1.bias",    "bn10_1.weight",    "bn10_1.bias",
            "conv10_2.weight",  "conv10_2.bias",    "bn10_2.weight",    "bn10_2.bias",
            "output.weight",    "output.bias"
        ]
        
        new_weights = []
        new_encoder_param_cnt = len(list(self._model.mnet.get_layer("encoder").parameters()))
        # select the first few params, assuming that 
        encoder_selection = [n for n,p in model.encoder.named_parameters()][new_encoder_param_cnt:] 

        if verbose: print("selecting encoder.")
        for s in encoder_selection:
            if verbose: print(f"\t{s}")
            new_weights.append(named_weights[f"encoder.{s}"])    

        if verbose: print("selecting decoder.")
        for s in decoder_selection:
            if verbose: print(f"\t{s}")
            new_weights.append(named_weights[f"decoder.{s}"])

        return new_weights

    # train or evaluate one epoch for all tasks (mode in [train, val])
    # return loss
    def _train_one_epoch(self, hnet, dataloaders, criterion, optimizer, mode):
        if mode == "train": hnet.train()
        elif mode == "val": hnet.eval()

        all_loss = []

        for (task_ids,y) in dataloaders[mode]:

            optimizer.zero_grad()

            # put data on GPU (if cuda)
            task_ids = task_ids.to(self._device)
            y = y.to(self._device)

            task_ids = task_ids.tolist()
            weights = hnet(cond_id=task_ids) 
            weights = torch.stack([torch.cat([w.flatten() for w in batch]) for batch in weights])
            loss = criterion(weights, y)

            # training
            if mode == "train":
                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())

            # validation
            elif mode == "val":
                with torch.no_grad():
                    all_loss.append(loss.item())
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del task_ids
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss)

    # run the entire training
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Pretrainer expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # build folder structure
        self._export_dir = os.path.join(self._logging_dir, self._export_path)
        os.makedirs(self._logging_dir, exist_ok=True)
        os.makedirs(self._export_dir, exist_ok=True)

        # data loading
        target_model_weights = self._target_model_weights
        model_paths = [(self._model.task_to_id(task), os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        train_ds = WeightDataset(model_paths, self.map_old_to_new_weights)
        self._dataloaders = {
            "train": DataLoader(train_ds, batch_size=self._batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(train_ds, batch_size=self._batch_size, shuffle=False, num_workers=4) # TODO: custom validation dataset
        }

        # self._evaluate_model_differences(target_model_weights)

        # sanity checks
        assert self._task_cnt == len(self._tasks)

        # check if the extpected output format of the HNET matches the labels/weights loaded from the original models
        new_shapes = self._model.mnet.get_cw_param_shapes()
        old_stud = stud.Student()
        old_shapes = self.map_old_to_new_weights({n:p.size() for n,p in old_stud.named_parameters()}, old_stud, verbose=self._verbose) 
        assert len(new_shapes) == len(old_shapes), f"HNET output generates {len(new_shapes)} weight tensors whereas we only loaded {len(old_shapes)} into the label from the original models!"
        for new,old in zip(new_shapes, old_shapes):
            assert new == old, "Mismatch between HNET output format and loaded model weight format. Make sure the order of parameters is the same!"
        

    # setup & run the entire training
    def execute(self):
        super().execute()
        
        export_path_best = os.path.join(self._export_dir, "best.pth")
        export_path_final = os.path.join(self._export_dir, "final.pth")

        model = self._model

        hnet = model.hnet
        hnet.to(self._device)

        epochs = self._epochs
        lr = self._lr
        lr_decay = self._lr_decay
        optimizer = torch.optim.Adam(hnet.parameters(), lr=lr)

        losses = {
            "MSELoss": torch.nn.MSELoss()
        }
        loss = losses[self._loss_fn]
        
        # report to wandb
        wandb.watch(hnet, loss, log=self._wandb_watch_log, log_freq=self._wandb_watch_log_freq)
        
        all_epochs = []
        smallest_loss = None

        # training loop
        for epoch in range(0, epochs):
            # decrease learning rate over time
            if epoch in self._decay_epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_decay
                lr = lr * lr_decay

            # train the networks
            loss_train = self._train_one_epoch(hnet, self._dataloaders, loss, optimizer, "train")
            if epoch % 25 == 0 and self._verbose: self.pretty_print_epoch(epoch, "train", loss_train, lr)

            # validate the networks
            loss_val = self._train_one_epoch(hnet, self._dataloaders, loss, optimizer, "val")
            if epoch % 25 == 0 and self._verbose: self.pretty_print_epoch(epoch, "val", loss_val, lr)

            ### REPORTING / STATS ###
            if epoch % 25 == 0:
                all_epochs.append([epoch, loss_train, loss_val]) 

                # if better performance than all previous => save weights as checkpoint
                is_best_model = smallest_loss is None or loss_val < smallest_loss
                if epoch % self._auto_checkpoint_steps == 0 or is_best_model:
                    checkpoint_dir = os.path.join(self._logging_dir, f"pretrain_checkpoint_in_epoch_{epoch}/")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    path = f"{checkpoint_dir}/{epoch}_{loss_val:f}.pth"

                    self.save(path, model, self._save_checkpoints_to_wandb)
                    
                    # overwrite the best model
                    if is_best_model:
                        smallest_loss = loss_val
                        self.save(export_path_best, model, self._save_checkpoints_to_wandb)
                
                # save/overwrite results at the end of each epoch
                stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "pretrain_all_results").replace("\\", "/")
                table = wandb.Table(data=all_epochs, columns=["Epoch", "Loss-Train", "Loss-Val"])
                wandb.log({
                        f"{self.name} - pretrain - epoch": epoch,
                        f"{self.name} - pretrain - loss train": loss_train,
                        f"{self.name} - pretrain - loss val": loss_val,
                        f"{self.name} - pretrain - learning rate": lr,
                        stats_file:table
                    })
        
        # save the final hypernetwork
        self.save(export_path_final, model, save_to_wandb=True)

        return model

    # saves the hnet to disk & wandb
    def save(self, path : str, model : HyperModel, save_to_wandb : bool = True):
        model.save(path)

        if save_to_wandb:
            wandb.save(path, base_path=wandb.run.dir)

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")

    def cleanup(self):
        super().cleanup()

        del self._dataloaders
