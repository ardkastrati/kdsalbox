import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import wandb

from backend.utils import print_pretty_header

from backend.multimodel.hyper_model import HyperModel

import backend.student as stud

class WeightDataset(Dataset):
    def __init__(self, paths, rearrange_weights_fn):
        self._models = [(task_id, self.load_model_weights(path, rearrange_weights_fn)) for task_id,path in paths]

    def __getitem__(self, index):
        return self._models[index] # returns tuple (task_id, weight tensor)
    
    def __len__(self):
        return len(self._models)

    def load_model_weights(self, path, rearrange_weights_fn):
        model = stud.student()
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['student_model'])
        
        named_weights = {n:p.data.flatten() for n,p in model.decoder.named_parameters()}
        weights = rearrange_weights_fn(named_weights)

        return torch.cat(weights)
    


class PreTrainer(object):
    '''
    Class that trains a hypernetwork to output some specific weights (e.g. to learn pretrained weights before the actual training begins)
    '''
    def __init__(self, conf, hyper_model : HyperModel):
        self._hyper_model = hyper_model

        pretrain_conf = conf["pretrain"]
        model_conf = conf["model"]
        wandb_conf = pretrain_conf["wandb"]

        # params
        batch_size = pretrain_conf["batch_size"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._logging_dir = pretrain_conf["logging_dir"]
        self._verbose = pretrain_conf["verbose"]
        self._export_path = pretrain_conf["export_path"]
        self._auto_checkpoint_steps = pretrain_conf["auto_checkpoint_steps"]

        self._tasks = pretrain_conf["tasks"]
        self._task_cnt = model_conf["hnet"]["task_cnt"]

        self._loss_fn = pretrain_conf["loss"]
        self._epochs = pretrain_conf["epochs"]
        self._lr = pretrain_conf["lr"]
        self._lr_decay = pretrain_conf["lr_decay"]
        self._decay_epochs = pretrain_conf["decay_epochs"]

        self.wandb_watch_log = wandb_conf["watch"]["log"]
        self.wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]

        # data loading
        target_model_weights = pretrain_conf["target_model_weights"]
        model_paths = [(hyper_model.task_to_id(task), os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        train_ds = WeightDataset(model_paths, self.map_old_to_new_weights)
        self._dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4) # TODO: custom validation dataset
        }

        # self._evaluate_model_differences(target_model_weights)

        # sanity checks
        assert self._task_cnt == len(self._tasks)
    
    # evaluate how different the layers of the models are
    def _evaluate_model_differences(self, target_model_weights):
        model_p = [(task, os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        model_weights = []
        for t,p in model_p:
            model = stud.student()
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

    def map_old_to_new_weights(self, named_weights):
        selection = [
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
        for s in selection:
            new_weights.append(named_weights[s])

        return new_weights

    # train or evaluate one epoch for all tasks (mode in [train, val])
    # return loss
    def train_one(self, hnet, dataloaders, criterion, optimizer, mode):
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
    def start_train(self):
        # build folder structure
        export_dir = os.path.join(self._logging_dir, self._export_path)
        export_path_best = os.path.join(export_dir, "best.pth")
        export_path_final = os.path.join(export_dir, "final.pth")
        os.makedirs(self._logging_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)

        # initialize networks
        model = self._hyper_model
        model.build()

        # new_shapes = model.mnet.get_cw_param_shapes()
        # old_shapes = self.map_old_to_new_weights({n:p.size() for n,p in stud.student().decoder.named_parameters()}) 
        # for new,old in zip(new_shapes, old_shapes):
        #     if new != old:
        #         print("shit")

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
        wandb.watch(hnet, loss, log=self.wandb_watch_log, log_freq=self.wandb_watch_log_freq)
        
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
            loss_train = self.train_one(hnet, self._dataloaders, loss, optimizer, "train")
            if epoch % 25 == 0 and self._verbose: self.pretty_print_epoch(epoch, "train", loss_train, lr)

            # validate the networks
            loss_val = self.train_one(hnet, self._dataloaders, loss, optimizer, "val")
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

                    self.save(path, model)
                    
                    # overwrite the best model
                    if is_best_model:
                        smallest_loss = loss_val
                        self.save(export_path_best, model)
                
                # save/overwrite results at the end of each epoch
                stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "pretrain_all_results").replace("\\", "/")
                table = wandb.Table(data=all_epochs, columns=["Epoch", "Loss-Train", "Loss-Val"])
                wandb.log({
                        "pretrain - epoch": epoch,
                        "pretrain - loss train": loss_train,
                        "pretrain - loss val": loss_val,
                        "pretrain - learning rate": lr,
                        stats_file:table
                    })
        
        # save the final hnet
        self.save(export_path_final, model)

    # setup & run the entire training
    def execute(self):
        if self._verbose: print_pretty_header("PRE-TRAINING")
        if self._verbose: print("Pretrainer started...")

        self.start_train()
    
        if self._verbose: print("Done with pretraining!")

    # saves the hnet to disk & wandb
    def save(self, path, model):
        model.save(path)
        wandb.save(path, base_path=wandb.run.dir)

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")

    def delete(self):
        del self._dataloaders
