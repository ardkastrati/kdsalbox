import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

from backend.utils import print_pretty_header
from backend.datasets import TrainDataManager
from backend.parameters import ParameterMap

class HyperTrainer(object):
    '''
    Class that trains a hypernetwork & main network architecture and reports to wandb
    '''
    def __init__(self, conf, hyper_model):
        self._hyper_model = hyper_model

        train_conf = conf["train"]
        model_conf = conf["model"]
        preprocess_conf = conf["preprocess"]

        # params
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._logging_dir = train_conf["logging_dir"]
        self._verbose = train_conf["verbose"]
        self._export_path = train_conf["export_path"]

        tasks = train_conf["tasks"]
        assert model_conf["task_cnt"] == len(tasks)
        self._task_cnt = model_conf["task_cnt"]

        batch_size = train_conf["batch_size"]
        self._epochs = train_conf["epochs"]
        self._lr = train_conf["lr"]
        self._lr_decay = train_conf["lr_decay"]
        self._freeze_encoder_steps = train_conf["freeze_encoder_steps"]

        # convert to preprocess params
        preprocess_parameter_map = ParameterMap()
        preprocess_parameter_map.set_from_dict(preprocess_conf)

        # data loading
        input_saliencies = train_conf["input_saliencies"]
        train_img_path = train_conf["input_images_train"]
        val_img_path = train_conf["input_images_val"]
        sal_folders = [os.path.join(input_saliencies, model) for model in tasks] # path to saliency folder for all models

        train_datasets = [TrainDataManager(train_img_path, sal_path, self._verbose, preprocess_parameter_map) for sal_path in sal_folders]
        val_datasets = [TrainDataManager(val_img_path, sal_path, self._verbose, preprocess_parameter_map) for sal_path in sal_folders]

        self._dataloaders = {
            "train": {model:DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4) for (model,ds) in zip(tasks, train_datasets)},
            "val": {model:DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4) for (model,ds) in zip(tasks, val_datasets)},
        }
    
    # train or evaluate one epoch for all models (mode in [train, val])
    # return loss, model
    def train_one(self, model, dataloaders, criterion, optimizer, mode):
        all_loss = []

        # defines which batch will be loaded from which task/model
        all_batches = np.concatenate([np.repeat(model.task_to_id(task), len(dataloader)) for (task,dataloader) in dataloaders[mode].items()])
        np.random.shuffle(all_batches)

        # TODO: TEMP
        all_batches = all_batches[:1]

        # for each model
        data_iters = [iter(d) for d in dataloaders[mode].values()]
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
                print(f"Batch {i}: current accumulated loss {np.mean(all_loss)}", flush=True)
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss), model

    # run the entire training
    def start_train(self):
        if not os.path.exists(self._logging_dir):
            os.makedirs(self._logging_dir)

        if self._verbose: print("Encoder frozen...")

        # initialize networks
        model = self._hyper_model
        model.build()
        model.to(self._device)

        epochs = self._epochs
        lr = self._lr
        lr_decay = self._lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = torch.nn.BCELoss()
        
        # report to wandb
        wandb.watch((model.hnet, model.mnet), loss, log="all", log_freq=1)
        
        all_epochs = []
        smallest_loss = None
        best_epoch = None
        best_model = model

        # training loop
        for epoch in range(0, epochs):
            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                model.mnet.unfreeze_encoder()

            # train the networks
            model.train()
            loss_train, model = self.train_one(model, self._dataloaders, loss, optimizer, "train")

            # log
            if self._verbose: self.pretty_print_epoch(epoch, "train", loss_train, lr)
            wandb.log({
                    "mode": "train",
                    "epoch": epoch,
                    "loss": loss_train,
                    "learning rate": lr
                })


            # validate the networks
            model.eval()
            loss_val, model = self.train_one(model, self._dataloaders, loss, optimizer, "val")

            # log
            if self._verbose: self.pretty_print_epoch(epoch, "val", loss_val, lr)
            wandb.log({
                    "mode": "val",
                    "epoch": epoch,
                    "loss": loss_val,
                    "learning rate": lr
                })

            # if better performance than all previous => save weights as checkpoint
            if smallest_loss is None or loss_val < smallest_loss or epoch % 10 == 0:
                checkpoint_dir = os.path.join(self._logging_dir, f"checkpoint_in_epoch_{epoch}/")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                smallest_loss, best_epoch, best_model, model = self.save_weight(smallest_loss, best_epoch, best_model, loss_val, epoch, model, checkpoint_dir)
            all_epochs.append([epoch, loss_train, loss_val]) 

            # decrease learning rate over time
            if epoch == 15 or epoch == 30 or epoch == 60:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_decay
                lr = lr * lr_decay
            
            # save results to CSV
            res_path = os.path.join(self._logging_dir, "all_results.csv")
            np.savetxt(res_path, all_epochs, fmt="%s", delimiter=",", header="EPOCH,LOSS_TRAIN,LOSS_VAL", comments="")
            # save to wandb
            artifact = wandb.Artifact("all_results", type="result")
            artifact.add_file(res_path)
            wandb.log_artifact(artifact)

        # save the final best model
        if not os.path.exists(os.path.join(self._logging_dir, self._export_path)):
            os.makedirs(os.path.join(self._logging_dir, self._export_path))
            
        if self._verbose: print(f"Save best model to {os.path.join(self._logging_dir, self._export_path, 'best.pth')}")
        self.save(os.path.join(self._logging_dir, self._export_path, "best.pth"), best_model)

    # setup & run the entire training
    def execute(self):
        if self._verbose: print_pretty_header("TRAINING")
        if self._verbose: print("Trainer started...")

        self.start_train()
    
        if self._verbose: print("Done with training!")

    # saves a hypernetwork and main network tuple to a given path under a given name
    def save(self, path, model):
        # save on disk
        model.save(path)

        # save to wandb
        artifact = wandb.Artifact("hnet_and_mnet", type="hnet_and_mnet")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    
    # saves a hypernetwork and main network tuple to a given path under a given name
    # and updates the smallest_loss and best_epoch values
    def save_weight(self, smallest_loss, best_epoch, best_model, loss_val, epoch, model, checkpoint_dir):
        path = f"{checkpoint_dir}/{epoch}_{loss_val:f}.pth"
        self.save(path, model)
        
        if smallest_loss is None or loss_val < smallest_loss:
            smallest_loss = loss_val
            best_epoch = epoch
            best_model = model
        return smallest_loss, best_epoch, best_model, model

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")

    def delete(self):
        del self._dataloaders
