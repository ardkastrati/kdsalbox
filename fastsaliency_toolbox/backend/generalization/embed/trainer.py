import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from hypnettorch.hnets.mlp_hnet import HMLP
from hypnettorch.mnets import MLP

import wandb


from backend.utils import print_pretty_header
from backend.generalization.embed.model import Student

class Trainer(object):
    def __init__(self, conf):
        self._conf = conf

        train_parameter_map = conf['train_parameter_map']
        self._train_parameter_map = train_parameter_map
        self._logging_dir = train_parameter_map.get_val('logging_dir')
        self._verbose = train_parameter_map.get_val('verbose')
        self._batch_size = train_parameter_map.get_val('batch_size')
        self._export_path = train_parameter_map.get_val('export_path')
        self._freeze_encoder_steps = train_parameter_map.get_val('freeze_encoder_steps')

        self._preprocessing_parameter_map = conf["preprocessing_parameter_map"]
        self._gpu = str(conf["gpu"])
        
        train_folders_paths = self._conf["train_folders_paths"]
        val_folders_paths = self._conf["val_folders_paths"]
        self._task_cnt = self._conf["task_cnt"] # the amount of models we are currently learning

        train_datasets = [TrainDataManager(img_path, sal_path, self._verbose, self._preprocessing_parameter_map) for (img_path,sal_path) in train_folders_paths]
        val_datasets = [TrainDataManager(img_path, sal_path, self._verbose, self._preprocessing_parameter_map) for (img_path,sal_path) in val_folders_paths]

        self._dataloaders = {
            'train': [DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for ds in train_datasets],
            'val': [DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for ds in val_datasets],
        }

    # saves a hypernetwork and main network tuple to a given path under a given name
    def save(self, path, models):
        hnet, mnet = models

        # save on disk
        d = {}
        d["hnet_model"] = hnet.state_dict()
        d["mnet_model"] = mnet.state_dict()

        with torch.no_grad():
            print(mnet.decoder.conv10_2.weight)
        torch.save(d, path)

        # save to wandb
        artifact = wandb.Artifact("hnet_and_mnet", type="hnet_and_mnet")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    
    # saves a hypernetwork and main network tuple to a given path under a given name
    # and updates the smallest_loss and best_epoch values
    def save_weight(self, smallest_loss, best_epoch, best_models, loss_val, epoch, models, checkpoint_dir):
        path = f'{checkpoint_dir}/{epoch}_{loss_val:f}.pth'
        self.save(path, models)
        
        if smallest_loss is None or loss_val < smallest_loss:
            smallest_loss = loss_val
            best_epoch = epoch
            best_models = models
        return smallest_loss, best_epoch, best_models, models

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print('--------------------------------------------->>>>>>')
        print(f'Epoch {epoch}: loss {mode} {loss}, lr {lr}')
        print('--------------------------------------------->>>>>>')
    
    # train or evaluate one epoch for all models (mode in [train, val])
    # return loss, hnet, mnet
    def train_one(self, hnet, mnet, dataloaders, criterion, optimizer, mode):
        all_loss = []

        # defines which batch will be loaded from which model
        all_batches = np.concatenate([np.repeat(model_id, len(dataloader)) for (model_id,dataloader) in enumerate(dataloaders[mode])])
        np.random.shuffle(all_batches)

        # for each model
        data_iters = [iter(d) for d in dataloaders[mode]]
        for (i,model_id) in enumerate(all_batches):
            X,y = next(data_iters[model_id])

            optimizer.zero_grad()

            # put data on GPU (if cuda)
            if torch.cuda.is_available():
                X = X.cuda(torch.device("cuda", self._gpu))
                y = y.cuda(torch.device("cuda", self._gpu))

            weights = hnet(cond_id=model_id.item())
            pred = mnet.forward(X, weights=weights)
            loss = criterion(pred, y)

            # training
            if mode == 'train':
                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())

            # validation
            elif mode == 'val':
                with torch.no_grad():
                    all_loss.append(loss.item())

            # logging
            if i%100 == 0:
                print(f'Batch {i}: current accumulated loss {np.mean(all_loss)}')
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss), hnet, mnet 

    # run the entire training
    def start_train(self):
        if self._verbose: print("Encoder frozen...")

        # initialize networks
        mnet = Student()
        
        hnet = HMLP(
            mnet.external_param_shapes(), 
            layers=self._conf["hnet_hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
            cond_in_size=self._conf["hnet_embedding_size"], # the size of the embeddings
            num_cond_embs=self._task_cnt # the number of embeddings we want to learn
        )

        if torch.cuda.is_available():
            mnet = mnet.cuda(torch.device("cuda", self._gpu))
            hnet = hnet.cuda(torch.device("cuda", self._gpu))

        lr = self._conf["lr"]
        lr_decay = self._conf["lr_decay"]
        params = list(hnet.internal_params) + mnet.internal_params # learn the params of the hypernetwork as well as the internal params of the main network
        optimizer = torch.optim.Adam(params, lr=lr)
        loss = torch.nn.BCELoss()
        
        # report to wandb
        wandb.watch((hnet, mnet), loss, log="all", log_freq=10)
        
        all_epochs = []
        smallest_loss = None
        best_epoch = None
        best_model = (hnet,mnet) # the best combination of hypernetwork and mainnetwork so far

        # training loop
        for epoch in range(0, 40):
            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                mnet.unfreeze_encoder()

            # train the networks
            hnet.train()
            mnet.train()
            loss_train, hnet, mnet = self.train_one(hnet, mnet, self._dataloaders, loss, optimizer, 'train')
            # log
            if self._verbose: self.pretty_print_epoch(epoch, 'train', loss_train, lr)
            wandb.log({
                    "mode": "train",
                    "epoch": epoch,
                    "loss": loss_train,
                    "learning rate": lr
                })

            # validate the networks
            hnet.eval()
            mnet.eval()
            loss_val, hnet, mnet = self.train_one(hnet, mnet, self._dataloaders, loss, optimizer, 'val')
            # log
            if self._verbose: self.pretty_print_epoch(epoch, 'val', loss_val, lr)
            wandb.log({
                    "mode": "val",
                    "epoch": epoch,
                    "loss": loss_val,
                    "learning rate": lr
                })

            # if better performance than all previous => save weights as checkpoint
            if smallest_loss is None or loss_val < smallest_loss or epoch % 10 == 0:
                checkpoint_dir = os.path.join(self._logging_dir, 'checkpoint_in_epoch_{}/'.format(epoch))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                smallest_loss, best_epoch, best_model, (hnet, mnet) = self.save_weight(smallest_loss, best_epoch, best_model, loss_val, epoch, (hnet, mnet), checkpoint_dir)
            all_epochs.append([epoch, loss_train, loss_val]) 

            # decrease learning rate over time
            if epoch == 15 or epoch == 30 or epoch == 60:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                lr = lr * lr_decay
            
            # save results to CSV
            res_path = os.path.join(self._logging_dir, 'all_results.csv')
            np.savetxt(res_path, all_epochs, fmt='%s', delimiter=',', header='EPOCH,LOSS_TRAIN,LOSS_VAL', comments='')
            # save to wandb
            artifact = wandb.Artifact("all_results", type="result")
            artifact.add_file(res_path)
            wandb.log_artifact(artifact)

        # save the final best model
        if not os.path.exists(os.path.join(self._logging_dir, self._export_path)):
            os.makedirs(os.path.join(self._logging_dir, self._export_path))
        self.save(os.path.join(self._logging_dir, self._export_path, 'exported.pth'), best_model)

    # setup & run the entire training
    def execute(self):
        wandb.login()
        with wandb.init(project="kdsalbox-generalization", entity="ba-yanickz", config=self._conf):
            self._train_parameter_map.pretty_print()
            
            if self._verbose: print_pretty_header("TRAINING")
            if self._verbose: print("Trainer started...")

            self.start_train()
        
            if self._verbose: print("Done with training!")

    def delete(self):
        del self._dataloader
