import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from hypnettorch.hnets.mlp_hnet import HMLP

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

        self._preprocess_parameter_map = conf["preprocess_parameter_map"]
        self._gpu = str(conf["gpu"])
        # TODO: is this correct?
        self._device = self._gpu if torch.cuda.is_available() else 'cpu'
        
        train_folders_paths = self._conf["train_folders_paths"]
        val_folders_paths = self._conf["val_folders_paths"]
        self._model_cnt = len(train_folders_paths) # the amount of models we are currently learning

        train_datasets = [TrainDataManager(img_path, sal_path, self._verbose, self._preprocess_parameter_map) for (img_path,sal_path) in train_folders_paths]
        val_datasets = [TrainDataManager(img_path, sal_path, self._verbose, self._preprocess_parameter_map) for (img_path,sal_path) in val_folders_paths]

        self._dataloaders = {
            'train': [DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for ds in train_datasets],
            'val': [DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for ds in val_datasets],
        }

    # saves a hypernetwork and main network tuple to a given path under a given name
    def save(self, path, models):
        hnet, mnet = models
        d = {}
        d["hnet_model"] = hnet.state_dict()
        d["mnet_model"] = mnet.state_dict()
        torch.save(d, path)
    
    # saves a hypernetwork and main network tuple to a given path under a given name
    # and updates the smallest_loss and best_epoch values
    def save_weight(self, smallest_loss, best_epoch, best_models, loss_val, epoch, models, checkpoint_dir):
        path_hnet = f'{checkpoint_dir}/{epoch}_{loss_val:f}.pth'
        self.save(path_hnet, models)
        
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

        # for each model
        for model_id,dataloader in enumerate(dataloaders[mode]): # TODO: try different approach than just simply learning the models 1 by 1
            print(f"Model {model_id}")

            # for each batch
            for i, (X, y) in enumerate(dataloader):   
                print(f"Batch {i}")
                optimizer.zero_grad()

                # put data on GPU (if cuda)
                if torch.cuda.is_available():
                    X = X.cuda(torch.device(self._gpu))
                    y = y.cuda(torch.device(self._gpu))

                weights = hnet(cond_id=model_id)
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
                if i%25 == 0:
                    wandb.log({
                        "model": model_id,
                        "batch": i,
                        "loss": np.mean(all_loss)
                    })
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

        # initialize hypernetwork
        mnet = Student() # TODO: student should implement mainnet interface

        # TODO: get params from config
        hnet = HMLP(
            mnet.param_shapes, 
            layers=[100, 100], # the sizes of the hidden layers
            cond_in_size=16, # the size of the embeddings
            num_cond_embs=self._model_cnt # the number of embeddings we want to learn
        ).to(self._device)

        lr = 0.01
        lr_decay = 0.1
        optimizer = torch.optim.Adam(list(student.parameters()), lr=lr)
        loss = torch.nn.BCELoss()
        
        # report to wandb
        wandb.watch(student, loss, log="all", log_freq=10)
        
        all_epochs = []
        smallest_loss = None
        best_epoch = None
        best_model = (hnet,mnet) # the best combination of hypernetwork and mainnetwork so far

        # training loop
        for epoch in range(0, 40):
            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                student.unfreeze_encoder()

            # train the networks
            hnet.train()
            mnet.train()
            loss_train, hnet, mnet = self.train_one(hnet, mnet, self._dataloaders, loss, optimizer, 'train')
            if self._verbose: self.pretty_print_epoch(epoch, 'train', loss_train, lr)

            # validate the networks
            hnet.eval()
            mnet.eval()
            loss_val, hnet, mnet = self.train_one(hnet, mnet, self._dataloaders, loss, optimizer, 'val')
            if self._verbose: self.pretty_print_epoch(epoch, 'val', loss_val, lr)

            # if better performance than all previous => save weights as checkpoint
            if smallest_loss is None or loss_val < smallest_loss or epoch % 10 == 0:
                checkpoint_dir = os.path.join(self._logging_dir, 'checkpoint_in_epoch_{}/'.format(epoch))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                smallest_loss, best_epoch, best_model, student = self.save_weight(smallest_loss, best_epoch, best_model, loss_val, epoch, student, checkpoint_dir)
            all_epochs.append([epoch, loss_train, loss_val]) 

            # decrease learning rate over time
            if epoch == 15 or epoch == 30 or epoch == 60:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                lr = lr * lr_decay
            
            # save results to CSV
            np.savetxt(os.path.join(self._logging_dir, 'all_results.csv'), all_epochs, fmt='%s', delimiter=',', header='EPOCH,LOSS_TRAIN,LOSS_VAL', comments='')

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
