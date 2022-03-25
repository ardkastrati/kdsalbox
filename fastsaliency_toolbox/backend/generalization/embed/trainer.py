import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from backend.generalization.datasets import DataManager
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

        ds_train, ds_validate = self.get_datamanagers()
        
        self._dataloader = {
            'train': DataLoader(ds_train, batch_size=self._batch_size, shuffle=False, num_workers=4),
            'val': DataLoader(ds_validate, batch_size=self._batch_size, shuffle=False, num_workers=4)
        }

    def get_datamanagers(self):
        if self._verbose:
            print("Train setup:")
        ds_train = DataManager(self._conf["train_folders_paths"], self._verbose, self._preprocess_parameter_map)

        if self._verbose:
            print("Validation setup:")
        ds_validate = DataManager(self._conf["val_folders_paths"], self._verbose, self._preprocess_parameter_map)

        return (ds_train, ds_validate)

    def save(self, path, model):
        d = {}
        d['student_model'] = model.state_dict()
        torch.save(d, path)
        
    def save_weight(self, smallest_val, best_epoch, best_model, loss_val, epoch, model, checkpoint_dir):
        path = '{}/{}_{:f}.pth'.format(checkpoint_dir, epoch, loss_val)
        self.save(path, model)
        
        if smallest_val is None or loss_val < smallest_val:
            smallest_val = loss_val
            best_epoch = epoch
            best_model = model
        return smallest_val, best_epoch, best_model, model

    def pretty_print_epoch(self, epoch, mode, loss, lr):
        print('--------------------------------------------->>>>>>')
        print('Epoch {}: loss {} {}, lr {}'.format(epoch, mode, loss, lr))
        print('--------------------------------------------->>>>>>')
    
    # train one batch
    def train_one(self, model, dataloader, criterion, optimizer, mode):
        all_loss = []
        
        for i, (X, y) in enumerate(dataloader[mode]):   
            optimizer.zero_grad()

            # put data on GPU (if cuda)
            if torch.cuda.is_available():
                X = X.cuda(torch.device(self._gpu))
                y = y.cuda(torch.device(self._gpu))
            pred = model.forward(X)
            losses = criterion(pred, y)

            # training
            if mode == 'train':
                losses.backward()
                optimizer.step()
                all_loss.append(losses.item())

            # validation
            elif mode == 'val':
                with torch.no_grad():
                    all_loss.append(losses.item())

            # loggin
            if i%100 == 0:
                print('Batch {}: current accumulated loss {}'.format(i, np.mean(all_loss)))
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss), model 

    # run the entire training
    def start_train(self):
        if self._verbose: print("Encoder frozen...")

        student = Student()
        loss = torch.nn.BCELoss()

        # report to wandb
        wandb.watch(student, loss, log="all", log_freq=10)

        lr = 0.01
        lr_decay = 0.1
        optimizer = torch.optim.Adam(list(student.parameters()), lr=lr)
        all_epochs = []

        smallest_val = None
        best_epoch = None
        best_model = student
        for epoch in range(0, 40):
            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                student.unfreeze_encoder()

            # train the student
            student.train()
            loss_train, student = self.train_one(student, self._dataloader, loss, optimizer, 'train')
            if self._verbose: self.pretty_print_epoch(epoch, 'train', loss_train, lr)

            # validate on the student
            student.eval()
            loss_val, model = self.train_one(student, self._dataloader, loss, optimizer, 'val')
            if self._verbose: self.pretty_print_epoch(epoch, 'val', loss_val, lr)

            # if better performance than all previous => save weights as checkpoint
            if smallest_val is None or loss_val < smallest_val or epoch % 10 == 0:
                checkpoint_dir = os.path.join(self._logging_dir, 'checkpoint_in_epoch_{}/'.format(epoch))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                smallest_val, best_epoch, best_model, student = self.save_weight(smallest_val, best_epoch, best_model, loss_val, epoch, student, checkpoint_dir)
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