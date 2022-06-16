"""
Trainer
-------

Does train a student on all original images in a folder using all saliency images in a different folder. 
Checkout the TrainDataLoader documentation for further details.

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .datasets import TrainDataManager
from .utils import print_pretty_header

class Trainer(object):
    def __init__(self,
                 model_manager,
                 train_parameter_map, preprocess_parameter_map, gpu=0):


        self._model = model_manager.get_matching(train_parameter_map.get_val('model'))
        self._logging_dir = train_parameter_map.get_val('logging_dir')
        self._input_images = train_parameter_map.get_val('input_images')
        self._input_saliencies = train_parameter_map.get_val('input_saliencies')
        self._recursive = train_parameter_map.get_val('recursive')
        self._verbose = train_parameter_map.get_val('verbose')
        self._batch_size = train_parameter_map.get_val('batch_size')
        self._freeze_encoder_steps = train_parameter_map.get_val('batch_size')
        self._export_path = train_parameter_map.get_val('export_path')
        self._preprocess_parameter_map = preprocess_parameter_map
        self._gpu = str(gpu)

        if self._verbose:
            print("Train setup:")
        ds_train = TrainDataManager(os.path.join(self._input_images, 'train'), self._input_saliencies, self._verbose, self._preprocess_parameter_map)
        if self._verbose:
            print("Validation setup:")
        ds_validate = TrainDataManager(os.path.join(self._input_images, 'val'), self._input_saliencies, self._verbose, self._preprocess_parameter_map)
        self._dataloader = {
            'train': DataLoader(ds_train, batch_size=self._batch_size, shuffle=False, num_workers=4),
            'val': DataLoader(ds_validate, batch_size=self._batch_size, shuffle=False, num_workers=4)
        }

    def save(self, path, model):
        d = {}
        d['student_model'] = model.state_dict()
        torch.save(d, path)
        #if optimizer:
        #    d['optimizer'] = optimizer.state_dict()
        #t.save(d, path)
        
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

    
    def memory_check(self, position=None):
        print(position)
        for i in range(8):
            print(torch.cuda.memory_reserved(i))
            print(torch.cuda.memory_allocated(i))
            print("")
    
    def train_one(self, model, dataloader, optimizer, mode):
        all_loss, all_NSS, all_CC, all_SIM = [], [], [], []
        my_loss = torch.nn.BCELoss()
        # my_loss = lambda s_map1, s_map2: kldiv(s_map1, s_map2) - nss(s_map1, s_map2) - cc(s_map1, s_map2)
        
        for i, (X, y) in enumerate(dataloader[mode]):   
            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.cuda(torch.device(self._gpu))
                y = y.cuda(torch.device(self._gpu))
            pred = model.forward(X)
            losses = my_loss(pred, y)

            if mode == 'train':
                losses.backward()
                optimizer.step()
                all_loss.append(losses.item())

            elif mode == 'val':
                with torch.no_grad():
                    all_loss.append(losses.item())

            if i%100 == 0:
                print('Batch {}: current accumulated loss {}'.format(i, np.mean(all_loss)))
            
            # Remove batch from gpu
            if torch.cuda.is_available():
                del X
                del y
                torch.cuda.empty_cache()
                
        return np.mean(all_loss), model 


    def start_train(self):
        if self._verbose: print("Encoder frozen...")
        student = self._model.get_student()

        #def count_parameters(model):
        #    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(count_parameters(student))
        #print(count_parameters(student.encoder))

        lr = 0.01
        lr_decay = 0.1
        optimizer = torch.optim.Adam(list(student.parameters()), lr=lr)
        all_epochs = []

        smallest_val = None
        best_epoch = None
        best_model = student
        for epoch in range(0, 40, 1):
            if epoch == 3:
                if self._verbose: print("Encoder unfrozen")
                student.unfreeze_encoder()
            student.train()
            loss_train, student = self.train_one(student, self._dataloader, optimizer, 'train')
            if self._verbose: self.pretty_print_epoch(epoch, 'train', loss_train, lr)

            student.eval()
            loss_val, model = self.train_one(student, self._dataloader, optimizer, 'val')
            if self._verbose: self.pretty_print_epoch(epoch, 'val', loss_val, lr)

            if smallest_val is None or loss_val < smallest_val or epoch % 10 == 0:
                checkpoint_dir = os.path.join(self._logging_dir, 'checkpoint_in_epoch_{}/'.format(epoch))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                smallest_val, best_epoch, best_model, student = self.save_weight(smallest_val, best_epoch, best_model, loss_val, epoch, student, checkpoint_dir)
            all_epochs.append([epoch, loss_train, loss_val]) 

            if epoch == 15 or epoch == 30 or epoch == 60:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                lr = lr * lr_decay
            np.savetxt(os.path.join(self._logging_dir, 'all_results.csv'), all_epochs, fmt='%s', delimiter=',', header='EPOCH,LOSS_TRAIN,LOSS_VAL', comments='')
        if not os.path.exists(os.path.join(self._logging_dir, self._export_path)):
            os.makedirs(os.path.join(self._logging_dir, self._export_path))
        self.save(os.path.join(self._logging_dir, self._export_path, 'exported.pth'), best_model)

    def execute(self):
        if self._verbose: print_pretty_header("TRAINING " + self._model.name)
        if self._verbose: print("Trainer started...")
        self.start_train()
        if self._verbose: print("Done with {}!".format(self._model.name))

    def delete(self):
        del self._dataloader

if __name__ == '__main__':
    from .pseudomodels import ModelManager
    m = ModelManager('models/', verbose=True, pretrained=False)
    print(m._model_map)

    from .config import Config
    c = Config('config.json')
    c.train_parameter_map.pretty_print()

    t = Trainer(m, c.train_parameter_map, c.preprocessing_parameter_map, gpu=0)
    t.execute()



