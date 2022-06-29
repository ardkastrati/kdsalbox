from math import ceil
from typing import List
import numpy as np
import torch

from backend.multitask.hnet.train_api.actions import EpochAction
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.train_api.actions import BatchAction

class LrDecay(EpochAction):
    """ Lets the learning rate decay by a given factor in the specified epochs """
    def __init__(self, lr_decay : float, decay_epochs : List[int]):
        super().__init__()

        self._lr_decay = lr_decay
        self._decay_epochs = decay_epochs

    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch
        optimizer = trainer.optimizer

        should_decay = epoch in self._decay_epochs
        if should_decay:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self._lr_decay

class FreezeEncoder(EpochAction):
    """ Freezes the MNET encoder at epoch 0 and unfreezes it again at a specified epoch """
    def __init__(self, unfreeze_at_epoch : int):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
    
    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_freeze = (epoch == 0)
        if should_freeze:
            trainer.model.mnet.freeze_encoder()
        
        should_unfreeze = (epoch == self._unfreeze_at_epoch)
        if should_unfreeze:
            trainer.model.mnet.unfreeze_encoder()

class FreezeHNETShared(EpochAction):
    """ Freezes all but the task specific weights of the HNET at epoch 0 and unfreezes them again at a specified epoch """
    def __init__(self, unfreeze_at_epoch : int):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
    
    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_freeze = (epoch == 0)
        if should_freeze:
            # TODO: set lr of all shared params to 0
            print("should_freeze")
        
        should_unfreeze = (epoch == self._unfreeze_at_epoch)
        if should_unfreeze:
            # TODO: set lr of all shared params to <lr>
            print("should_unfreeze")

class LoadModel(EpochAction):
    """ Loads a model from a specific path """
    def __init__(self, path : str):
        super().__init__()

        self._path = path
    
    def invoke(self, trainer: ATrainer):
        model = trainer.model
        model.load(self._path)

class LogEpochLosses(EpochAction):
    """ Logs the training loss and validation loss at the end of an epoch """
    def __init__(self, log_freq : int = 1):
        super().__init__()

        self._log_freq = log_freq

    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch
        
        should_log = (epoch % self._log_freq == 0)
        if not should_log: return

        train_loss = trainer.train_losses[-1]
        val_loss = trainer.val_losses[-1]

        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}:")
        print(f"\t Train loss = {train_loss}")
        print(f"\t Val loss   = {val_loss}")
        print("--------------------------------------------->>>>>>", flush=True)

class BatchLogger(BatchAction):
    """ Prints the loss every couple of epochs """
    def __init__(self, log_freq : int):
        super().__init__()

        self._log_freq = log_freq

    def invoke(self, trainer: ATrainer, mode: str, batch_losses: List[float], total_batches : int):
        batch_index = len(batch_losses) - 1
        
        should_log = (batch_index % self._log_freq == 0)
        if should_log:
            print(f"[{mode}] Batch {batch_index}/{total_batches}: current accumulated loss {np.mean(batch_losses)}", flush=True)


class WeightWatcher(BatchAction):
    """ Prints the average absolute gradient on all weights of the last layer of the HNET per task """
    def __init__(self, log_freq : int, groups : int = None):
        super().__init__()

        self._log_freq = log_freq
        self._groups = groups

    def invoke(self, trainer: ATrainer, mode: str, batch_losses: List[float], total_batches : int):
        batch_index = len(batch_losses) - 1
        
        should_log = (batch_index % self._log_freq == 0) and mode == "train" # grads will be 0 during val
        if not should_log: return

        grads_per_task = trainer.model.hnet.get_gradients_on_outputs()

        for task_id in grads_per_task.keys():
            grads_per_target : List[torch.Tensor] = grads_per_task[task_id]

            grad_means = [grads.abs().mean().item() for grads in grads_per_target]

            # if groups then aggregate into n almost same sized groups
            if self._groups:
                split_size = ceil(len(grad_means) / self._groups)
                means_per_group : List[torch.Tensor] = list(torch.tensor(np.array(grad_means)).split(split_size))
                mean_per_group = [means.mean() for means in means_per_group]
                
                grad_means = mean_per_group
            
            sum = np.sum(np.array([grads.abs().sum().item() for grads in grads_per_target]))

            print(trainer.optimizer.param_groups)

            print(f"Task {task_id} (abs sum = {sum}) : {', '.join([f'{mean:.3f}' for mean in grad_means])}")
        