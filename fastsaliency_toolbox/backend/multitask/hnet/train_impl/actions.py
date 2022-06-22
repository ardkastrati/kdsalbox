from typing import List
import numpy as np

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

class FreezeHNETForCatchup(EpochAction):
    """ Freezes all but the task specific weights of the HNET at epoch 0 and unfreezes them again at a specified epoch """
    def __init__(self, unfreeze_at_epoch : int):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
    
    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_freeze = (epoch == 0)
        if should_freeze:
            trainer.model.hnet.freeze_hnet_for_catchup()
        
        should_unfreeze = (epoch == self._unfreeze_at_epoch)
        if should_unfreeze:
            trainer.model.hnet.freeze_hnet_for_catchup()

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