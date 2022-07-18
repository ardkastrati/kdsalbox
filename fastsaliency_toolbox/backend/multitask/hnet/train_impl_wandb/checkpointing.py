import os

import wandb

from backend.multitask.hnet.train_api.checkpointing import Checkpointer
from backend.multitask.hnet.models.hyper_model import HyperModel
from backend.multitask.hnet.train_api.training import ATrainer

class CheckpointerWandb(Checkpointer):
    """
    Saves the model to the specified location and optionally saves checkpoints to wandb aswell.

    Creates a checkpoint iff:
        - epoch is a multiple of auto_checkpoint_freq
        - current epoch has the smallest loss value 
          & the last checkpoint is at least max_freq epochs back
    """
    def __init__(self, auto_checkpoint_freq : int, base_checkpoint_dir : str, save_to_wandb : bool, max_freq : int = 1):
        super().__init__()

        self._auto_checkpoint_freq = auto_checkpoint_freq
        self._base_checkpoint_dir = base_checkpoint_dir
        self._save_to_wandb = save_to_wandb
        self._max_freq = max_freq
        self._last_checkpoint = -1
    
    
    def _is_best_model(self, trainer : ATrainer) -> bool:
        val_losses = trainer.val_losses
        cur_loss = val_losses[-1]
        is_smallest_loss = True
        for loss in val_losses:
            if loss < cur_loss:
                is_smallest_loss = False
                break
        
        return is_smallest_loss

    def _should_do_checkpoint(self, trainer: ATrainer) -> bool:
        epoch = trainer.epoch
        is_checkpoint_epoch = epoch % self._auto_checkpoint_freq == 0
        if is_checkpoint_epoch: return True
        
        if epoch - self._last_checkpoint < self._max_freq: return False

        is_smallest_loss = self._is_best_model(trainer)
        if is_smallest_loss: return True

        return False

    def try_make_checkpoint(self, trainer: ATrainer) -> bool:
        model = trainer.model

        # always overwrite the model if its better than any previous
        if self._is_best_model(trainer):
            best_path = os.path.join(self._base_checkpoint_dir, "best.pth")
            self.save(best_path, model, self._save_to_wandb)

        # check if an actual checkpoint should be created
        if self._should_do_checkpoint(trainer):
            epoch = trainer.epoch
            loss_val = trainer.val_losses[-1]

            path = os.path.join(self._base_checkpoint_dir, 
                f"{epoch}_{loss_val:f}.pth")

            if not self._save_to_wandb:
                path = os.path.join(path, "ignore_wandb")

            self.save(path, model, self._save_to_wandb)
        
            self._last_checkpoint = epoch


    def restore_best(self, trainer: ATrainer) -> HyperModel:
        model = trainer.model
        
        best_path = os.path.join(self._base_checkpoint_dir, "best.pth")
        model.load(best_path, model.device)
        

    def save(self, path : str, model : HyperModel, save_to_wandb : bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model.save(path)
        if save_to_wandb:
            wandb.save(path, base_path=wandb.run.dir)
