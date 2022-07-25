import wandb
from backend.multitask.hnet.train_api.actions import StartAction
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.train_api.actions import EpochAction

class WatchWandb(StartAction):
    """ Watches a network and reports to wandb """
    def __init__(self, log : str, log_freq : int):
        super().__init__()

        self._log = log
        self._log_freq = log_freq

    def invoke(self, trainer: ATrainer):
        model = trainer.model
        loss_fn = trainer.loss_fn

        wandb.watch((model.hnet, model.mnet), loss_fn, log=self._log, log_freq=self._log_freq)

class ReportLiveMetricsWandb(EpochAction):
    """ Reports live metrics (loss values, lr etc) to wandb """
    def __init__(self, report_prefix : str, base_dir : str, log_freq : int = 1):
        super().__init__()

        self._base_dir = base_dir
        self._report_prefix = report_prefix
        self._log_freq = log_freq
    
    def invoke(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_invoke = (epoch % self._log_freq == 0)
        if not should_invoke: return

        train_loss = trainer.train_losses[-1]
        val_loss = trainer.val_losses[-1]

        lr = trainer.optimizer.param_groups[0]["lr"]

        wandb.log({
                f"{self._report_prefix} - epoch": epoch,
                f"{self._report_prefix} - loss train": train_loss,
                f"{self._report_prefix} - loss val": val_loss,
                f"{self._report_prefix} - learning rate": lr
            })