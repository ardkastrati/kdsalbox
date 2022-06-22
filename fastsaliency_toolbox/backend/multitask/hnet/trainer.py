"""
ATrainer
-----------------

Abstract baseclass that provides the framework for any trainer

"""

from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.datasets import RunDataManager
from backend.parameters import ParameterMap
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.train_impl.actions import LogEpochLosses, BatchLogger, FreezeEncoder, LrDecay
from backend.multitask.hnet.train_impl.training import Trainer
from backend.multitask.hnet.train_impl_wandb.actions import WatchWandb, ReportLiveMetricsWandb
from backend.multitask.hnet.train_impl_wandb.checkpointing import CheckpointerWandb
from backend.multitask.hnet.train_impl_wandb.progress_tracking import RunProgressTrackerWandb
from backend.multitask.hnet.train_api.training import TrainStep
from backend.multitask.hnet.train_impl.training import MultitaskTrainStep

# try to get a parameter value, if it doesnt exist then return default
def _get(conf : Dict, key, default = None):
    if key in conf.keys():
        return conf[key]
    return default

class ATrainer(AStage, ABC):
    def __init__(self, conf, name, verbose):
        AStage.__init__(self, name=name, verbose=verbose)
        ABC.__init__(self)

        self._model : HyperModel = None

        train_conf = conf[self._name]

        self._batch_size = train_conf["batch_size"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._auto_checkpoint_steps = train_conf["auto_checkpoint_steps"]
        self._input_images_run = train_conf["input_images_run"]

        self._tasks = conf["tasks"]
        self._task_cnt = conf["model"]["hnet"]["task_cnt"]

        self._loss_fn = train_conf["loss"]
        self._epochs = train_conf["epochs"]
        self._lr = train_conf["lr"]
        self._lr_decay = train_conf["lr_decay"]
        self._decay_epochs = train_conf["decay_epochs"]

        self._batch_log_freq = _get(train_conf, "batch_log_freq")
        self._log_freq = _get(train_conf, "log_freq", 1)

        wandb_conf = train_conf["wandb"]
        self._wandb_watch_log = wandb_conf["watch"]["log"]
        self._wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]
        self._save_checkpoints_to_wandb = wandb_conf["save_checkpoints_to_wandb"]

        # convert parameter dicts to parametermap such that it can be used in process()
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])

    @abstractmethod
    def get_data_providers(self) -> Dict[str, DataProvider]:
        pass

    def get_losses(self) -> Dict[str, nn.Module]:
        return {
            "BCELoss": torch.nn.BCELoss(),
            "L1Loss": torch.nn.L1Loss(),
            "MSELoss": torch.nn.MSELoss(),
            "HuberLoss": torch.nn.HuberLoss(),
        }
    
    @abstractmethod
    def get_stepper(self) -> TrainStep:
        pass

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Trainer expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # prepare trainer
        self._dataproviders = self.get_data_providers()
        self._run_dataloader = DataLoader(RunDataManager(self._input_images_run, "", verbose=False, recursive=False), batch_size=1)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        losses = self.get_losses()
        loss_fn = losses[self._loss_fn]
        stepper = self.get_stepper()

        self._trainer = Trainer(self._epochs, self._model, optimizer, loss_fn, stepper, self._dataproviders)\
            .set_checkpointer(CheckpointerWandb(self._auto_checkpoint_steps, self._logging_dir, self._save_checkpoints_to_wandb, max_freq=self._log_freq))\
            .add_progress_tracker(RunProgressTrackerWandb(self._run_dataloader, self._postprocess_parameter_map, report_prefix=self._name, log_freq=self._log_freq))\
            .add_start_action(WatchWandb(self._wandb_watch_log, self._wandb_watch_log_freq))\
            .add_epoch_start_action(LrDecay(self._lr_decay, self._decay_epochs))\
            .add_epoch_end_action(LogEpochLosses(self._log_freq))\
            .add_epoch_end_action(ReportLiveMetricsWandb(self._name, self._logging_dir, self._log_freq))
        
        if self._batch_log_freq:
            self._trainer.add_batch_action(BatchLogger(self._batch_log_freq))

        # sanity checks
        assert self._task_cnt == len(self._tasks)

    def execute(self):
        super().execute()

        model = self._model
        model.build()
        model.to(self._device)

        self._trainer.train()

        return model
    
    def cleanup(self):
        super().cleanup()
        del self._dataproviders
        del self._run_dataloader

class ASaliencyTrainer(ATrainer):
    """ Basic trainer for training with input images and saliency images """
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[self._name]
        self._input_saliencies = train_conf["input_saliencies"]
        self._train_img_path = train_conf["input_images_train"]
        self._val_img_path = train_conf["input_images_val"]

        self._freeze_encoder_steps = train_conf["freeze_encoder_steps"]

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        
        self._trainer.add_epoch_start_action(FreezeEncoder(self._freeze_encoder_steps))

    def get_stepper(self) -> TrainStep:
        return MultitaskTrainStep()
