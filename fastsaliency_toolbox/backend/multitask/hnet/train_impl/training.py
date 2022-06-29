from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from backend.multitask.hnet.train_api.progress_tracking import ProgressTracker
from backend.multitask.hnet.train_api.actions import EpochAction, BatchAction, StartAction, EndAction
from backend.multitask.hnet.train_api.checkpointing import Checkpointer
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.train_api.training import ATrainer, TrainStep
from backend.multitask.hnet.models.hyper_model import HyperModel

class Trainer(ATrainer):
    """ A trainer with the option to track the progress, do checkpointing and also support actions such as lr-decay """
    def __init__(self, epochs : int, model : HyperModel, optimizer : torch.optim.Optimizer,
        loss_fn : nn.Module, stepper : TrainStep, data_providers : Dict[str, DataProvider]):
        super().__init__(epochs, model, optimizer, loss_fn)

        self._stepper = stepper
        self._data_providers = data_providers

        self._progress_trackers : List[ProgressTracker] = []
        self._start_actions : List[StartAction] = []
        self._epoch_start_actions : List[EpochAction] = []
        self._epoch_end_actions : List[EpochAction] = []
        self._batch_actions : List[BatchAction] = []
        self._end_actions : List[EndAction] = []
        self._checkpointer : Checkpointer = None
    
    def add_progress_tracker(self, progress_tracker : ProgressTracker):
        self._progress_trackers.append(progress_tracker)
        return self
    
    def add_start_action(self, action : StartAction):
        self._start_actions.append(action)
        return self

    def add_epoch_start_action(self, action : EpochAction):
        self._epoch_start_actions.append(action)
        return self

    def add_epoch_end_action(self, action : EpochAction):
        self._epoch_end_actions.append(action)
        return self

    def add_batch_action(self, action : BatchAction):
        self._batch_actions.append(action)
        return self
    
    def add_end_action(self, action : EndAction):
        self._end_actions.append(action)
        return self

    def set_checkpointer(self, checkpointer : Checkpointer):
        self._checkpointer = checkpointer
        return self

    def set_dataproviders(self, dataproviders : Dict[str, DataProvider]):
        self._data_providers = dataproviders
        return self

    def clear(self):
        """ Clears progress tracking, checkpointer and all actions """
        self._progress_trackers.clear()
        self._checkpointer = None
        self._start_actions.clear()
        self._epoch_start_actions.clear()
        self._batch_actions.clear()
        self._epoch_end_actions.clear()
        self._end_actions.clear()

        return self
    
    def train(self):
        # invoke all start actions
        for action in self._start_actions:
            action.invoke(self)

        # report initial progress
        for progress_tracker in self._progress_trackers:
            progress_tracker.track_progress(self)

        # for each epoch
        for epoch in range(self.epochs):
            self._epoch = epoch

            # invoke all epoch start actions
            for action in self._epoch_start_actions:
                action.invoke(self)

            # training
            train_loss = self._train_one("train")
            self.train_losses.append(train_loss)

            # validation
            val_loss = self._train_one("val")
            self.val_losses.append(val_loss)

            # check if a checkpoint should be created
            if self._checkpointer is not None and self._checkpointer.should_make_checkpoint(self):
                self._checkpointer.make_checkpoint(self)

            # track the progress on every epoch
            for progress_tracker in self._progress_trackers:
                progress_tracker.track_progress(self)

            # invoke all epoch end actions
            for action in self._epoch_end_actions:
                action.invoke(self)

        # restore the best checkpoint
        if self._checkpointer:
            self._checkpointer.restore_best(self)
        
        # invoke all end actions
        for action in self._end_actions:
            action.invoke(self)
    
    # trains one epoch
    def _train_one(self, mode : str):
        model = self._model

        if mode == "train": model.train()
        elif mode == "val": model.eval()

        all_loss = []
        data_provider = self._data_providers[mode]
        total_batches = data_provider.batch_cnt
        for data in data_provider.batches:
            # do one step / process a batch
            loss = self._stepper.step(self, data, mode)
            all_loss.append(loss)

            # invoke all batch actions
            for action in self._batch_actions:
                action.invoke(self, mode, all_loss, total_batches)
            
            # remove batch from gpu (if cuda)
            if torch.cuda.is_available():
                del data
                torch.cuda.empty_cache()
                
                
        return np.mean(all_loss)

class MultitaskTrainStep(TrainStep):
    """ Training step on a HNET MNET configuration with (task_id, X, y) data """
    def __init__(self):
        super().__init__()

    def step(self, trainer: ATrainer, data, mode : str) -> float:
        optimizer = trainer.optimizer
        loss_fn = trainer.loss_fn
        model = trainer.model
        device = model.device

        task_id, X, y = data

        optimizer.zero_grad()

        # put data on GPU (if cuda)
        X = X.to(device)
        y = y.to(device)

        pred = model(task_id, X)
        loss = loss_fn(pred, y)

        # training
        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_item = loss.item()

        # remove batch from gpu (if cuda)
        if torch.cuda.is_available():
            del pred
            del loss

        return loss_item

class WeightsTrainStep(TrainStep):
    """ Training step on HNET to output weights with (task_ids, weights) data """
    def __init__(self):
        super().__init__()

    def step(self, trainer: ATrainer, data, mode: str) -> float:
        optimizer = trainer.optimizer
        model = trainer.model
        hnet = model.hnet
        loss_fn = trainer.loss_fn
        device = model.device

        task_ids, y = data

        optimizer.zero_grad()

        # put data on GPU (if cuda)
        task_ids = task_ids.to(device)
        y = y.to(device)

        task_ids = task_ids.tolist()
        weights = hnet(task_id=task_ids) 
        weights = torch.stack([torch.cat([w.flatten() for w in datapoint]) for datapoint in weights])
        loss = loss_fn(weights, y)

        # training
        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_item = loss.item()
        
        # remove batch from gpu (if cuda)
        if torch.cuda.is_available():
            del weights
            del loss

        return loss_item

