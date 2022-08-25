from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn

from backend.multitask.hnet.models.hyper_model import HyperModel

class ATrainer(ABC):
    def __init__(self, epochs : int, model : HyperModel, optimizer : torch.optim.Optimizer, loss_fn : nn.Module):
        super().__init__()

        self._epochs = epochs
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        self.reset()
    
    def reset(self):
        self._epoch : int = -1
        self._val_losses : List[float] = []
        self._train_losses : List[float] = []
    
    @property
    def epochs(self):
        return self._epochs
    
    @property 
    def epoch(self):
        return self._epoch
    
    @property
    def train_losses(self):
        return self._train_losses
    
    @property
    def val_losses(self):
        return self._val_losses
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer : torch.optim.Optimizer):
        self._optimizer = optimizer
        return self
    
    @property
    def loss_fn(self):
        return self._loss_fn

    @abstractmethod
    def train(self):
        pass

class TrainStep(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, trainer : ATrainer, data, mode : str) -> float:
        pass