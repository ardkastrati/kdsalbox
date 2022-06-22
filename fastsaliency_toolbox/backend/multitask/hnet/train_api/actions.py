from abc import ABC, abstractmethod
from typing import List

from backend.multitask.hnet.train_api.training import ATrainer

class StartAction(ABC):
    """ Action to be executed at the start of training """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def invoke(self, trainer : ATrainer):
        pass

class EndAction(ABC):
    """ Action to be executed at the end of training """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def invoke(self, trainer : ATrainer):
        pass

class EpochAction(ABC):
    """ Action to be executed every epoch """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def invoke(self, trainer : ATrainer):
        pass

class BatchAction(ABC):
    """ Action to be executed every batch """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def invoke(self, trainer : ATrainer, mode : str, batch_losses : List[float], total_batches : int):
        pass