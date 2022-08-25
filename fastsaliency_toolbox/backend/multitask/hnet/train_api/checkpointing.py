from abc import ABC, abstractmethod

from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.models.hyper_model import HyperModel

class Checkpointer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def try_make_checkpoint(self, trainer : ATrainer) -> bool:
        """ Tries to make a checkpoint and returns true iff a checkpoint has been created """
        pass

    @abstractmethod
    def restore_best(self, trainer : ATrainer) -> HyperModel:
        """ Restores the best checkpoint """
        pass


