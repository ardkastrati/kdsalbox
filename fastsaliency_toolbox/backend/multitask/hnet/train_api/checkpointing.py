from abc import ABC
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.models.hyper_model import HyperModel

class Checkpointer(ABC):
    def __init__(self):
        super().__init__()

    def should_make_checkpoint(self, trainer : ATrainer) -> bool:
        """ Given the state of the trainer, should a checkpoint be made? """
        pass

    def make_checkpoint(self, trainer : ATrainer):
        """ Actually make a checkpoint """
        pass

    def restore_best(self, trainer : ATrainer) -> HyperModel:
        """ Restores the best checkpoint """
        pass


