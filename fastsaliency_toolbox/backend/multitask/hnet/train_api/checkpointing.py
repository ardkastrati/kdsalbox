from abc import ABC
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.hyper_model import HyperModel

class Checkpointer(ABC):
    def __init__(self):
        super().__init__()

    def should_make_checkpoint(self, trainer : ATrainer) -> bool:
        pass

    def make_checkpoint(self, trainer : ATrainer):
        pass

    def restore_best(self, trainer : ATrainer) -> HyperModel:
        pass


