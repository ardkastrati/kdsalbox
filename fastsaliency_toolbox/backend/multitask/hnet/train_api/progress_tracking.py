from abc import ABC, abstractmethod

from backend.multitask.hnet.train_api.training import ATrainer

class ProgressTracker(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def track_progress(self, trainer : ATrainer):
        pass