"""
Progress tracking is basically a special action that is intended to track the progress of the model.
When progress tracking is done is left to the implementation (usually after checkpointing or each epoch).

"""

from abc import ABC, abstractmethod

from backend.multitask.hnet.train_api.training import ATrainer

class ProgressTracker(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def track_progress(self, trainer : ATrainer):
        pass