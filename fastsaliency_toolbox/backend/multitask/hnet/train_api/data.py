"""
Responsible to provide data for training.

"""

from abc import ABC, abstractmethod
from typing import Generator

class DataProvider(ABC):
    """ Abstracts data-loading.
    
        Allows flexible data-loading such as:
            - sampling from a set of dataloaders (as is the case in multitask training)
            - sampling from a dataloader (as is the case in singletask training)
    """
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def batches(self) -> Generator:
        """ Returns a generator of data """
        pass

    @property
    @abstractmethod
    def batch_cnt(self) -> int:
        pass