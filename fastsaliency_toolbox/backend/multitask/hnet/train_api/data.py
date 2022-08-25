from abc import ABC, abstractmethod
from typing import Generator

class DataProvider(ABC):
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