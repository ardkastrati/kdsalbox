from typing import Dict
import numpy as np
from torch.utils.data import DataLoader

from backend.multitask.hnet.train_api.data import DataProvider

class MultitaskBatchProvider(DataProvider):
    """ Loads data from different different sources (one per task_id) """
    def __init__(self, batches_per_task : int, consecutive_batches_per_task : int, dataloaders : Dict[int, DataLoader]):
        super().__init__()

        self._batches_per_task = batches_per_task
        self._consecutive_batches_per_task = consecutive_batches_per_task
        self._dataloaders = dataloaders
        self._task_ids = dataloaders.keys()

        # compute the amount of batches
        task_cnt = len(self._task_ids)
        limit = self._batches_per_task // self._consecutive_batches_per_task
        self._batch_cnt = task_cnt * limit * consecutive_batches_per_task

    @property
    def batches(self):
        limit = self._batches_per_task // self._consecutive_batches_per_task

        # generate the seed array
        all_batches = np.concatenate([np.repeat(task_id, limit) for task_id in self._task_ids])
        # shuffle the seed array
        np.random.shuffle(all_batches)
        # repeat each element of the seed array n times
        all_batches = np.repeat(all_batches, self._consecutive_batches_per_task)

        # create a data iterator for each task
        # Note: DataLoader shuffles when iterator is created
        data_iters = [iter(d) for d in self._dataloaders.values()] 

        # for each batch in all_batches
        for task_id in all_batches:
            # load a batch from the corresponding data iterator
            X,y = next(data_iters[task_id])
            yield (task_id.item(), X, y)
    
    @property
    def batch_cnt(self) -> int:
        return self._batch_cnt

class BatchAndTaskProvider(DataProvider):
    """ Loads data from a dataloader and prepends a task_id """
    def __init__(self, dataloader : DataLoader, task_id : int):
        super().__init__()

        self._task_id = task_id
        self._dataloader = dataloader

        self._batch_cnt = len(dataloader)

    @property
    def batches(self):
        for X,y in self._dataloader:
            yield (self._task_id, X, y)
    
    @property
    def batch_cnt(self) -> int:
        return self._batch_cnt

class BatchProvider(DataProvider):
    """ Wraps a dataloader """
    def __init__(self, dataloader : DataLoader):
        super().__init__()

        self._dataloader = dataloader
        self._batch_cnt = len(dataloader)

    @property
    def batches(self):
        for X,y in self._dataloader:
            yield (X, y)
    
    @property
    def batch_cnt(self) -> int:
        return self._batch_cnt
