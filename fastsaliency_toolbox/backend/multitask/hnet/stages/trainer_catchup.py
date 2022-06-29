"""
Trainer
-------

Trains a hypernetwork and mainnetwork on multiple tasks at the same time
but freezes all the parameters of the HNET that are shared between tasks

TODO: add a lot more documentation here about all the parameters

"""

import os
from typing import Dict
from torch.utils.data import DataLoader

from backend.datasets import TrainDataManager
from backend.multitask.hnet.train_impl.data import MultitaskBatchProvider
from backend.multitask.hnet.stages.trainer import ASaliencyTrainer
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.train_impl.actions import FreezeHNETForCatchup

class TrainerCatchup(ASaliencyTrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[name]

        self._imgs_per_task_train = train_conf["imgs_per_task_train"]
        self._imgs_per_task_val = train_conf["imgs_per_task_val"]

        self._batches_per_task_train = self._imgs_per_task_train // self._batch_size
        self._batches_per_task_val = self._imgs_per_task_val // self._batch_size

        self._consecutive_batches_per_task = train_conf["consecutive_batches_per_task"]

    def get_data_providers(self) -> Dict[str, DataProvider]:
        sal_folders = [os.path.join(self._input_saliencies, task) for task in self._tasks] # path to saliency folder for all models

        train_datasets = [TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]
        val_datasets = [TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        train_dataloaders = { self._model.task_to_id(task): 
            DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) 
            for (task,ds) in zip(self._tasks, train_datasets)}

        val_dataloaders = { self._model.task_to_id(task):
            DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) 
            for (task,ds) in zip(self._tasks, val_datasets)}

        dataproviders = {
            "train": MultitaskBatchProvider(self._batches_per_task_train, self._consecutive_batches_per_task, train_dataloaders),
            "val": MultitaskBatchProvider(self._batches_per_task_val, self._consecutive_batches_per_task, val_dataloaders)
        }

        return dataproviders

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        
        self._trainer.add_epoch_start_action(FreezeHNETForCatchup(self._epochs - 1))
