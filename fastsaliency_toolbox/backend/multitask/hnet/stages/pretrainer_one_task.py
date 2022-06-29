"""
PreTrainerOneTask
-----------------

Pretrains a hypernetwork and mainnetwork on just one task 
(but the architecture can be later used to train multiple tasks)
and reports the progress and metrics to wandb.

"""

import os
from typing import Dict
from torch.utils.data import DataLoader

from backend.datasets import TrainDataManager
from backend.multitask.hnet.train_impl.data import BatchAndTaskProvider
from backend.multitask.hnet.stages.trainer import ASaliencyTrainer
from backend.multitask.hnet.train_api.data import DataProvider

class PreTrainerOneTask(ASaliencyTrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

    def get_data_providers(self) -> Dict[str, DataProvider]:
        assert self._model is not None, "Model cannot be none during data provider setup"
        selected_task = self._tasks[0]
        selected_task_id = self._model.task_to_id(selected_task)

        sal_path = os.path.join(self._input_saliencies, selected_task)
        train_dataset = TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map)
        val_dataset = TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map)
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4) 
        val_dataloader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4) 

        dataproviders = {
            "train": BatchAndTaskProvider(train_dataloader, selected_task_id),
            "val": BatchAndTaskProvider(val_dataloader, selected_task_id)
        }

        return dataproviders
