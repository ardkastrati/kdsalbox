"""
Trainer Generalization Task
------------------------

Incrementally trains a model on a new task and reports how good the model performs on the new task

"""

from typing import Dict
import os
from torch.utils.data import DataLoader

from backend.datasets import TrainDataManager
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.stages.trainer import ASaliencyTrainer
from backend.multitask.hnet.train_impl.data import BatchAndTaskProvider
from backend.multitask.hnet.train_impl.actions import FreezeHNETShared, LoadModel
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.train_impl.training import Trainer

class TrainerGeneralizationTask(ASaliencyTrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[name]

        self._imgs_per_task_train = train_conf["imgs_per_task_train"]
        self._imgs_per_task_val = train_conf["imgs_per_task_val"]

        self._freeze_hnet_shared_steps = train_conf["freeze_hnet_shared_steps"]

        self._initial_model_path = ""

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)

        self._initial_model_path = os.path.join(self._logging_dir, "original.pth")
        self._model.save(self._initial_model_path)

    def build_trainer(self) -> Trainer:
        return super().build_trainer().add_epoch_start_action(FreezeHNETShared(self._freeze_hnet_shared_steps))\
            .add_end_action(LoadModel(self._initial_model_path)) # reset to the original model at the end of training


    def get_data_providers(self) -> Dict[str, DataProvider]:
        sal_folders = [os.path.join(self._input_saliencies, task) for task in self._tasks] # path to saliency folder for all models

        train_datasets = [TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]
        val_datasets = [TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        train_dataloaders = { 
            task: DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) 
            for (task,ds) in zip(self._tasks, train_datasets)
        }

        val_dataloaders = { 
            task: DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) 
            for (task,ds) in zip(self._tasks, val_datasets)
        }

        dataproviders = {
            task: 
            {
                "train": BatchAndTaskProvider(train_dataloaders[task], self._model.task_to_id(task)),
                "val": BatchAndTaskProvider(val_dataloaders[task], self._model.task_to_id(task))
            } 
            for task in self._tasks
        }

        # sanity checks
        assert self._imgs_per_task_train <= min([len(ds) for ds in train_datasets])
        assert self._imgs_per_task_val <= min([len(ds) for ds in val_datasets])

        return dataproviders

    def execute(self):
        AStage.execute(self)

        self._model.to(self._device)

        for task in self._tasks:
            trainer = self.build_trainer()
            trainer.set_dataproviders(self._dataproviders[task])
            trainer.train()

        return self._model

        