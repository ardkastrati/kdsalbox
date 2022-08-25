"""
TrainerGeneralizationTask
-------------------------

DESCRIPTION:
    Trains a model on a new task and reports how good the model performs on the new task.
    Only trains the embedding first and then trains the entire network ('freeze_hnet_shared_steps' config parameter)

RETURN VALUE:
    The best (according to validation loss) generalized model

CONFIG:
train_generalization_task:
    tasks                   (List[str]) : all tasks that will be trained

    input_saliencies        (str)       : path to saliency map base folder (base_folder/task/img.jpg)
    input_images_train      (str)       : path to images for training (folder/img.jpg), img in saliencies
    train_img_cnt           (int)       : how many images of the available images should be used for training for each task
    input_images_val        (str)       : path to images for validation (folder/img.jpg), img in saliencies
    val_img_cnt             (int)       : how many images of the available images should be used for validation for each task
    input_images_run        (str)       : path to images for running (folder/img.jpg)

    loss                    (str)       : One of BCELoss, L1Loss, MSELoss, HuberLoss
    batch_size              (int)       : batch size
    epochs                  (int)       : amount of training epochs
    lr                      (float)     : learning rate
    lr_decay                (float)     : by how much should the lr decay in decay_epochs
    decay_epochs            (int)       : which epoch should the lr decay
    freeze_encoder_steps    (int)       : for how many epochs should the encoder be frozen at the beginning
    freeze_hnet_shared_steps(int)       : for how many epochs should the shared hypernetwork parameters be frozen at the beginning (training embedding only)

    auto_checkpoint_steps   (int)       : automatically make checkpoint every x epochs
    max_checkpoint_freq     (int)       : limits the amount of checkpoints that can be made (e.g. if improves every epoch)

    log_freq                (int)       : how often should epoch summary be logged in console
    batch_log_freq          (int)       : how often should loss be logged in console (make this very large to turn off)
    wandb:
        save_checkpoints_to_wandb (bool): should models be saved to wandb (false to reduce storage, export will happen anyway)
        watch:
            log           (Optional str): see wandb.watch documentation
            log_freq      (int)         : see wandb.watch documentation

"""

from typing import Dict
import os
import torch
from torch.utils.data import DataLoader

from backend.datasets import TrainDataManager
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.stages.trainer_stage import ASaliencyTrainer
from backend.multitask.hnet.train_impl.data import BatchAndTaskProvider
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.train_impl.training import Trainer
from backend.multitask.hnet.train_impl.actions import FreezeHNETShared

class TrainerGeneralizationTask(ASaliencyTrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[name]

        self._train_img_cnt = train_conf["train_img_cnt"]
        self._val_img_cnt = train_conf["val_img_cnt"]

        self._freeze_hnet_shared_steps = train_conf["freeze_hnet_shared_steps"]

        self._initial_model_path = ""

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)

        self._initial_model_path = os.path.join(self._logging_dir, "original.pth")
        self._model.save(self._initial_model_path)

    def get_optimizer(self) -> torch.optim.Optimizer:
        params = self._model.task_parameters(task_ids=[self._model.task_to_id(t_id) for t_id in self._tasks])
        return torch.optim.Adam(params, lr=self._lr)

    def build_trainer(self) -> Trainer:
        return super().build_trainer()\
            .add_epoch_start_action(FreezeHNETShared(self._freeze_hnet_shared_steps, self._lr))


    def get_data_providers(self) -> Dict[str, DataProvider]:
        sal_folders = [os.path.join(self._input_saliencies, task) for task in self._tasks] # path to saliency folder for all models

        train_datasets = [TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map, self._train_img_cnt) for sal_path in sal_folders]
        val_datasets = [TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map, self._val_img_cnt) for sal_path in sal_folders]

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
        assert self._train_img_cnt <= min([len(ds) for ds in train_datasets])
        assert self._val_img_cnt <= min([len(ds) for ds in val_datasets])

        return dataproviders

    def execute(self):
        AStage.execute(self)

        self._model.to(self._device)

        for task in self._tasks:
            trainer = self.build_trainer()
            trainer.set_dataproviders(self._dataproviders[task])
            trainer.train()

        return self._model

        