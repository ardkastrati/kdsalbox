"""
MainTrainer
-----------

DESCRIPTION:
    Trains a hypernetwork and mainnetwork on multiple tasks at the same time
    and reports the progress and metrics to wandb.

RETURN VALUE:
    The best (according to validation loss) trained model

CONFIG:
train:
    tasks                   (List[str]) : all the tasks that will be trained

    input_saliencies        (str)       : path to saliency map base folder (base_folder/task/img.jpg)
    input_images_train      (str)       : path to images for training (folder/img.jpg), img in saliencies
    imgs_per_task_train     (int)       : how many images of the available images should be used for training for each task
    input_images_val        (str)       : path to images for validation (folder/img.jpg), img in saliencies
    imgs_per_task_val       (int)       : how many images of the available images should be used for validation for each task
    input_images_run        (str)       : path to images for running (folder/img.jpg)

    loss                    (str)       : One of BCELoss, L1Loss, MSELoss, HuberLoss
    batch_size              (int)       : batch size
    consecutive_batches_per_task (int)  : how many batches in a row should be sampled from the same task
    epochs                  (int)       : amount of training epochs
    lr                      (float)     : learning rate
    lr_decay                (float)     : by how much should the lr decay in decay_epochs
    decay_epochs            (int)       : which epoch should the lr decay
    freeze_encoder_steps    (int)       : for how many epochs should the encoder be frozen at the beginning

    auto_checkpoint_steps   (int)       : automatically make checkpoint every x epochs
    max_checkpoint_freq     (int)       : limits the amount of checkpoints that can be made (e.g. if improves every epoch)

    batch_log_freq          (int)       : how often should loss be logged in console
    wandb:
        save_checkpoints_to_wandb (bool): should models be saved to wandb (false to reduce storage, export will happen anyway)
        watch:
            log           (Optional str): see wandb.watch documentation
            log_freq      (int)         : see wandb.watch documentation

"""

import os
from typing import Dict
from torch.utils.data import DataLoader

from backend.datasets import TrainDataManager
from backend.multitask.hnet.stages.trainer_stage import ASaliencyTrainer
from backend.multitask.hnet.train_impl.data import MultitaskBatchProvider
from backend.multitask.hnet.train_api.data import DataProvider

class MainTrainer(ASaliencyTrainer):
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

        # sanity checks
        assert self._imgs_per_task_train <= min([len(ds) for ds in train_datasets])
        assert self._imgs_per_task_val <= min([len(ds) for ds in val_datasets])

        return dataproviders