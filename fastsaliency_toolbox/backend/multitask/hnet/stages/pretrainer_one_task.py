"""
PreTrainerOneTask
-----------------

DESCRIPTION:
    Pretrains a hypernetwork and mainnetwork on just one task 
    (but the architecture can be later used to train multiple tasks)
    and reports the progress and metrics to wandb.

RETURN VALUE:
    The best (according to validation loss) pretrained model

CONFIG:
pretrain_one_task:
    tasks                   (List[str]) : first task of this list will be pretrained

    input_saliencies        (str)       : path to saliency map base folder (base_folder/task/img.jpg)
    input_images_train      (str)       : path to images for training (folder/img.jpg), img in saliencies
    input_images_val        (str)       : path to images for validation (folder/img.jpg), img in saliencies
    input_images_run        (str)       : path to images for running (folder/img.jpg)

    loss                    (str)       : One of BCELoss, L1Loss, MSELoss, HuberLoss
    batch_size              (int)       : batch size
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
from backend.multitask.hnet.train_impl.data import BatchAndTaskProvider
from backend.multitask.hnet.stages.trainer_stage import ASaliencyTrainer
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
