"""
PreTrainerWeights
-----------------

DESCRIPTION:
    Trains a hypernetwork to output some specific weights (e.g. to learn pretrained weights before the actual training begins)

RETURN VALUE:
    The best (according to validation loss) pretrained model

CONFIG:
pretrain_weights:
    tasks                   (List[str]) : all the tasks that will be trained

    target_model_weights    (str)       : base path to folder (folder/task/task.pth)
    input_images_run        (str)       : path to images for running (folder/img.jpg)

    loss                    (str)       : One of BCELoss, L1Loss, MSELoss, HuberLoss
    batch_size              (int)       : batch size
    epochs                  (int)       : amount of training epochs
    lr                      (float)     : learning rate
    lr_decay                (float)     : by how much should the lr decay in decay_epochs
    decay_epochs            (int)       : which epoch should the lr decay

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
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader

import backend.student as stud
from backend.multitask.hnet.datasets import WeightDataset
from backend.multitask.hnet.stages.trainer_stage import ATrainer
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.train_api.training import TrainStep
from backend.multitask.hnet.train_impl.data import BatchProvider
from backend.multitask.hnet.train_impl.training import WeightsTrainStep
from backend.multitask.hnet.train_impl.actions import WeightWatcher


class PreTrainerWeights(ATrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[self._name]
        self._target_model_weights = train_conf["target_model_weights"]

        self._mnet_conf = conf["model"]["mnet"]

    def get_data_providers(self) -> Dict[str, DataProvider]:
        model_paths = [(self._model.task_to_id(task), os.path.join(self._target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        train_dataset = WeightDataset(model_paths, self.map_old_to_new_weights, flatten=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4) # TODO: custom validation dataset

        dataproviders = {
            "train": BatchProvider(train_dataloader),
            "val": BatchProvider(val_dataloader) 
        }

        return dataproviders

    # run the entire training
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)

        # check if the extpected output format of the HNET matches the labels/weights loaded from the original models
        new_shapes = self._model.mnet.get_cw_param_shapes()
        old_stud = stud.Student()
        old_shapes = self.map_old_to_new_weights({n:p.size() for n,p in old_stud.named_parameters()}, old_stud, verbose=self._verbose) 
        assert len(new_shapes) == len(old_shapes), f"HNET output generates {len(new_shapes)} weight tensors whereas we only loaded {len(old_shapes)} into the label from the original models!"
        for new,old in zip(new_shapes, old_shapes):
            assert new == old, "Mismatch between HNET output format and loaded model weight format. Make sure the order of parameters is the same!"

    def execute(self):
        return super().execute()

    def get_stepper(self) -> TrainStep:
        return WeightsTrainStep()

    def map_old_to_new_weights(self, named_weights : List[Tuple[str,torch.Size]], legacy_model, verbose=False):
        decoder_selection = [
            "conv7_3.weight",   "conv7_3.bias",     "bn7_3.weight",     "bn7_3.bias",
            "conv8_1.weight",   "conv8_1.bias",     "bn8_1.weight",     "bn8_1.bias",
            "conv8_2.weight",   "conv8_2.bias",     "bn8_2.weight",     "bn8_2.bias", 
            "conv9_1.weight",   "conv9_1.bias",     "bn9_1.weight",     "bn9_1.bias",
            "conv9_2.weight",   "conv9_2.bias",     "bn9_2.weight",     "bn9_2.bias",
            "conv10_1.weight",  "conv10_1.bias",    "bn10_1.weight",    "bn10_1.bias",
            "conv10_2.weight",  "conv10_2.bias",    "bn10_2.weight",    "bn10_2.bias",
            "output.weight",    "output.bias"
        ]
        
        new_weights = []
        new_encoder_param_cnt = len(list(self._model.mnet.get_layer("encoder").parameters()))
        # select the first few params, assuming that 
        encoder_selection = [n for n,p in legacy_model.encoder.named_parameters()][new_encoder_param_cnt:] 

        for s in encoder_selection:
            new_weights.append(named_weights[f"encoder.{s}"])    

        for s in decoder_selection:
            new_weights.append(named_weights[f"decoder.{s}"])

        return new_weights

    def map_old_to_new_buffers(self, named_buffers : List[Tuple[str,torch.Size]], legacy_model, verbose=False):
        
        encoder_selection = [n for n,p in legacy_model.encoder.named_buffers()]
        decoder_selection = [n for n,p in legacy_model.decoder.named_buffers()] # bn layers are already in correct order in stud 

        new_buffers = []
        for s in encoder_selection:
            new_buffers.append(named_buffers[f"encoder.{s}"])    

        for s in decoder_selection:
            new_buffers.append(named_buffers[f"decoder.{s}"])

        return new_buffers
