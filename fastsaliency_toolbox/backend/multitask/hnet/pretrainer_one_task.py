"""
PreTrainerOneTask
-----------------

Pretrains a hypernetwork and mainnetwork on just one task 
(but the architecture can be later used to train multiple tasks)
and reports the progress and metrics to wandb.

"""

import os
import torch
from torch.utils.data import DataLoader
import wandb

from backend.datasets import TrainDataManager, RunDataManager
from backend.parameters import ParameterMap
from backend.image_processing import process
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel
from backend.multitask.hnet.trainer_utils import run_model_on_images_and_report_to_wandb as track_progress
from backend.multitask.hnet.trainer_utils import train_one_epoch_just_one_task as train_one_epoch
from backend.multitask.hnet.trainer_utils import train_val_one_epoch_and_report_to_wandb as train_and_val
from backend.multitask.hnet.trainer_utils import save

class PreTrainerOneTask(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        train_conf = conf["pretrain_one_task"]

        self._batch_size = train_conf["batch_size"]

        self._export_path = "export/"
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._auto_checkpoint_steps = train_conf["auto_checkpoint_steps"]
        self._input_saliencies = train_conf["input_saliencies"]
        self._train_img_path = train_conf["input_images_train"]
        self._input_images_run = train_conf["input_images_run"]
        self._val_img_path = train_conf["input_images_val"]

        self._tasks = conf["tasks"]
        self._task_cnt = conf["model"]["hnet"]["task_cnt"]
        self._selected_task = self._tasks[0]

        self._loss_fn = train_conf["loss"]
        self._epochs = train_conf["epochs"]
        self._lr = train_conf["lr"]
        self._lr_decay = train_conf["lr_decay"]
        self._freeze_encoder_steps = train_conf["freeze_encoder_steps"]
        self._decay_epochs = train_conf["decay_epochs"]

        self._log_freq = train_conf["log_freq"]

        wandb_conf = train_conf["wandb"]
        self._wandb_watch_log = wandb_conf["watch"]["log"]
        self._wandb_watch_log_freq = wandb_conf["watch"]["log_freq"]
        self._save_checkpoints_to_wandb = wandb_conf["save_checkpoints_to_wandb"]

        # convert parameter dicts to parametermap such that it can be used in process()
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Trainer expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # setup logging
        self._export_dir = os.path.join(self._logging_dir, self._export_path)
        os.makedirs(self._logging_dir, exist_ok=True)
        os.makedirs(self._export_dir, exist_ok=True)

        # data loading
        sal_path = os.path.join(self._input_saliencies, self._selected_task)

        train_dataset = TrainDataManager(self._train_img_path, sal_path, self._verbose, self._preprocess_parameter_map)
        val_dataset = TrainDataManager(self._val_img_path, sal_path, self._verbose, self._preprocess_parameter_map)

        self._dataloaders = {
            "train": DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4),
        }

        self._run_dataloader = DataLoader(RunDataManager(self._input_images_run, "", verbose=False, recursive=False), batch_size=1)

        # sanity checks
        assert self._task_cnt == len(self._tasks)


    def execute(self):
        super().execute()

        export_path_best = os.path.join(self._export_dir, "best.pth")
        export_path_final = os.path.join(self._export_dir, "final.pth")

        model = self._model
        model.build()
        model.to(self._device)

        selected_task_id = model.task_to_id(self._selected_task)

        lr = self._lr
        lr_decay = self._lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = {
            "BCELoss": torch.nn.BCELoss(),
            "L1Loss": torch.nn.L1Loss()
        }
        loss = losses[self._loss_fn]
        
        # report to wandb
        wandb.watch((model.hnet, model.mnet), loss, log=self._wandb_watch_log, log_freq=self._wandb_watch_log_freq)
        
        all_epochs = []
        smallest_loss = None

        model.mnet.freeze_encoder()
        if self._verbose: print("Encoder frozen...")

        # evaluate how the model performs initially
        track_progress(f"{self.name} - Initialization", self._tasks, model, self._run_dataloader, 
            self._postprocess_parameter_map, self._device)

        # training loop
        for epoch in range(0, self._epochs):
            # decrease learning rate over time
            if epoch in self._decay_epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_decay
                lr = lr * lr_decay

            # unfreeze the encoder after given amount of epochs
            if epoch == self._freeze_encoder_steps:
                if self._verbose: print("Encoder unfrozen")
                model.mnet.unfreeze_encoder()

            # runs one epoch
            train_one = lambda mode : train_one_epoch(model, self._dataloaders, loss,
                optimizer, mode, self._device, selected_task_id, self._log_freq)
            
            # generates some images to see how good the model works this stage
            run_one = lambda is_best_model : track_progress(f"{self.name} - Progress Epoch {epoch}", self._tasks, model, 
                self._run_dataloader, self._postprocess_parameter_map, self._device)

            # trains and validates & does checkpointing as well as reporting to wandb
            smallest_loss = train_and_val(model, train_one, epoch, lr, smallest_loss, all_epochs, 
                self._auto_checkpoint_steps, self._save_checkpoints_to_wandb, 
                self._logging_dir, self._verbose, export_path_best, self._name, 
                run_one)
        
        # save the final model
        save(export_path_final, model, save_to_wandb=True)

        return model
    
    def cleanup(self):
        super().cleanup()
        del self._dataloaders
