"""
Collection of ProgressTrackers

"""

from typing import List
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.image_processing import process
from backend.utils import save_image
from backend.parameters import ParameterMap
from backend.multitask.hnet.train_api.progress_tracking import ProgressTracker
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.models.hyper_model import HyperModel

class RunProgressTracker(ProgressTracker):
    """ Runs a model for all images provided by the dataloader and for all tasks
        and stores the resulting images on disk """
    def __init__(self, run_dataloader : DataLoader, tasks : List[str], postprocess_parameter_map : ParameterMap, report_prefix : str = "", log_freq : int = 1):
        super().__init__()

        self._run_dataloader = run_dataloader
        self._tasks = tasks
        self._postprocess_parameter_map = postprocess_parameter_map
        self._report_prefix = report_prefix
        self._log_freq = log_freq
    
    def track_progress(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_invoke = (epoch < 0) or (epoch % self._log_freq == 0)
        if not should_invoke: return

        self.track_progress_core(trainer.model, f"{self._report_prefix} - Epoch {epoch}")        
        
    def track_progress_core(self, model : HyperModel, report_name : str = None):
        with torch.no_grad():
            device = model.device

            for task in self._tasks:
                task_id = model.task_to_id(task)

                for (image, _, output_paths) in self._run_dataloader:
                    output_path = output_paths[0] # only batch size of 1
                    path, file_name = os.path.split(output_path)
                    out_folder = os.path.join(path, task)
                    if report_name:
                        out_folder = os.path.join(path, report_name, task) # path/report_name/task/
                    os.makedirs(out_folder, exist_ok=True)

                    output_path = os.path.join(out_folder, file_name) # path/report_name/task/file.jpg

                    image = image.to(device)
                    saliency_map = model.compute_saliency(image, task_id)
                    post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocess_parameter_map)*255).astype(np.uint8), 0, 255)
                    
                    print(post_processed_image.shape)
                    save_image(output_path, post_processed_image)

                    if torch.cuda.is_available(): # avoid GPU out of mem
                        del image
                        del saliency_map
                        torch.cuda.empty_cache()