import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from backend.multitask.hnet.train_api.progress_tracking import ProgressTracker
from backend.multitask.hnet.train_api.training import ATrainer
from backend.image_processing import process
from backend.parameters import ParameterMap

class RunProgressTrackerWandb(ProgressTracker):
    """ Runs a model for all images provided by the dataloader and for all tasks
        and uploads the resulting saliency maps to wandb """
    def __init__(self, run_dataloader : DataLoader, postprocess_parameter_map : ParameterMap, report_prefix : str = "", log_freq : int = 1):
        super().__init__()

        self._run_dataloader = run_dataloader
        self._postprocess_parameter_map = postprocess_parameter_map
        self._report_prefix = report_prefix
        self._log_freq = log_freq
    
    def track_progress(self, trainer: ATrainer):
        epoch = trainer.epoch

        should_invoke = (epoch < 0) or (epoch % self._log_freq == 0)
        if not should_invoke: return

        model = trainer.model
        tasks = model.tasks
        device = model.device

        with torch.no_grad():
            cols = ["Model"]
            cols.extend([os.path.basename(output_path[0]) for (_, _, output_path) in self._run_dataloader])

            data = []
            for task in tasks:
                row = [task]
                task_id = model.task_to_id(task)

                for (image, _, _) in self._run_dataloader:
                    image = image.to(device)
                    saliency_map = model.compute_saliency(image, task_id)
                    post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocess_parameter_map)*255).astype(np.uint8), 0, 255)
                    img = wandb.Image(post_processed_image)
                    row.append(img)

                    if torch.cuda.is_available(): # avoid GPU out of mem
                        del image
                        del saliency_map
                        torch.cuda.empty_cache()
                
                data.append(row)

            table = wandb.Table(data=data, columns=cols)
            wandb.log({f"{self._report_prefix} - Epoch {epoch}": table})