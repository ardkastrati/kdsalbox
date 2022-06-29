import os
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from backend.image_processing import process
from backend.metrics import NSS, CC, SIM, KL as KL
from backend.parameters import ParameterMap
from backend.multitask.hnet.train_api.progress_tracking import ProgressTracker
from backend.multitask.hnet.train_api.training import ATrainer
from backend.multitask.hnet.models.hyper_model import HyperModel

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

        self.track_progress_core(trainer.model, f"{self._report_prefix} - Epoch {epoch}")        
        
    def track_progress_core(self, model : HyperModel, report_name : str):
        with torch.no_grad():
            device = model.device
            tasks = model.tasks

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
            wandb.log({report_name: table})


class TesterProgressTrackerWandb(ProgressTracker):
    """ Tests a model for each task and computes & reports important metrics such as KLDiv or CC. """
    def __init__(self, dataloaders : Dict[str,DataLoader], batches_per_task : int, 
        postprocess_parameter_map : ParameterMap, logging_dir : str,
        log_freq : int = 1, per_image_statistics : bool = False, verbose : bool = True):
        """

        Args:
            dataloaders (Dict[str,DataLoader]): one dataloader for each task. Provide (X,y,names of images).
                (make sure the dataloaders all are shuffled)
            batches_per_task (int): how many batches to sample per task
            post_process_parameter_map (ParameterMap): parameter map for postprocessing the generated saliency images
            log_freq (int): will only execute if epoch % log_freq == 0
            per_image_statistics (bool): whether or not to report stats for each individual image 
                (make sure dataloader batchsize == 1 if true)
            verbose (bool): print to console

        """
        super().__init__()

        self._dataloaders = dataloaders
        self._batches_per_task = batches_per_task
        self._postprocess_parameter_map = postprocess_parameter_map

        self._per_image_statistics = per_image_statistics
        self._log_freq = log_freq
        self._verbose = verbose
        self._logging_dir = logging_dir

        if self._per_image_statistics:
            for d in self._dataloaders.values():
                assert d.batch_size == 1, "Batch size cannot be greater than 1 if using per image statistics"

    def track_progress(self, trainer: ATrainer):
        epoch = trainer.epoch
        model = trainer.model

        should_invoke = (epoch % self._log_freq == 0)
        if not should_invoke: return

        self.track_progress_core(model)
        

    def track_progress_core(self, model : HyperModel):
        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL = [], [], [], [], [], []
        for task,dataloader in self._dataloaders.items():
            loss, NSS, CC, SIM, KL = self._test_one_task(task, model, dataloader)

            all_names.append(task)
            all_loss.append(loss)
            all_NSS.append(NSS)
            all_CC.append(CC)
            all_SIM.append(SIM)
            all_KL.append(KL)

        # total
        loss, NSS, CC, SIM, KL = np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL))
        all_names.append("Average")
        all_loss.append(loss)
        all_NSS.append(NSS)
        all_CC.append(CC)
        all_SIM.append(SIM)
        all_KL.append(KL)

        # save test results
        stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "test_results").replace("\\", "/")
        data = np.array([all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL]).transpose().tolist()
        table = wandb.Table(data=data, columns=["Task", "Loss", "NSS", "CC", "SIM", "KL"])
        wandb.run.log({stats_file:table})


    def _test_one_task(self, task : str, model : HyperModel, dataloader : DataLoader):
        model.eval()
        task_id = model.task_to_id(task)
        device = model.device

        if self._verbose: print(f"Testing {task}...")

        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL = [], [], [], [], [], []
        loss_fn = torch.nn.BCELoss()

        for i,(X, y, names) in enumerate(dataloader):
            if i >= self._batches_per_task: break
            
            X = X.to(device)
            y = y.to(device)

            pred = model(task_id, X)
            loss = loss_fn(pred, y)
            detached_pred_t = pred.cpu().detach()
            detached_y_t = y.cpu().detach()
            detached_pred = detached_pred_t.numpy()
            detached_y = detached_y_t.numpy()

            # Doing the postprocessing steps needed for the metrics (We might want to do this also for Task Evaluation stuff?)
            y = process(y, self._postprocess_parameter_map)
            detached_pred = process(detached_pred, self._postprocess_parameter_map)
            NSSes = [NSS(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            CCes = [CC(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            SIMes = [SIM(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            KLes = [KL(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]

            with torch.no_grad():
                all_names.extend(names)
                all_loss.append(loss.item())
                all_NSS.extend(NSSes)
                all_CC.extend(CCes)
                all_SIM.extend(SIMes)
                all_KL.extend(KLes)

            if torch.cuda.is_available():
                # Remove batch from gpu
                del pred
                del loss
                del X
                del y
                torch.cuda.empty_cache()

            if i%100 == 0:
                print(f"Batch {i}/{self._batches_per_task}: current accumulated loss {np.mean(all_loss)}", flush=True)

        # save per image stats
        if self._per_image_statistics:
            stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "image_statistics", task).replace("\\", "/")
            data = np.array([all_names, all_NSS, all_CC, all_SIM, all_KL]).transpose().tolist()
            table = wandb.Table(data=data, columns=["Image", "NSS", "CC", "SIM", "KL"])
            wandb.run.log({stats_file:table})
            
        return np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL))
        