"""
Tester
------

Computes a few metrics for all images in a folder and their corresponding saliency maps and reports them.
By setting conf["test"]["per_image_statistics"], the stats will be produced per image additionally.

"""

import os
import numpy as np
import torch
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader
import wandb

from backend.datasets import TestDataManager
from backend.metrics import NSS, CC, SIM, KL as KL_c
from backend.image_processing import process
from backend.parameters import ParameterMap
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel


class Tester(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        test_conf = conf["test"]

        self._batch_size = 1 # TODO: add support for batch_size > 1 (make sure per_image_statistics still works!)
        self._imgs_per_task_test = test_conf["imgs_per_task_test"]
        self._tasks = conf["tasks"]
        self._input_saliencies = test_conf["input_saliencies"]
        self._input_images_test = test_conf["input_images_test"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._per_image_statistics = test_conf["per_image_statistics"]
        self._batches_per_task_test = self._imgs_per_task_test // self._batch_size

        # convert to pre-/postprocess params
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])
    
    # runs the test for one task/model
    def _test_one_task(self, task, dataloader):

        model = self._model
        model.eval()
        task_id = model.task_to_id(task)

        if self._verbose: print(f"Testing {task}...")

        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c = [], [], [], [], [], [], []
        my_loss = torch.nn.BCELoss()

        data_iter = iter(dataloader)
        for i in range(self._data_offset): next(data_iter) # skip first few images

        for i in range(self._batches_per_task_test):
            (X, y, names) = next(data_iter)
            X = X.to(self._device)
            y = y.to(self._device)

            pred = model(task_id, X)
            losses = my_loss(pred, y)
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
            KLes = [self.KL(map1, map2) for (map1, map2) in zip(detached_pred_t, detached_y_t)]
            KL_ces = [KL_c(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]

            with torch.no_grad():
                all_names.extend(names)
                all_loss.append(losses.item())
                all_NSS.extend(NSSes)
                all_CC.extend(CCes)
                all_SIM.extend(SIMes)
                all_KL.extend(KLes)
                all_KL_c.extend(KL_ces)

            if torch.cuda.is_available():
                # Remove batch from gpu
                del X
                del y
                torch.cuda.empty_cache()

            if i%100 == 0:
                print(f"Batch {i}/{self._batches_per_task_test}: current accumulated loss {np.mean(all_loss)}", flush=True)

        # save per image stats
        if self._per_image_statistics:
            stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "image_statistics", task).replace("\\", "/")
            data = np.array([all_names, all_NSS, all_CC, all_SIM, all_KL, all_KL_c]).transpose().tolist()
            table = wandb.Table(data=data, columns=["Image", "NSS", "CC", "SIM", "KL", "KL custom"])
            wandb.run.log({stats_file:table})
            
        return np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL)), np.nanmean(np.asarray(all_KL_c))
        
        
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Tester expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # prepare logging
        os.makedirs(self._logging_dir, exist_ok=True)

        # data loading
        input_saliencies = self._input_saliencies
        test_img_path = self._input_images_test
        sal_folders = [os.path.join(input_saliencies, task) for task in self._tasks] # path to saliency folder for all models

        test_datasets = [TestDataManager(test_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        min_ds_len = min([len(ds) for ds in test_datasets])
        self._data_offset = 0 if min_ds_len - self._imgs_per_task_test == 0 else np.random.randint(0, (min_ds_len - self._imgs_per_task_test) // self._batch_size)
        self._dataloaders = {task:DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for (task,ds) in zip(self._tasks, test_datasets)}

        self.KL = KLDivLoss(reduce="batchmean")

        # sanity checks
        assert self._imgs_per_task_test <= min_ds_len

    def execute(self):
        super().execute()
        
        # foreach task
        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c = [], [], [], [], [], [], []
        for task,dataloader in self._dataloaders.items():
            loss, NSS, CC, SIM, KL, KL_c = self._test_one_task(task, dataloader)

            all_names.append(task)
            all_loss.append(loss)
            all_NSS.append(NSS)
            all_CC.append(CC)
            all_SIM.append(SIM)
            all_KL.append(KL)
            all_KL_c.append(KL_c)

        # total
        loss, NSS, CC, SIM, KL, KL_c = np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL)), np.nanmean(np.asarray(all_KL_c))
        all_names.append("Average")
        all_loss.append(loss)
        all_NSS.append(NSS)
        all_CC.append(CC)
        all_SIM.append(SIM)
        all_KL.append(KL)
        all_KL_c.append(KL_c)

        # save test results
        stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "test_results").replace("\\", "/")
        data = np.array([all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c]).transpose().tolist()
        table = wandb.Table(data=data, columns=["Task", "Loss", "NSS", "CC", "SIM", "KL", "KL custom"])
        wandb.run.log({stats_file:table})

        return self._model

    def cleanup(self):
        super().cleanup()

        del self._dataloaders