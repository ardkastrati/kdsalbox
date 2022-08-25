"""
Computes the final metrics [KL, KL std, CC, CC std] for a model, that can then be used to generate the final scores as in Ards paper.

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from backend.image_processing import process
from backend.metrics import KL, CC
from backend.datasets import TestDataManager
from backend.parameters import ParameterMap
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.models.hyper_model import HyperModel


class TesterFinal(AStage):
    def __init__(self, conf, input_saliencies, input_images, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        eval_conf = conf["eval"]

        self._batch_size = 1 # TODO: add support for batch_size > 1 (make sure per_image_statistics still works!)
        self._tasks = eval_conf["tasks"]
        self._input_saliencies = input_saliencies
        self._input_images_test = input_images

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"

        # convert to pre-/postprocess params
        self._preprocess_parameter_map = ParameterMap().set_from_dict(conf["preprocess"])
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])
    
        
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Tester expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._model.build()
        self._logging_dir = work_dir_path

        # data loading
        sal_folders = [os.path.join(self._input_saliencies, task) for task in self._tasks] # path to saliency folder for all models
        test_datasets = [TestDataManager(self._input_images_test, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        self._dataloaders = { 
            task: DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) 
            for (task,ds) in zip(self._tasks, test_datasets)
        }

    def execute(self):
        super().execute()
        
        self._model.to(self._device)
        
        all_names, all_CC_mean, all_CC_std, all_KL_mean, all_KL_std = [], [], [], [], []
        for task,dataloader in self._dataloaders.items():
            with torch.no_grad():
                CC_mean, CC_std, KL_mean, KL_std = self._test_one_task(task, self._model, dataloader)

                all_names.append(task)
                all_CC_mean.append(CC_mean)
                all_CC_std.append(CC_std)
                all_KL_mean.append(KL_mean)
                all_KL_std.append(KL_std)

        np.savetxt(os.path.join(self._logging_dir, 'test_results.csv'), np.array([all_names, all_CC_mean, all_CC_std, all_KL_mean, all_KL_std]).T, fmt='%s', delimiter=',', header='Model,CC_mean,CC_std,KL_mean,KL_std', comments='')

        return self._model

    def _test_one_task(self, task : str, model : HyperModel, dataloader : DataLoader):
        model.eval()
        task_id = model.task_to_id(task)
        device = model.device

        if self._verbose: print(f"Testing {task}...")

        all_names, all_CC, all_KL = [], [], []
        loss_fn = torch.nn.BCELoss()

        for i,(X, y, names) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(task_id, X)
            
            _,_,h,w = X.shape
            pred = F.interpolate(pred, size=(h,w)) # make sure the output has the same size as the input

            loss = loss_fn(pred, y)

            detached_pred_t = pred.cpu().detach()
            detached_y_t = y.cpu().detach()
            detached_pred = detached_pred_t.numpy()
            detached_y = detached_y_t.numpy()

            # Doing the postprocessing steps needed for the metrics (We might want to do this also for Task Evaluation stuff?)
            y = process(y, self._postprocess_parameter_map)
            detached_pred = process(detached_pred, self._postprocess_parameter_map)
            CCes = [CC(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            KLes = [KL(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]

            with torch.no_grad():
                all_names.extend(names)
                all_CC.extend(CCes)
                all_KL.extend(KLes)

            if torch.cuda.is_available():
                # Remove batch from gpu
                del pred
                del loss
                del X
                del y
                torch.cuda.empty_cache()

            if i%100 == 0:
                print(f"Batch {i}", flush=True)
            
        return np.nanmean(np.asarray(all_CC)), np.nanstd(np.asarray(all_CC)), np.nanmean(np.asarray(all_KL)), np.nanstd(np.asarray(all_KL))

    def cleanup(self):
        super().cleanup()

        del self._dataloaders