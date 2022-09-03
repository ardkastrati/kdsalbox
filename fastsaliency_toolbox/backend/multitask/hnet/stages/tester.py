"""
Tester
------

DESCRIPTION:
    Computes a few metrics [loss, NSS, CC, SIM, KL]
    for all images in a folder and their corresponding saliency maps and reports them.
    By setting conf["test"]["per_image_statistics"], the stats will be produced per image additionally.

RETURN VALUE:
    Same as input

CONFIG:
test:
    tasks                   (List[str]) : all tasks that should be tested

    input_saliencies        (str)       : path to saliency map base folder (base_folder/task/img.jpg)
    input_images_test       (str)       : path to images for testing (folder/img.jpg), img in saliencies

    imgs_per_task_test      (int)       : how many of the available images should be used for testing

    per_image_statistics    (bool)      : generate the stats per image?

"""

import os
import torch
from torch.utils.data import DataLoader

from backend.datasets import TestDataManager
from backend.parameters import ParameterMap
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.models.hyper_model import HyperModel
from backend.multitask.hnet.train_impl_wandb.progress_tracking import TesterProgressTrackerWandb


class Tester(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        test_conf = conf[name]

        self._batch_size = 1 # TODO: add support for batch_size > 1 (make sure per_image_statistics still works!)
        self._imgs_per_task_test = test_conf["imgs_per_task_test"]
        self._tasks = test_conf["tasks"]
        self._input_saliencies = test_conf["input_saliencies"]
        self._input_images_test = test_conf["input_images_test"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._per_image_statistics = test_conf["per_image_statistics"]
        self._batches_per_task_test = self._imgs_per_task_test // self._batch_size

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
            task: DataLoader(ds, batch_size=self._batch_size, shuffle=True, num_workers=4) 
            for (task,ds) in zip(self._tasks, test_datasets)
        }

        self._tester = TesterProgressTrackerWandb(self._dataloaders, self._batches_per_task_test, self._postprocess_parameter_map, self._logging_dir, 
            per_image_statistics=self._per_image_statistics, verbose=self._verbose)

    def execute(self):
        super().execute()
        
        self._model.to(self._device)
        self._tester.track_progress_core(self._model)

        return self._model

    def cleanup(self):
        super().cleanup()

        del self._dataloaders