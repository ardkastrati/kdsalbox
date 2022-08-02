"""
Runner
------

For each task in conf["tasks"], 
this runner will go over all images in conf["run"]["input_images_run"]
and compute & store the saliency map for each of them.

"""

import torch
from torch.utils.data import DataLoader

from backend.parameters import ParameterMap
from backend.datasets import RunDataManager
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.models.hyper_model import HyperModel
from backend.multitask.hnet.train_impl_wandb.progress_tracking import RunProgressTrackerWandb

class Runner(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        run_conf = conf[name]
        self._tasks = run_conf["tasks"]
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._input_dir = run_conf["input_images_run"]
        self._overwrite = run_conf["overwrite"]

        # convert params
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Runner expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._model.build()
        self._logging_dir = work_dir_path

        # prepare dataloader
        self._dataloader = DataLoader(RunDataManager(self._input_dir, "", self._verbose, recursive=False), batch_size=1)

        self._runner = RunProgressTrackerWandb(self._dataloader, self._tasks, self._postprocess_parameter_map, self._name)


    def execute(self):
        super().execute()
        
        self._model.to(self._device)
        self._runner.track_progress_core(self._model, self._name)

        return self._model

    def cleanup(self):
        super().cleanup()

        del self._dataloader