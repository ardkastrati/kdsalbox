"""
ModelLoader
------

DESCRIPTION:
    Loads a model from a wandb save

RETURN VALUE:
    The loaded model

CONFIG:
    load_model:
        run_path: specifies the path of the wandb run
        file_name: specifies the path to the pth

"""

import torch
import wandb

from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.models.hyper_model import HyperModel


class ModelLoader(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        load_conf = conf[name]
        self._run_path = load_conf["run_path"]
        self._file_name = load_conf["file_name"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "ModelLoader expects a non-built HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path


    def execute(self):
        super().execute()
        
        print("Restore model")
        model_file = wandb.restore(self._file_name, run_path=self._run_path)

        print("Load model")
        self._model.build()
        self._model.load(model_file.name, self._device)

        self._model.to(self._device)

        return self._model