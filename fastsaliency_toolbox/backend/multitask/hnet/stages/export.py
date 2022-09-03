"""
Export Stage
------------

DESCRIPTION:
    Save the current model to disk or wandb

RETURN VALUE:
    Same as input

CONFIG:
    None

"""


import os
import wandb

from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.models.hyper_model import HyperModel

class ExportStage(AStage):
    def __init__(self, name: str, path : str, verbose: bool = True):
        super().__init__(name, verbose)

        self._path = path
    
    def setup(self, work_dir_path: str = None, input=None):
        assert input is not None and isinstance(input, HyperModel), "Trainer expects a HyperModel to be passed as an input."

        self._model = input

    def execute(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._model.save(self._path)
        wandb.save(self._path, base_path=wandb.run.dir)

        print(f"Exporting model to {self._path}")

        return self._model

    def cleanup(self):
        pass