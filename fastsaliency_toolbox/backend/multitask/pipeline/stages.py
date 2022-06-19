"""
Stages
------

A collection of useful pipeline stages such as
    - ExportStage: Exports a model to a specified path

"""

import wandb
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel


class ExportStage(AStage):
    def __init__(self, name: str, path : str, verbose: bool = True):
        super().__init__(name, verbose)

        self._path = path
    
    def setup(self, work_dir_path: str = None, input=None):
        assert input is not None and isinstance(input, HyperModel), "Trainer expects a HyperModel to be passed as an input."

        self._model = input

    def execute(self):
        self._model.save(self._path)
        wandb.save(self._path, base_path=wandb.run.dir)

        print(f"Exporting model to {self._path}")

        return self._model

    def cleanup(self):
        pass

