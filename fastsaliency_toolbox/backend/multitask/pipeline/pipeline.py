"""
Pipeline
--------

Sequence of stages that will be executed one after the other.
The output from the previous will be the input of the next.
Allows for stages to be dynamically excluded by name.

"""

from abc import ABC, abstractmethod
import os
from typing import List

from backend.utils import print_pretty_header

class AStage(ABC):
    def __init__(self, name : str, verbose : bool = True):
        self._name : str = name
        self._verbose : bool = verbose

    @property
    def name(self) -> str:
        """ The name of this pipeline """
        return self._name
    
    def setup(self, work_dir_path : str = None, input = None):
        """ Sets up this pipeline stage before running. 
            Takes in an optional input as well as an optional working directory path. 
        """
        if self._verbose: print(f"Setting up {self._name}")
        if work_dir_path is not None: os.makedirs(work_dir_path, exist_ok=True)

    @abstractmethod
    def execute(self):
        """ Runs this pipeline stage and returns the output of this stage. """
        if self._verbose: print_pretty_header(self._name.upper())
        if self._verbose: print(f"Starting {self._name}")

        return None

    def cleanup(self):
        """ Cleans up after running this pipeline stage """
        if self._verbose: print(f"Cleaning up {self._name}")
        pass


class Pipeline(AStage):
    
    def __init__(self, stages : List[AStage], input=None, work_dir_path : str = None) -> None:
        """
        Args:
            stages (List[AStage]): The stages this pipline consists of
            input (Any): The initial input that will be used in the first stage
            working_dir_path (str): The working directory this pipeline will use for I/O operations
                specific to this pipeline.
        """
        self._stages : List[AStage] = stages
        self._work_dir_path : str = work_dir_path
        self._input = input

    def execute(self, exclude : List[str] = None):
        """ Executes this pipeline stage by stage but excludes all stages in the exclude list (by name). """
        stages = self._stages
        if exclude:
            stages = [s for s in stages if s.name not in exclude]

        current = self._input
        for stage in stages:
            work_dir_path = None if self._work_dir_path is None else os.path.join(self._work_dir_path, stage.name)
            stage.setup(input=current, work_dir_path=work_dir_path)
            current = stage.execute()
            stage.cleanup()

        return current
