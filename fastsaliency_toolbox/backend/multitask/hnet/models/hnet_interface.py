"""
AHNET
-----

Abstract hypernetwork class that exposes a default interface
that can be used by trainers for example.

"""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torch.nn as nn

class AHNET(nn.Module, ABC):
    def __init__(self, target_shapes : List[torch.Size]):
        nn.Module.__init__(self)
        ABC.__init__(self)

        self._target_shapes = target_shapes

    @property
    def target_shapes(self):
        return self._target_shapes

    @abstractmethod
    def get_gradients_on_outputs(self) -> Dict[int, List[torch.Tensor]]:
        """ Returns a dictionary that for each task_id contains a list of gradients (representing the different targets = MNET parameters) """
        pass

    @abstractmethod
    def task_parameters(self, task_ids : List[int]) -> List[torch.nn.parameter.Parameter]:
        """ Gets all the parameters that are unique to a list of tasks
            (e.g. the embedding vectors)
        """
        pass
