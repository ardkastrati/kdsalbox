"""
AHNET
-----

Abstract hypernetwork class that exposes a default interface
that can be used by trainers for example.

"""

from abc import ABC, abstractmethod
from typing import List
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
    def freeze_hnet_for_catchup(self):
        pass

    @abstractmethod
    def unfreeze_hnet_from_catchup(self):
        pass
