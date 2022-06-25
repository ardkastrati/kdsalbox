"""
HNET
----

General HNET that can take different forms depending on the specified type

"""

from typing import List, Union
import torch

from backend.multitask.hnet.hnet_interface import AHNET
from backend.multitask.hnet.nets.hnets import ChunkedHNET, SimpleHNET, SingleLayerHNET

class HNET(AHNET):
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        hnet_type = hnet_conf["type"]
        if hnet_type == "simple":
            self.hnet = SimpleHNET(target_shapes, hnet_conf)
        elif hnet_type == "chunked":
            self.hnet = ChunkedHNET(target_shapes, hnet_conf)
        elif hnet_type == "single_layer":
            self.hnet == SingleLayerHNET(target_shapes, hnet_conf)
        else:
            raise ValueError(f"Hypernetwork Configuration: Does not support type {hnet_type}")

    def forward(self, task_id : Union[int, List[int]]):
        return self.hnet.forward(task_id)

    def freeze_hnet_for_catchup(self):
        self.hnet.freeze_hnet_for_catchup()
        
    def unfreeze_hnet_from_catchup(self):
        self.hnet.unfreeze_hnet_from_catchup()

    def get_gradients_on_outputs(self) -> List[torch.Tensor]:
        return self.hnet.get_gradients_on_outputs()