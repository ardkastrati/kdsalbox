from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backend.multitask.hnet.hnet_interface import AHNET

class HNET(AHNET):
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        self._task_cnt = hnet_conf["task_cnt"]
        self._target_shapes = target_shapes
        self._target_numels = [shape.numel() for shape in target_shapes]

        total_weights = 0
        for n in self._target_numels:
            total_weights += n

        self._l1 = nn.Linear(self._task_cnt, total_weights, bias=False)

    def forward(self, task_id : int):
        device = list(self.parameters())[0].device
        x = F.one_hot(torch.LongTensor(np.array([task_id])), self._task_cnt)[0].float().to(device) # one hot encoding of cond_id/task_id
        x = self._l1(x)

        # make the HNET output the specified target shapes
        weights = list(torch.split(x, split_size_or_sections=self._target_numels))
        for i,s in enumerate(self._target_shapes):
            weights[i] = weights[i].view(s)

        return weights

    def freeze_hnet_for_catchup(self):
        # no catchup parameters
        pass

    def unfreeze_hnet_from_catchup(self):
        # no catchup parameters
        pass

    def get_gradients_on_outputs(self) -> Dict[int, List[torch.Tensor]]:
        weight_param = self.get_parameter("_l1.weight")
        grads = weight_param.grad.detach() # has shape (total_weights, task_cnt)

        grads_per_task = {}
        for task_id in range(self._task_cnt):
            grad = grads[:,task_id] # has shape (total_weights)
            grad_per_target = list(grad.split(self._target_numels))
            grads_per_task[task_id] = grad_per_target

        return grads_per_task