from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HNET(nn.Module):
    def __init__(self, task_cnt, target_shapes : List[torch.Size]):
        super().__init__()

        self._task_cnt = task_cnt
        self._target_shapes = target_shapes
        self._target_numels = [shape.numel() for shape in target_shapes]

        total_weights = 0
        for n in self._target_numels:
            total_weights += n

        self._l1 = nn.Linear(task_cnt, total_weights)

    def forward(self, cond_id : int):
        device = list(self.parameters())[0].device
        x = F.one_hot(torch.LongTensor(np.array([cond_id])), self._task_cnt)[0].float().to(device) # one hot encoding of cond_id/task_id
        x = self._l1(x)

        # make the HNET output the specified target shapes
        weights = list(torch.split(x, split_size_or_sections=self._target_numels))
        for i,s in enumerate(self._target_shapes):
            weights[i] = weights[i].view(s)

        return weights
