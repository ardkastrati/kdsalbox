"""
Contains a bunch of hypernetwork architectures

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from hypnettorch.hnets.mlp_hnet import HMLP

from backend.multitask.hnet.hnet_interface import AHNET


class ChunkedHNET(AHNET):
    """
    ChunkedHNET
    -----------

    Hypernetwork that uses chunking of the outputs to be more compact.
    In essence the outputs are split into equally sized chunks that are generated
    in a for loop. Each iteration gets additionally fed a chunk embedding (analog to the index of the iteration).

    """
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        self.hnet = ChunkedHMLP(
            target_shapes=target_shapes, 
            chunk_size=hnet_conf["chunk_size"],
            layers=hnet_conf["hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
            cond_in_size=hnet_conf["embedding_size"], # the size of the embeddings
            num_cond_embs=hnet_conf["task_cnt"], # the number of embeddings we want to learn
            cond_chunk_embs = hnet_conf["chunk_emb_per_task"], # chunk embeddings depend on task id
            chunk_emb_size=hnet_conf["chunk_emb_size"] # size of the chunk embeddings
        )

    def forward(self, task_id : Union[int, List[int]]):
        return self.hnet.forward(cond_id = task_id)

    def freeze_hnet_for_catchup(self):
        print([n for n,p in self.hnet.named_parameters()])
        
    def unfreeze_hnet_from_catchup(self):
        pass

    def get_gradients_on_outputs(self) -> Dict[int, List[torch.Tensor]]:
        print("Warning: observing gradients of ChunkedHLMP is currently not yet supported!")
        return {}




class SimpleHNET(AHNET):
    """
    SimpleHNET
    ----------

    HNET that outputs all the weights at once.
    Note that the the network will use the conditional task_embedding as an input
    (which will be learned too).

    """
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        self.hnet = HMLP(
            target_shapes=target_shapes, 
            layers=hnet_conf["hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
            cond_in_size=hnet_conf["embedding_size"], # the size of the embeddings
            num_cond_embs=hnet_conf["task_cnt"], # the number of embeddings we want to learn
        )

    def forward(self, task_id : Union[int, List[int]]):
        return self.hnet.forward(cond_id = task_id)

    def freeze_hnet_for_catchup(self):
        print([n for n,p in self.hnet.named_parameters()])
        
    def unfreeze_hnet_from_catchup(self):
        pass

    def get_gradients_on_outputs(self) -> Dict[int, List[torch.Tensor]]:
        print("Warning: observing gradients of SimpleHNET is currently not yet supported!")
        return {}






class SingleLayerHNET(AHNET):
    """
    SingleLayerHNET
    ---------------

    Hypernetwork that uses a one-hot vector representing the task_id and one linear layer
    that goes from <task_cnt> to <amount of MNET weights>

    """
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        self._task_cnt = hnet_conf["task_cnt"]
        self._target_shapes = target_shapes
        self._target_numels = [shape.numel() for shape in target_shapes]

        total_weights = 0
        for n in self._target_numels:
            total_weights += n

        self._l1 = nn.Linear(self._task_cnt, total_weights, bias=False)

    def forward(self, task_id : Union[int, List[int]]):
        device = list(self.parameters())[0].device

        if isinstance(task_id, int):
            return self._generate_weights_for_task_id(task_id, device)
        elif isinstance(task_id, list):
            return [self._generate_weights_for_task_id(tid, device) for tid in task_id]
        else:
            raise TypeError(task_id)
    
    def _generate_weights_for_task_id(self, task_id : int, device):
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