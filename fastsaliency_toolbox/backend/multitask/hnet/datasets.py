from typing import Callable, List, Tuple
import torch
from torch.utils.data import Dataset

import backend.student as stud
from backend.multitask.hnet.models.hyper_model import HyperModel

ListOfNamedWeights = List[Tuple[str,torch.Tensor]]
ListOfWeights = List[torch.Tensor]

class WeightDataset(Dataset):
    """ WeightDataset
    
    A dataset with X = the indices of the tasks of the trained models
    and y = the weights of the trained models (for single tasks) 
    [and if rearrange_buffers_fn is not None then it will return a tuple (weights, buffers)]

    Args:
        paths (List[Tuple(int, str)]): 
            the (task_id, path) to all the models that should be loaded
        rearrange_weights_fn (Callable[[ListOfNamedWeights, HyperModel], ListOfWeights]): 
            function that remaps a list of named weights to a list of weights that fits the
            custom weight layers of a hypermodel.
        flatten (bool): if true then the weights will be flattened and concatenated, 
            otherwise they will be returned as a list of weights with their original shapes
        rearrange_buffers_fn (Callable[[ListOfNamedWeights, HyperModel], ListOfWeights]):
            function that remaps a list of named buffers into a new order.
            if None then no buffers will be returned
    
    """
    def __init__(self, 
        paths : List[Tuple[int, str]], 
        rearrange_weights_fn : Callable[[ListOfNamedWeights, HyperModel], ListOfWeights], 
        flatten : bool = False,
        rearrange_buffers_fn : Callable[[ListOfNamedWeights, HyperModel], ListOfWeights] = None):

        self._models = [(task_id, self.load_model_weights(path, rearrange_weights_fn, flatten, rearrange_buffers_fn)) for task_id,path in paths]

    def __getitem__(self, index):
        return self._models[index] # returns tuple (task_id, weight tensor)
    
    def __len__(self):
        return len(self._models)

    def load_model_weights(self, path, rearrange_weights_fn, flatten, rearrange_buffers_fn):
        model = stud.Student()
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['student_model'])

        named_weights = {n:p.data for n,p in model.named_parameters()} # dict(param name, parameter data in original shape)
        weights = rearrange_weights_fn(named_weights, model)

        if flatten:
            weights = [w.view(-1) for w in weights]
            weights = torch.cat(weights)
        
        if rearrange_buffers_fn:
            named_buffers = {n:b.data for n,b in model.named_buffers()} # dict(buffer name, buffer data in original shape)
            buffers = rearrange_buffers_fn(named_buffers, model)
            
            return (weights, buffers)
        
        return weights