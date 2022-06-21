"""
HyperModel
----------

Abstracts a hypernetwork and mainnetwork such that it can be treated as one model.

"""

from typing import Callable, Dict, List, Tuple
import torch
import torch.nn as nn

class HyperModel():
    def __init__(self, 
        hnet_mnet_fn : Callable[[Dict], Tuple[nn.Module, nn.Module]], 
        tasks : List[str]):
        """
        Args:
            hnet_mnet_fn (Function -> (hnet, mnet)): Factory function that creates the hypernetwork and the main-network when invoked.
            tasks (List[str]): The names of the tasks the network will be learning. Used to create a internal mapping of task-name
                to task-id such that when the model is loaded with a different ordering of tasks it will still work.
        """
        self._hnet_mnet_fn = hnet_mnet_fn
        self.hnet : nn.Module = None
        self.mnet : nn.Module = None
        self._task_id_map = {t:i for i,t in enumerate(tasks)}

    def build(self):
        """ Actually creates the hypernetwork and mainnetwork in memory """
        hnet,mnet = self._hnet_mnet_fn()
        self.hnet = hnet
        self.mnet = mnet
        return self

    def to(self, device):
        """ Moves the hypernetwork and mainnetwork to a device """
        self.hnet.to(device)
        self.mnet.to(device)

    def parameters(self):
        """ Gets all the parameters of the model
            (all the parameters of the hypernetwork as well as all trainable parameters of the mainnetwork).
        """
        return list(self.hnet.parameters()) + list(self.mnet.parameters())

    def __call__(self, task_id, X):
        """ Invokes a forward pass on the hypernetwork and uses the produced weights to invoke a forward pass on the mainnetwork. """

        # if the model has not been built then do so
        if (self.hnet is None or self.mnet is None): 
            self.build()

        weights = self.hnet(cond_id=task_id)
        Y = self.mnet.forward(X, weights=weights)
        return Y

    def train(self):
        """ Put the hypernetwork and mainnetwork into train mode. """
        self.hnet.train()
        self.mnet.train()
    
    def eval(self):
        """ Put the hypernetwork and mainnetwork into eval mode. """
        self.hnet.eval()
        self.mnet.eval()

    def compute_saliency(self, img : torch.Tensor, task_id : int) -> torch.Tensor:
        """ Runs and returns the hypernetwork and mainnetwork on an image for a given task and returns the computed saliency map. """
        self.eval()
        with torch.no_grad():
            sal = self(task_id, img)
        return sal

    def task_to_id(self, task):
        """ Maps a task name to a task id. """
        return self._task_id_map[task]

    def load(self, path, device):    
        """ Load the state of both the hypernetwork and the mainnetwork as well as the mapping of the task names to ids from a given filepath. """
        d = torch.load(path, map_location=device)
        self.mnet.load_state_dict(d["mnet_model"])
        self.hnet.load_state_dict(d["hnet_model"])

        # custom model/task <-> embedding-id mapping
        if "model_map" in d.keys():
            model_map = d["model_map"]
            self._task_id_map = model_map

    def save(self, path):
        """ Save teh state of both the hypernetwork and the mainnetwork as well as the mapping of the task names to ids to a given filepath. """
        d = {}
        d["model_map"] = self._task_id_map
        d["hnet_model"] = self.hnet.state_dict()
        d["mnet_model"] = self.mnet.state_dict()

        torch.save(d, path)