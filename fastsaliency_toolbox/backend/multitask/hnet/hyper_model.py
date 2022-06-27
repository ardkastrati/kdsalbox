"""
HyperModel
----------

Abstracts a hypernetwork and mainnetwork such that it can be treated as one model.

"""

from typing import Callable, List
import torch

from backend.multitask.hnet.hnet_interface import AHNET
from backend.multitask.custom_weight_layers import CustomWeightsLayer

class HyperModel():
    def __init__(self, 
        mnet_fn : Callable[[], CustomWeightsLayer],
        hnet_fn : Callable[[CustomWeightsLayer], AHNET], 
        tasks : List[str]):
        """
        Args:
            mnet_fn (Function -> mnet): function that returns the mnet when invoked
            hnet_fn (Fuction(mnet) -> hnet): function that builds the hnet from the mnet
            tasks (List[str]): The names of the tasks the network will be learning. Used to create a internal mapping of task-name
                to task-id such that when the model is loaded with a different ordering of tasks it will still work.
        """
        self._tasks = tasks
        self._device = None
        self._mnet_fn = mnet_fn
        self._hnet_fn = hnet_fn
        self.hnet : AHNET = None
        self.mnet : CustomWeightsLayer = None
        self._task_id_map = {t:i for i,t in enumerate(tasks)}

    def build(self, force_rebuild=False):
        """ Actually creates the hypernetwork and mainnetwork in memory """
        # only create model if not created before
        if not force_rebuild and self.hnet is not None and self.mnet is not None: return

        self.mnet = self._mnet_fn()
        self.mnet.compute_cw_param_shapes()

        self.hnet = self._hnet_fn(self.mnet)

        print("Successfully built HNET")
        print("The named parameters that are learned are:")
        for (n,s) in self.mnet.get_named_cw_param_shapes():
            print(f"\t-{n} ({s})")


        return self

    def to(self, device):
        """ Moves the hypernetwork and mainnetwork to a device """
        self.hnet.to(device)
        self.mnet.to(device)

        self._device = device

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

        weights = self.hnet(task_id=task_id)
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
        """ Runs and returns the hypernetwork and mainnetwork on an image for a given task and returns the computed saliency map.
            IMPORTANT: Will not record the gradient! So do not use this for training.
         """
        self.eval()
        with torch.no_grad():
            sal = self(task_id, img)
        return sal

    @property
    def tasks(self):
        """ All the tasks that this model supports """
        return self._tasks

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

    @property
    def device(self):
        return self._device