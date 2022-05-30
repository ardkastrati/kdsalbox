import torch

class HyperModel():
    def __init__(self, hnet_mnet_fn, tasks):
        self._hnet_mnet_fn = hnet_mnet_fn
        self.hnet = None
        self.mnet = None
        self._task_id_map = {t:i for i,t in enumerate(tasks)}

    # actually creates the model in memory
    def build(self):
        hnet,mnet = self._hnet_mnet_fn()
        self.hnet = hnet
        self.mnet = mnet

    # moves the model to a device
    def to(self, device):
        self.hnet.to(device)
        self.mnet.to(device)

    # gets all the parameters of the model
    def parameters(self):
        # learn the params of the hypernetwork as well as the internal params of the main network
        return list(self.hnet.parameters()) + list(self.mnet.parameters())

    def __call__(self, task_id, X):
        weights = self.hnet(cond_id=task_id)
        Y = self.mnet.forward(X, weights=weights)
        return Y

    def train(self):
        self.hnet.train()
        self.mnet.train()
    
    def eval(self):
        self.hnet.eval()
        self.mnet.eval()

    # runs and returns the models on an image for a given task
    def compute_saliency(self, img, task_id):
        self.eval()
        sal = self(task_id, img)
        return sal

    # maps a task name to a task id
    def task_to_id(self, task):
        return self._task_id_map[task]

    # load model state
    def load(self, path, device):    
        d = torch.load(path, map_location=device)
        self.mnet.load_state_dict(d["mnet_model"])
        self.hnet.load_state_dict(d["hnet_model"])

        # custom model/task <-> embedding-id mapping
        if "model_map" in d.keys():
            model_map = d["model_map"]
            self._task_id_map = model_map

    # save model state
    def save(self, path):
        d = {}
        d["model_map"] = self._task_id_map
        d["hnet_model"] = self.hnet.state_dict()
        d["mnet_model"] = self.mnet.state_dict()

        torch.save(d, path)