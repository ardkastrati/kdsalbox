""" 
API that facilitates the use of layers with external weights in models. 
A CustomWeightsLayer is basically a model that can have
    - normal nn.Module layers with their own parameters
    - a leaf cwl that requires a given amount of external weights fed into the forward call
    - child cwls (will be handled recursively)

The layers are called in sequential order of registration.

"""

from typing import Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

####################
#    INTERFACES    #
####################

class CustomWeightsLayer(nn.Module):
    """
        Class that manages a collection of CustomWeightsLayer and nn.Module
    """
    def __init__(self):
        super(CustomWeightsLayer, self).__init__()
        self._cw_layers = nn.ModuleDict()
        self._cw_param_shapes : List[torch.Size] = []
        self._named_cw_param_shapes : List[Tuple[str, torch.Size]] = []

    def forward(self, x, weights):
        index = 0
        for cwl in self._cw_layers.values():
            cws = len(cwl.get_cw_param_shapes())
            x = cwl(x, weights[index:index+cws])
            index += cws

        return x
    
    def add_cwl(self, layer, name=None):
        """ 
            Register either a CustomWeightsLayer or some standard nn.Module as a sub-layer of this layer.
            If the layer is a nn.Module then it is assumed that all weights are maintained internally.
            The layers will be applied in the order in which they are added.
        """
        if name is None:
            name = f"{len(self._cw_layers)}"

        if isinstance(layer, CustomWeightsLayer):
            self._cw_layers[name] = layer
        else:
            self._cw_layers[name] = _LeafModule(layer)
        
        return self

    def get_layer(self, name:str):
        return self._cw_layers[name]

    def compute_cw_param_shapes(self, name_prefix : str = ""):
        self._cw_param_shapes = []
        self._named_cw_param_shapes = []

        for cwl_name in self._cw_layers.keys():
            cwl = self._cw_layers[cwl_name]

            # compute the cw param shapes recursively
            cwl.compute_cw_param_shapes(f"{name_prefix}.{cwl_name}") 

            self._cw_param_shapes.extend(cwl.get_cw_param_shapes())
            self._named_cw_param_shapes.extend(cwl.get_named_cw_param_shapes())

    def get_cw_param_shapes(self):
        return self._cw_param_shapes
    
    def get_named_cw_param_shapes(self):
        return self._named_cw_param_shapes

class _LeafModule(CustomWeightsLayer):
    """
        Wraps a nn.Module in a CWL
        Does not treat any of the nn.Module's parameters as external/custom
        Simply calls the layer in the forward pass

        Cannot add sub-layers
    """
    def __init__(self, layer):
        super(_LeafModule, self).__init__()
        
        self.layer = layer

        self._cw_param_shapes = []
        self._named_cw_param_shapes = []

    def add_cwl(self, layer): 
        raise RuntimeError("Should not call register_layer on a _ModuleWrapper")

    def forward(self, x, weights):
        return self.layer(x)

    def compute_cw_param_shapes(self, name_prefix : str = ""):
        # computed once in c'tor
        pass        

class ModuleWrapper(CustomWeightsLayer):
    """
        Wraps a nn.Module in a CWL 
        Will learn all parameters of the blueprint externally
        Simply calls the provided forward method with the external weights

        Cannot add sub-layers
    """
    def __init__(self, blueprint : nn.Module, forward):
        """
            Takes a blueprint nn.Module to automatically get the param size from 
            and expects a forward method that takes a tensor and a list of weights
        """
        super().__init__()

        self.fwd = forward
        self._non_prefixed_named_cw_param_shapes = [(name, p.size()) for name,p in blueprint.named_parameters()]
        self._cw_param_shapes = [size for _,size in self._non_prefixed_named_cw_param_shapes]

        # initially we dont have a prefix on the parameter names
        self._named_cw_prefix = ""
        self._named_cw_param_shapes = self._non_prefixed_named_cw_param_shapes


    def add_cwl(self, layer): 
        raise RuntimeError("Should not call register_layer on a CWLWrapper")
    
    def forward(self, x, weights):
        return self.fwd(x, weights)

    def compute_cw_param_shapes(self, name_prefix : str = ""):
        # update param name prefix
        self._named_cw_prefix = name_prefix
        self._named_cw_param_shapes = [(f"{name_prefix}.{name}", shape) for name,shape in self._non_prefixed_named_cw_param_shapes]

        


#########################
#    FACTORY METHODS    #
#########################

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    """ Builds a Conv2d layer with custom weights """
    with_bias = lambda x, weights : F.conv2d(x, weight=weights[0], bias=weights[1], stride=stride, padding=padding, dilation=dilation, groups=groups)
    without_bias = lambda x, weights : F.conv2d(x, weight=weights[0], stride=stride, padding=padding, dilation=dilation, groups=groups)
    cwl = ModuleWrapper(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        with_bias if bias else without_bias
    )

    return cwl

def batchnorm2d(num_features):
    """ Builds a BatchNorm2D layer with custom weights """
    cwl = AffineBatchNorm2d(num_features)
    return cwl

def block(in_channels : int, out_channels : int, non_lin_fn : Callable, interpolate : bool):
    """ Builds a standard conv block containing a Conv2d layer, BatchNorm2d layer, non-linearity (e.g. nn.Sigmoid()) and optionally a interpolation. """
    non_lin = non_lin_fn()
    assert len(list(non_lin.parameters())) == 0, "activation functions with parameters are not yet supported"

    cwl = CustomWeightsLayer()
    cwl.add_cwl(conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    cwl.add_cwl(batchnorm2d(out_channels))
    cwl.add_cwl(non_lin)

    if interpolate:
        cwl.add_cwl(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)) # Upsample has no learnable parameters

    return cwl



#########################
#    STATEFUL LAYERS    #
#########################

class _BN2dCW(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(_BN2dCW, self).__init__(num_features, eps, momentum, False, track_running_stats)

    def forward(self, input, weights):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weights[0],
            weights[1],
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class AffineBatchNorm2d(CustomWeightsLayer):
    def __init__(self, num_features):
        super(AffineBatchNorm2d, self).__init__()

        self._bn = _BN2dCW(num_features)

        blueprint = nn.BatchNorm2d(num_features)
        self._non_prefixed_named_cw_param_shapes = [(name, p.size()) for name,p in blueprint.named_parameters()]
        self._cw_param_shapes = [size for _,size in self._non_prefixed_named_cw_param_shapes]

        # initially we dont have a prefix on the parameter names
        self._named_cw_prefix = ""
        self._named_cw_param_shapes = self._non_prefixed_named_cw_param_shapes
    
    def forward(self, x, weights):
        return self._bn(x, weights)

    def compute_cw_param_shapes(self, name_prefix : str = ""):
        # update param name prefix
        self._named_cw_prefix = name_prefix
        self._named_cw_param_shapes = [(f"{name_prefix}.{name}", shape) for name,shape in self._non_prefixed_named_cw_param_shapes]

    def add_cwl(self, layer, name=None):
        raise RuntimeError("Should not call register_layer on a BatchNorm2d")
    
