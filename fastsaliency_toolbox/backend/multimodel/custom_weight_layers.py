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
        self._cw_param_shapes = []

    def forward(self, x, weights):
        index = 0
        for cwl in self._cw_layers.values():
            cws = len(cwl.get_cw_param_shapes())
            x = cwl(x, weights[index:index+cws])
            index += cws

        return x
    
    def register_layer(self, layer, name=None):
        """ 
            Register either a CustomWeightsLayer or some standard nn.Module as a sub-layer of this layer.
            If the layer is a nn.Module then it is assumed that all weights are maintained internally.
            The layers will be applied in the order in which they are added.
        """
        if name is None:
            name = f"layer_{len(self._cw_layers)}"

        if isinstance(layer, CustomWeightsLayer):
            self._cw_layers[name] = layer
        else:
            self._cw_layers[name] = _LeafModule(layer)

    def get_layer(self, name:str):
        return self._cw_layers[name]

    def compute_cw_param_shapes(self):
        self._cw_param_shapes = []
        for cwl in self._cw_layers.values():
            cwl.compute_cw_param_shapes() # just to make sure all the child layers have been properly initialized
            self._cw_param_shapes.extend(cwl.get_cw_param_shapes())

    def get_cw_param_shapes(self):
        return self._cw_param_shapes

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

        self.compute_cw_param_shapes()

    def register_layer(self, layer): 
        raise RuntimeError("Should not call register_layer on a _ModuleWrapper")

    def forward(self, x, weights):
        return self.layer(x)

    def compute_cw_param_shapes(self):
        self._cw_param_shapes = []

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
        self._cw_param_shapes = [p.size() for p in blueprint.parameters()]

    def register_layer(self, layer): 
        raise RuntimeError("Should not call register_layer on a CWLWrapper")
    
    def forward(self, x, weights):
        return self.fwd(x, weights)

    def compute_cw_param_shapes(self): 
        pass # initialized once in c'tor


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

    cwl.compute_cw_param_shapes()
    return cwl

def batchnorm2d(num_features):
    """ Builds a BatchNorm2D layer with custom weights """
    cwl = BatchNorm2d(num_features)
    return cwl

def block(in_channels, out_channels, non_lin, interpolate):
    """ Builds a standard conv block containing a Conv2d layer, BatchNorm2d layer, non-linearity (e.g. nn.Sigmoid()) and optionally a interpolation. """
    assert len(list(non_lin.parameters())) == 0 # activation functions with parameters are not yet supported

    cwl = CustomWeightsLayer()
    cwl.register_layer(conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    cwl.register_layer(batchnorm2d(out_channels))
    cwl.register_layer(non_lin)

    if interpolate:
        cwl.register_layer(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)) # Upsample has no learnable parameters

    cwl.compute_cw_param_shapes()
    return cwl



#########################
#    STATEFUL LAYERS    #
#########################

# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class _BN2dCW(nn.BatchNorm2d):
    def forward(self, x, weights):
        self._check_input_dim(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = x.mean([0, 2, 3])
            # use biased var in train
            var = x.var([0, 2, 3], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var

        weight = weights[0]
        bias = weights[1]
        return F.batch_norm(x, self.running_mean, self.running_var, weight, bias, self.training, self.momentum, self.eps)

class BatchNorm2d(CustomWeightsLayer):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()

        self._bn = _BN2dCW(num_features, affine=False)

        blueprint = nn.BatchNorm2d(num_features, affine=True)
        self._cw_param_shapes = [p.size() for p in blueprint.parameters()]
    
    def forward(self, x, weights):
        return self._bn(x, weights)

    def compute_cw_param_shapes(self):
        pass

    def register_layer(self, layer, name=None):
        raise RuntimeError("Should not call register_layer on a BatchNorm2d")
    
