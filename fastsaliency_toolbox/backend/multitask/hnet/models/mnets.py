"""
Provides an interface that exposes a most general main network
and a concrete implementation that selects a main network type 
specified in the config.

Also provides actual main network architectures.

TUTORIAL:
    To add new main network architecture:
    Add a new initialization method and add an if case in MNET
    You can now set type=<your new type name> in the config.

TODO: remodel this to use the same design-pattern as the hypernetworks 
(each new architecture has new class). 
Note that this will likely break loading of any model that was trained
using the current implementation because the parameters have different names.

"""

from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

import backend.multitask.hnet.cwl.custom_weight_layers as cwl
from backend.multitask.hnet.cwl.mobilenetv2_wrapper import mobilenet_v2_pretrained

class AMNET(cwl.CustomWeightsLayer, ABC):
    """ Abstract main network class that exposes a default interface
        that can be used by trainers for example. """
    def __init__(self):
        cwl.CustomWeightsLayer.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def freeze_encoder(self):
        """ Freezes the encoder of the main network """
        pass

    @abstractmethod
    def unfreeze_encoder(self):
        """ Unfreezes the encoder of the main network """
        pass

non_lins = {
    "LeakyReLU": nn.LeakyReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6
}

class MNET(AMNET):
    """ MNET that has the same architecture as in the original KDSalBox paper 
        but uses custom weights for all layers of the decoder and optionally also the encoder. """
    def __init__(self, mnet_conf):
        super(MNET, self).__init__()

        mnet_type = mnet_conf["type"]
        # TODO: add other mnet architectures here (e.g. ContextMod or AdaIN architectures)
        if mnet_type == "original":
            self._init_original(mnet_conf)
        else:
            raise ValueError(f"Main Network Configuration: Does not support type {mnet_type}")
        self.compute_cw_param_shapes()
    
    def forward(self, x, weights):
        _,_,h,w = x.shape
        x = super(MNET, self).forward(x, weights)
        return F.upsample_bilinear(x, (h,w)) # rescale to original size

    def _init_original(self, mnet_conf):
         # encoder
        enc_type = mnet_conf["encoder_type"]
        if enc_type == "cwl":
            mobilenet_cutoff = mnet_conf["mobilenet_cutoff"]
            self.add_cwl(self._get_cwl_mobilenet(mobilenet_cutoff), "encoder")
        elif enc_type == "mv2":
            self.add_cwl(self._get_mobilenet(), "encoder")
        else:
            raise ValueError(f"Main network configuration: Does not support encoder_type {enc_type}")
            
        # decoder
        dec_type = mnet_conf["decoder_type"]
        if dec_type == "full":
            self.add_cwl(FullDecoder(mnet_conf), "decoder")
        else:
            raise ValueError(f"Main network configuration: Does not support decoder_type {dec_type}")

        # activation function
        self.add_cwl(nn.Sigmoid(), "sigmoid")


    def _get_mobilenet(self):
        return nn.Sequential(*list(mobilenet_v2(pretrained=True).features))

    def _get_cwl_mobilenet(self, cutoff : int):
        return mobilenet_v2_pretrained(cutoff)

    def freeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(True)

class FullDecoder(cwl.CustomWeightsLayer):
    """ Decoder architecture just like in the original KDSalBox paper but supporting custom weights. """
    def __init__(self, mnet_conf):
        super(FullDecoder, self).__init__()
        
        non_lin_fn = non_lins[mnet_conf["decoder_non_linearity"]]

        # layers not actually used in forward but providing the param shapes
        self.add_cwl(cwl.block(1280, 512, non_lin_fn, interpolate=True))\
            .add_cwl(cwl.block(512, 256, non_lin_fn, interpolate=True))\
            .add_cwl(cwl.block(256, 256, non_lin_fn, interpolate=True))\
            .add_cwl(cwl.block(256, 128, non_lin_fn, interpolate=True))\
            .add_cwl(cwl.block(128, 128, non_lin_fn, interpolate=True))\
            .add_cwl(cwl.block(128, 64, non_lin_fn, interpolate=False))\
            .add_cwl(cwl.block(64, 64, non_lin_fn, interpolate=False))\
            .add_cwl(cwl.conv2d(64, 1, kernel_size=1, padding=0))