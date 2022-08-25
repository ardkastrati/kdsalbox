"""
MNET
----

The main network. Check out the config for all the available parameters.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

import backend.multitask.hnet.cwl.custom_weight_layers as cwl
from backend.multitask.hnet.models.decoders import FullDecoder
from backend.multitask.hnet.cwl.mobilenetv2_wrapper import mobilenet_v2_pretrained

class MNET(cwl.CustomWeightsLayer):
    def __init__(self, mnet_conf):
        super(MNET, self).__init__()

        # encoder
        enc_type = mnet_conf["encoder_type"]
        if enc_type == "cwl":
            mobilenet_cutoff = mnet_conf["mobilenet_cutoff"]
            self.add_cwl(self.get_cwl_mobilenet(mobilenet_cutoff), "encoder")
        elif enc_type == "mv2":
            self.add_cwl(self.get_mobilenet(), "encoder")
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

        self.compute_cw_param_shapes()
    
    def forward(self, x, weights):
        _,_,h,w = x.shape
        x = super(MNET, self).forward(x, weights)
        return F.upsample_bilinear(x, (h,w)) # rescale to original size

    def get_mobilenet(self):
        return nn.Sequential(*list(mobilenet_v2(pretrained=True).features))

    def get_cwl_mobilenet(self, cutoff : int):
        return mobilenet_v2_pretrained(cutoff)

    def freeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(True)