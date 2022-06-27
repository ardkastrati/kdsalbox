"""
MNET
----

The main network. Check out the config for all the available parameters.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2

import backend.multitask.custom_weight_layers as cwl
from backend.multitask.hnet.nets.decoders import FullDecoder
from backend.multitask.mobilenetv2_wrapper import mobilenet_v2_pretrained

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
    
    def get_mobilenet(self):
        return nn.Sequential(*mobilenet_v2(pretrained=True).features)

    def get_cwl_mobilenet(self, cutoff : int):
        return mobilenet_v2_pretrained(cutoff)

    def freeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(True)