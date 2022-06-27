"""
decoders
--------

Contains a bunch of decoder architectures

FullDecoder: all the weights of the decoder have to be provided externally

"""

import torch.nn as nn

import backend.multitask.custom_weight_layers as cwl

non_lins = {
    "LeakyReLU": lambda : nn.LeakyReLU(),
    "ReLU": lambda : nn.ReLU(),
    "ReLU6": lambda : nn.ReLU6()
}

class FullDecoder(cwl.CustomWeightsLayer):
    def __init__(self, mnet_conf):
        super(FullDecoder, self).__init__()
        
        non_lin_fn = non_lins[mnet_conf["decoder_non_linearity"]]

        # layers not actually used in forward but providing the param shapes
        self.register_layer(cwl.block(1280, 512, non_lin_fn(), interpolate=True))
        self.register_layer(cwl.block(512, 256, non_lin_fn(), interpolate=True))
        self.register_layer(cwl.block(256, 256, non_lin_fn(), interpolate=True))
        self.register_layer(cwl.block(256, 128, non_lin_fn(), interpolate=True))
        self.register_layer(cwl.block(128, 128, non_lin_fn(), interpolate=True))
        self.register_layer(cwl.block(128, 64, non_lin_fn(), interpolate=False))
        self.register_layer(cwl.block(64, 64, non_lin_fn(), interpolate=False))
        self.register_layer(cwl.conv2d(64, 1, kernel_size=1, padding=0))