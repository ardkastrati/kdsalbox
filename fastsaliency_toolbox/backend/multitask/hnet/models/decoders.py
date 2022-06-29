"""
decoders
--------

Contains a bunch of decoder architectures

FullDecoder: all the weights of the decoder have to be provided externally

"""

import torch.nn as nn

import backend.multitask.hnet.cwl.custom_weight_layers as cwl

non_lins = {
    "LeakyReLU": nn.LeakyReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6
}

class FullDecoder(cwl.CustomWeightsLayer):
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