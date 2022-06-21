import torch.nn as nn
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

import backend.multitask.custom_weight_layers as cwl
from backend.multitask.mobilenetv2_wrapper import mobilenet_v2_pretrained
from backend.multitask.hnet.full.hnet import HNET

class Decoder(cwl.CustomWeightsLayer):
    def __init__(self, mnet_conf):
        super(Decoder, self).__init__()
        non_lins = {
            "LeakyReLU": nn.LeakyReLU(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6()
        }
        non_lin = non_lins[mnet_conf["decoder_non_linearity"]]

        # layers not actually used in forward but providing the param shapes
        self.register_layer(cwl.block(1280, 512, non_lin, interpolate=True))
        self.register_layer(cwl.block(512, 256, non_lin, interpolate=True))
        self.register_layer(cwl.block(256, 256, non_lin, interpolate=True))
        self.register_layer(cwl.block(256, 128, non_lin, interpolate=True))
        self.register_layer(cwl.block(128, 128, non_lin, interpolate=True))
        self.register_layer(cwl.block(128, 64, non_lin, interpolate=False))
        self.register_layer(cwl.block(64, 64, non_lin, interpolate=False))
        self.register_layer(cwl.conv2d(64, 1, kernel_size=1, padding=0))

        # generate the shapes of the params that are learned by the hypernetwork
        self.compute_cw_param_shapes()


class Student(cwl.CustomWeightsLayer):
    def __init__(self, mnet_conf):
        super(Student, self).__init__()

        self.mobilenet_cutoff = mnet_conf["mobilenet_cutoff"]

        self.register_layer(mobilenet_v2_pretrained(self.mobilenet_cutoff), "encoder")
        self.register_layer(Decoder(mnet_conf), "decoder")
        self.register_layer(nn.Sigmoid(), "sigmoid")

        self.compute_cw_param_shapes()

    def freeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.get_layer("encoder").parameters():
            param.requires_grad_(True)


######################
#  LOAD & SAVE MODEL #
######################

# builds a hypernetwork and mainnetwork
def hnet_mnet_from_config(conf):
    model_conf = conf["model"]
    hnet_conf = model_conf["hnet"]
    mnet_conf = model_conf["mnet"]

    mnet = Student(mnet_conf)
    hnet = HNET(
        task_cnt=hnet_conf["task_cnt"],
        target_shapes=mnet.get_cw_param_shapes(), 
    )

    return hnet,mnet