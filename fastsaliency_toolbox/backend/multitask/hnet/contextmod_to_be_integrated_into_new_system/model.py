import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from hypnettorch.hnets import HMLP
from hypnettorch.mnets.mnet_interface import MainNetInterface, ContextModLayer

class Decoder(nn.Module):
    def __init__(self, mnet_conf, has_bias):
        super(Decoder, self).__init__()
        self._has_bias = has_bias

        non_lins = {
            "LeakyReLU": nn.LeakyReLU(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6()
        }
        self.non_lin = non_lins[mnet_conf["decoder_non_linearity"]]

        self.bn7_3 = nn.BatchNorm2d(num_features=512)
        self.bn8_1 = nn.BatchNorm2d(num_features=256)
        self.bn8_2 = nn.BatchNorm2d(num_features=256)
        self.bn9_1 = nn.BatchNorm2d(num_features=128)
        self.bn9_2 = nn.BatchNorm2d(num_features=128)
        self.bn10_1 = nn.BatchNorm2d(num_features=64)
        self.bn10_2 = nn.BatchNorm2d(num_features=64)
    
        self.conv7_3 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        

        # layers not actually used in forward but providing the param shapes
        
        # 1 gain and shift per featuremap
        self.cm7 = ContextModLayer((512, 1, 1), no_weights=True) # no_weights=True bc generated by hypernetwork
        self.cm8 = ContextModLayer((256, 1, 1), no_weights=True)
        self.cm9 = ContextModLayer((128, 1, 1), no_weights=True)
        self.cm10 = ContextModLayer((64, 1, 1), no_weights=True)

        self._external_param_shapes = []
        self._external_param_shapes.extend(self.cm7.param_shapes)
        self._external_param_shapes.extend(self.cm8.param_shapes)
        self._external_param_shapes.extend(self.cm9.param_shapes)
        self._external_param_shapes.extend(self.cm10.param_shapes)

        
    def forward(self, xb, weights):

        # prepare weights for cm layers
        cm_weights = []
        for i in range(0,len(weights),2):
            cm_weights.append((weights[i], weights[i+1]))


        xb = self.non_lin(self.bn7_3(self.cm7(self.conv7_3(xb), cm_weights[0])))
        xb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)
        xb = self.non_lin(self.bn8_1(self.conv8_1(xb)))
        xb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)
        xb = self.non_lin(self.bn8_2(self.cm8(self.conv8_2(xb), cm_weights[1])))
        xb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)

        xb = self.non_lin(self.bn9_1(self.conv9_1(xb)))
        xb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)
        xb = self.non_lin(self.bn9_2(self.cm9(self.conv9_2(xb), cm_weights[2])))
        xb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)

        xb = self.non_lin(self.bn10_1(self.conv10_1(xb)))
        xb = self.non_lin(self.bn10_2(self.cm10(self.conv10_2(xb), cm_weights[3])))

        xb = self.output(xb)
        return xb

    # gets the shape of the parameters of which we expect the weights to be generated by the hypernetwork
    def get_external_param_shapes(self):
        return self._external_param_shapes


class Student(nn.Module, MainNetInterface):
    def __init__(self, mnet_conf):
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        # MNET setup
        self._has_fc_out = False 
        self._mask_fc_out = False
        self._has_linear_out = False
        self._has_bias = True

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        # build model
        self.encoder = self.mobilenetv2_pretrain()
        self.decoder = Decoder(mnet_conf, has_bias=self._has_bias)
        self.sigmoid = nn.Sigmoid()

        # params that will be trained by the hypernetwork
        self._external_param_shapes = self.decoder.get_external_param_shapes()

        # params that will be trained by the model itself
        self._internal_params = list(self.parameters())
        self._param_shapes = []
        for param in self._internal_params:
            self._param_shapes.append(list(param.size()))


        self._is_properly_setup()

    def forward(self, xb, weights):
        enc = self.encoder(xb)
        dec = self.decoder(enc, weights)
        prob = self.sigmoid(dec)
        return prob


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(True)
    
    def external_param_shapes(self):
        return self._external_param_shapes

    ####################
    # MainNetInterface #
    ####################

    def distillation_targets(self):
        return None # we do not have any distillation targets

    #########
    # Other #
    #########

    def mobilenetv2_pretrain(self, pretrained=True):
        model = mobilenet_v2(pretrained=pretrained, progress=False)
        features = nn.Sequential(*list(model.features))
        return features



######################
#  LOAD & SAVE MODEL #
######################

# builds a hypernetwork and mainnetwork
def hnet_mnet_from_config(conf):
    model_conf = conf["model"]
    hnet_conf = model_conf["hnet"]
    mnet_conf = model_conf["mnet"]

    mnet = Student(mnet_conf)
    hnet = HMLP(
        mnet.external_param_shapes(), 
        layers=hnet_conf["hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
        cond_in_size=hnet_conf["embedding_size"], # the size of the embeddings
        num_cond_embs=hnet_conf["task_cnt"] # the number of embeddings we want to learn
    )

    return hnet,mnet