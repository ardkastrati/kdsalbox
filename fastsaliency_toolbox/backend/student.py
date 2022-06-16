"""
Student
-------

The pytorch network that is used to approximate the original models.

Architecture:
    - Encoder (MobilenetV2 features)
    - Decoder (CNN with BN layers and upsampling to achieve original input shape)
    - Sigmoid Activation Function

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bn7_3 = nn.BatchNorm2d(num_features=512)
        self.bn8_1 = nn.BatchNorm2d(num_features=256)
        self.bn8_2 = nn.BatchNorm2d(num_features=256)
        self.bn9_1 = nn.BatchNorm2d(num_features=128)
        self.bn9_2 = nn.BatchNorm2d(num_features=128)
        self.bn10_1 = nn.BatchNorm2d(num_features=64)
        self.bn10_2 = nn.BatchNorm2d(num_features=64)
        self.drop_layer = nn.Dropout2d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=False)
        self.conv7_3 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, xb):
        xb = F.relu(self.bn7_3(self.conv7_3(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn8_1(self.conv8_1(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn8_2(self.conv8_2(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn9_1(self.conv9_1(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn9_2(self.conv9_2(xb)))
        xb = self.upsample(xb)
        xb = F.relu(self.bn10_1(self.conv10_1(xb)))
        xb = F.relu(self.bn10_2(self.conv10_2(xb)))
        return self.output(xb)


class student(nn.Module):
    def __init__(self):
        super(student, self).__init__()
        self.activation = {}
        self.encoder = self.mobilenetv2_pretrain()
        self.decoder = self.simple_decoder()
        self.sigmoid = nn.Sigmoid()


    def mobilenetv2_pretrain(self, pretrained=True, forward_hook_index=range(1,19,1)):
        model = mobilenet_v2(pretrained=pretrained, progress=False)
        features = list(model.features)
        if forward_hook_index is not None:
            self.register_layers(features, forward_hook_index, 'student_encoder')

        features = nn.Sequential(*features)
        return features

    def simple_decoder(self):
        return Decoder()

    def forward(self, inputs):
        enc = self.encoder(inputs)
        student_e = self.get_student_features(range(0, 18, 1), 'student_encoder')
        self.last = student_e[17]
        dec = self.decoder(enc)
        prob = self.sigmoid(dec)
        return prob

    def register_layers(self, model, name_list, prefix):
        for i, idx in enumerate(name_list):
            model[idx].register_forward_hook(self.get_activation(prefix+'_{}'.format(i)))
        return model

    def get_student_features(self, name_list, prefix):
        data = []
        for name in name_list:
            data.append(self.activation[prefix+'_{}'.format(name)])
        return data

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(True)

    
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

if __name__ == '__main__':
    m = student()
    m.eval()

    model = student()
    checkpoint = torch.load("../example.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['student_model'])
    model.eval()