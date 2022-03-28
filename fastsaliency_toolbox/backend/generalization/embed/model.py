import torch as t
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
        
    def forward(self, lbl_and_xb):
        # print(self.parameters)                                # DOES contain the linear & embedding layer

        # embedding
        (lbl,xb) = lbl_and_xb
        eb = self.embed(t.tensor(lbl))
        # print(eb)
        eb = F.relu(self.pe_1(eb))
        eb = eb.view((512, 1280, 3, 3))
        # eb.retain_grad()                                      # does not change anything either

        # if self.first_eb0 is not None:
        #      print((self.first_eb0 - eb).norm())              # = 0.0
        # print(eb.requires_grad)                               # = true

        self.conv7_3.weight = nn.Parameter(eb, requires_grad=True)

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


class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
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