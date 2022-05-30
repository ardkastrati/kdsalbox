from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import backend.multimodel.custom_weight_layers as cwl

def mobilenet_v2_pretrained(cutoff=0):
    mv2 = mobilenet_v2(pretrained=True, progress=False)
    mv2f = MobileNetV2Features(cutoff)

    # copy all the weights of all parameters that are not learned externally
    # CAUTION: this requires MobileNetV2Features to have the same structure as MobilenetV2 
    # and the layers that aren't learned by the HNET should be the last n consecutive layers of the mobilenet
    with torch.no_grad():
        mv2_parameters = list(mv2.parameters())
        for i,p in enumerate(mv2f.parameters()):
            p.copy_(mv2_parameters[i])

    return mv2f

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

class ConvNormActivationCWL(cwl.CustomWeightsLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = cwl.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.register_layer(cwl.conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=norm_layer is None))
        if norm_layer is not None:
            self.register_layer(norm_layer(out_channels))
        if activation_layer is not None:
            self.register_layer(activation_layer(inplace=inplace))
        self.out_channels = out_channels

class InvertedResidualCWL(cwl.CustomWeightsLayer):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidualCWL, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = cwl.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio != 1:
            # pw
            self.register_layer(ConvNormActivationCWL(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        
        # dw
        self.register_layer(ConvNormActivationCWL(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        # pw-linear
        self.register_layer(cwl.conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        self.register_layer(norm_layer(oup))
        
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor, weights) -> torch.Tensor:
        if self.use_res_connect:
            return x + super().forward(x, weights)
        else:
            return super().forward(x, weights)


class MobileNetV2Features(cwl.CustomWeightsLayer):
    def __init__(self, cutoff=0):
        super(MobileNetV2Features, self).__init__()

        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = 32
        last_channel = 1280

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        index = 0
        if index >= cutoff:
            self.register_layer(ConvNormActivationCWL(3, input_channel, stride=2, norm_layer=cwl.BatchNorm2d, activation_layer=nn.ReLU6))
        else:
            self.register_layer(ConvNormActivation(3, input_channel, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6))
        
        # building inverted residual blocks
        index = 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1

                if index >= cutoff:
                    self.register_layer(InvertedResidualCWL(input_channel, output_channel, stride, expand_ratio=t, norm_layer=cwl.BatchNorm2d))
                else: 
                    self.register_layer(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, norm_layer=nn.BatchNorm2d))
                input_channel = output_channel
                index += 1
            
        # building last several layers
        if index >= cutoff:
            self.register_layer(ConvNormActivationCWL(input_channel, self.last_channel, kernel_size=1, norm_layer=cwl.BatchNorm2d, activation_layer=nn.ReLU6))
        else:
            self.register_layer(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6))
        index += 1

        # weight initialization
        # TODO: make this work if you want to use a non-pretrained hypernetwork
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

        self.compute_cw_param_shapes()

def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
