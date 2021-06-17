import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from biotorch.layers.weight_mirroring import Conv2d, Linear, Sequential


MODE = 'weight-mirroring'
MODE_STRING = 'Weight Mirroring'

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class IdentityPadding(nn.Module):
    def __init__(self, planes):
        self.planes = planes
        super(IdentityPadding, self).__init__()

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 option: str = 'A') -> None:

        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.shortcut = Sequential()

        if stride != 1 or inplanes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = IdentityPadding(planes)
            elif option == 'B':
                self.shortcut = Sequential(
                    Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

    def mirror_weights(self, x: Tensor,
                       mirror_learning_rate: float = 0.01,
                       noise_amplitude: float = 0.1,
                       growth_control: bool = False,
                       damping_factor: float = 0.5):

        # Create input noise
        input_noise_0 = (noise_amplitude * (torch.randn(x.size()))).to(x.device)
        output_noise = self.conv1(input_noise_0)
        # Compute noise correlation and update the backward weight matrix (Backward Matrix)
        self.conv1.update_B(x=input_noise_0, y=output_noise,
                            mirror_learning_rate=mirror_learning_rate,
                            growth_control=growth_control,
                            damping_factor=damping_factor)

        output_noise = self.bn1(output_noise)
        output_noise = self.relu(output_noise)

        # Create input noise
        input_noise = (noise_amplitude * (torch.randn(output_noise.size()))).to(x.device)

        output_noise = self.conv2(input_noise)
        # Compute noise correlation and update the backward weight matrix (Backward Matrix)
        self.conv2.update_B(x=input_noise, y=output_noise,
                            mirror_learning_rate=mirror_learning_rate,
                            growth_control=growth_control,
                            damping_factor=damping_factor)

        output_noise = self.bn2(output_noise)
        output_noise += self.shortcut(input_noise_0)
        output_noise = self.relu2(output_noise)
        return output_noise


class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock],
                 num_blocks: list,
                 num_classes: int = 10) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion

        return Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def mirror_weights(self, x: Tensor,
                       mirror_learning_rate: float = 0.01,
                       noise_amplitude: float = 0.1,
                       growth_control: bool = False,
                       damping_factor: float = 0.5):
        # Create input noise
        input_noise = (noise_amplitude * (torch.randn(x.size()))).to(x.device)
        output_noise = self.conv1(input_noise)
        # Compute noise correlation and update the backward weight matrix (Backward Matrix)
        self.conv1.update_B(x=input_noise, y=output_noise,
                            mirror_learning_rate=mirror_learning_rate,
                            growth_control=growth_control,
                            damping_factor=damping_factor)

        output_noise = self.bn1(output_noise)
        output_noise = self.relu(output_noise)

        output_noise = self.layer1.mirror_weights(output_noise,
                                                  mirror_learning_rate,
                                                  noise_amplitude,
                                                  growth_control,
                                                  damping_factor)
        output_noise = self.layer2.mirror_weights(output_noise,
                                                  mirror_learning_rate,
                                                  noise_amplitude,
                                                  growth_control,
                                                  damping_factor)
        output_noise = self.layer3.mirror_weights(output_noise,
                                                  mirror_learning_rate,
                                                  noise_amplitude,
                                                  growth_control,
                                                  damping_factor)

        output_noise = F.avg_pool2d(output_noise, output_noise.size()[3])
        output_noise = output_noise.view(out.size(0), -1)

        # Create input noise
        input_noise = (noise_amplitude * (torch.randn(output_noise.size()))).to(x.device)
        output_noise = self.fc(input_noise)
        # Compute noise correlation and update the backward weight matrix (Backward Matrix)
        self.fc.update_B(x=input_noise, y=output_noise,
                         mirror_learning_rate=mirror_learning_rate,
                         growth_control=growth_control,
                         damping_factor=damping_factor)


def resnet20(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(pretrained: bool = False, progress: bool = True, num_classes: int = 10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)
