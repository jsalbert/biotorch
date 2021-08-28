import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

This code is based in the implementation by Yerlan Idelbayev
(https://github.com/akamaster/pytorch_resnet_cifar10)

"""

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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = IdentityPadding(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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


class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock],
                 num_blocks: list,
                 num_classes: int = 10) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

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
