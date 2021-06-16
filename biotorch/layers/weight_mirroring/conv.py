import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.fa.conv import Conv2dGrad
from biotorch.layers.metrics import compute_matrix_angle


class Conv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'
    ):

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_backward)
        self.bias_backward = None
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_backward, 1)

        self.alignment = 0

    def update_B(self,
                 x,
                 y,
                 mirror_learning_rate=0.01,
                 growth_control=True,
                 damping_factor=0.5):

        # Compute correlation and update the backward weight matrix (FA Matrix)
        dW = mirror_learning_rate * F.conv2d(torch.transpose(x, 0, 1), torch.transpose(y, 0, 1), dilation=self.stride, padding=self.padding)
        dB = torch.transpose(F.interpolate(dW, size=(self.weight.size()[2], self.weight.size()[3])), 0, 1)
        self.weight_backward += torch.nn.Parameter(dB)

        # Prevent feedback weights growing too large
        if growth_control:
            x = torch.randn(x.size())
            y = F.conv2d(x, self.weight_backward)
            # Mean of the standard deviation of the output per every channel
            y_std = torch.mean(torch.std(y, axis=0), axis=[1, 2])
            # Broadcast y_std and normalize across channels
            self.weight_backward = nn.Parameter(damping_factor * (self.weight_backward.T / y_std).T, requires_grad=False)

    def forward(self, x):
        # Linear Feedback Alignment Backward
        return Conv2dGrad.apply(x,
                                self.weight,
                                self.weight_backward,
                                self.bias,
                                self.bias_backward,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)

    def compute_alignment(self):
        self.alignment = compute_matrix_angle(self.weight_backward, self.weight.T)
        return self.alignment
