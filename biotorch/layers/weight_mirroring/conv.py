import torch
import torch.nn as nn


from biotorch.autograd.fa.conv import Conv2dGrad
from torch.nn.common_types import _size_2_t
from typing import Union


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

    def update_B(self, weight_update, damping_factor=0.5):
        self.weight_backward += weight_update
        # Prevent feedback weights growing too large
        x = torch.randn(self.weight_backward.size())
        y = torch.matmul(self.weight_backward, x)
        y_std = torch.mean(torch.std(y, axis=0))
        self.weight_backward = nn.Parameter(damping_factor * self.weight_backward / y_std, requires_grad=False)

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
