import math
import torch
import biotorch.layers.fa as layers_fa


from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.fa.conv import Conv2dGrad


class Conv2d(layers_fa.Conv2d):
    """
    Implements the method from How Important Is Weight Symmetry in Backpropagation?

    Fixed Random Magnitude Sign-concordant Feedbacks (frSF):
    weight_backward = M ◦ sign(weight), where M is initialized once and fixed throughout each experiment

    (https://arxiv.org/pdf/1510.05067.pdf)
    """
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

    def forward(self, x):
        return Conv2dGrad.apply(x,
                                self.weight,
                                self.weight_backward * torch.sign(self.weight),
                                self.bias,
                                None,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)
