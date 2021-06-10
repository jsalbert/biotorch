import torch.nn as nn


from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.sign_2.conv import Conv2dGrad


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

        nn.init.xavier_uniform_(self.weight)
        self.bias_fa = None
        if self.bias is not None:
            nn.init.constant_(self.bias, 1)

    def forward(self, x):
        # Sign Weight Transport Backward
        return Conv2dGrad.apply(x,
                                self.weight,
                                self.bias,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)
