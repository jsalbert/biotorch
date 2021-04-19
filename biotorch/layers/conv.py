import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.functions import Conv2dFA

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union


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

        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_fa)
        self.bias_fa = None
        if self.bias is not None:
            self.bias_fa = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_fa, 1)

    def forward(self, x):
        if self.bias is None:
            return Conv2dFA.apply(x,
                                  self.weight,
                                  self.weight_fa,
                                  None,
                                  None,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.groups)
        else:
            return Conv2dFA.apply(x,
                                  self.weight,
                                  self.weight_fa,
                                  self.bias,
                                  self.bias_fa,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.groups)
