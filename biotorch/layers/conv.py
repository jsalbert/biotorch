import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.conv import Conv2dFAGrad

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union


class Conv2dFA(nn.Conv2d):
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

        super(Conv2dFA, self).__init__(
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
        return Conv2dFAGrad.apply(x,
                              self.weight,
                              self.weight_fa,
                              self.bias,
                              self.bias_fa,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)


class Conv2dDFA(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            y_dim: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'
    ):

        super(Conv2dDFA, self).__init__(
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

        self.weight_dfa = nn.Parameter(torch.Tensor(y_dim, self.weight.size()[1:3]), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(y_dim, self.bias.size()[1]), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)

    def forward(self, x):
        return Conv2dFAGrad.apply(x,
                                  self.weight,
                                  self.weight_dfa,
                                  self.bias,
                                  self.bias_dfa,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.groups)
