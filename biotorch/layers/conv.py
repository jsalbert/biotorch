import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.conv import Conv2dFAGrad, Conv2dBPDFA

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
            output_dim: int,
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

        self.weight_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels, self.kernel_size[0], self.kernel_size[1])),
                                       requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels)), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)

        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        return Conv2dBPDFA.apply(x,
                                 self.weight,
                                 self.bias,
                                 self.stride,
                                 self.padding,
                                 self.dilation,
                                 self.groups)

    def dfa_backward_hook(self, module, grad_input, grad_output):
        # If initial layer don't have grad w.r.t input
        if grad_input[0] is None:
            return grad_input
        else:
            out_grad = module.loss_gradient.unsqueeze(2).repeat(1, 1, grad_output[0].size()[2])
            out_grad = out_grad.unsqueeze(3).repeat(1, 1, 1, grad_output[0].size()[3])
            grad_dfa = torch.nn.grad.conv2d_input(input_size=grad_input[0].shape,
                                                  weight=module.weight_dfa,
                                                  grad_output=out_grad,
                                                  stride=module.stride,
                                                  padding=module.padding,
                                                  dilation=module.dilation,
                                                  groups=module.groups)
            # If no bias term
            if len(grad_input) == 2:
                return grad_dfa, grad_input[1]
            else:
                return grad_dfa, grad_input[1], grad_input[2]
