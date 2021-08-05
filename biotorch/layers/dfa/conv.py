import math
import torch
import torch.nn as nn


from biotorch.autograd.dfa.conv import Conv2dGrad
from torch.nn.common_types import _size_2_t
from typing import Union


class Conv2d(nn.Conv2d):
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
            padding_mode: str = 'zeros',
            layer_config: dict = None
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

        self.layer_config = layer_config

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "scaling_factor": False,
                "gradient_clip": False
            }

        self.options = self.layer_config["options"]
        self.loss_gradient = None
        self.weight_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels, self.kernel_size[0], self.kernel_size[1])),
                                       requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)

        self.bias_dfa = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels)), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)

        if self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)

        if self.options["scaling_factor"]:
            raise ValueError('scaling_factor not supported for DFA')

        # Will use gradients computed in the backward hook
        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        # Regular BackPropagation Forward-Backward
        with torch.no_grad():
            if self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights / torch.linalg.norm(self.weight))

        return Conv2dGrad.apply(x,
                                self.weight,
                                self.bias,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)

    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        # If layer don't have grad w.r.t input
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
