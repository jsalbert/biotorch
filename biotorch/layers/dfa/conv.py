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
                "init": "xavier",
                "gradient_clip": False
            }

        self.options = self.layer_config["options"]
        self.init = self.options["init"]
        self.loss_gradient = None
        self.weight_backward = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels,
                                                               self.kernel_size[0],
                                                               self.kernel_size[1])),
                                            requires_grad=False)

        self.bias_backward = None
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(size=(output_dim, self.in_channels)), requires_grad=False)

        self.init_parameters()

        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)

        # Will use gradients computed in the backward hook
        self.register_backward_hook(self.dfa_backward_hook)
        self.weight_ratio = 0

    def init_parameters(self) -> None:
        # Xavier initialization
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_backward)
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_backward, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)

    def compute_weight_ratio(self):
        with torch.no_grad():
            self.weight_diff = torch.linalg.norm(self.weight_backward) / torch.linalg.norm(self.weight)
        return self.weight_diff

    def forward(self, x):
        # Regular BackPropagation Forward-Backward
        with torch.no_grad():
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
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
                                                  weight=module.weight_backward,
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
