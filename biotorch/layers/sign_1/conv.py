import math
import torch
import biotorch.layers.fa as layers_fa


from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.fa.conv import Conv2dGrad


class Conv2d(layers_fa.Conv2d):
    """
    Implements the method from How Important Is Weight Symmetry in Backpropagation?

    Uniform Sign-concordant Feedbacks (uSF):
    Backward Weights = sign(W)

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
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
        self.register_backward_hook(self.gradient_clip)

    def forward(self, x):
        # To avoid Exploding Gradients, we scale the sign of the weights by a scaling factor
        # given by our layer initialization as in (https://arxiv.org/pdf/1811.03567.pdf)
        self.weight_backward = torch.nn.Parameter(self.scaling_factor * torch.sign(self.weight))

        return Conv2dGrad.apply(x,
                                self.weight,
                                self.weight_backward,
                                self.bias,
                                None,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)

    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
        return tuple(grad_input)
