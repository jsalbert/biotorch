import math
import torch
import biotorch.layers.fa as layers_fa


from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.fa.conv import Conv2dGrad


class Conv2d(layers_fa.Conv2d):
    """
    Implements the method from How Important Is Weight Symmetry in Backpropagation?

    Batchwise Random Magnitude Sign-concordant Feedbacks (brSF):
    weight_backward = M ◦ sign(weight), where M is redrawn after each update of W (i.e., each mini-batch).

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
        wb = torch.Tensor(self.weight.size()).to(self.weight.device)
        torch.nn.init.xavier_uniform_(wb)
        self.weight_backward = torch.nn.Parameter(wb * torch.sign(self.weight), requires_grad=False)

        return Conv2dGrad.apply(x,
                                self.weight,
                                self.weight_backward,
                                self.bias,
                                None,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)
