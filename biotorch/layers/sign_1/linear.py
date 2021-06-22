import math
import torch
import biotorch.layers.fa as layers_fa

from biotorch.autograd.fa.linear import LinearGrad


class Linear(layers_fa.Linear):
    """
    Method from [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/pdf/1510.05067.pdf)

    Uniform Sign-concordant Feedbacks (uSF):
    weight_backward = sign(weight)

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))

    def forward(self, x):
        # Linear Sign Weight Transport Backward
        self.weight_backward = torch.nn.Parameter(self.scaling_factor * torch.sign(self.weight))
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, None)
