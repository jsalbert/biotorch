import torch
import biotorch.layers.fa as layers_fa

from biotorch.autograd.fa.linear import LinearGrad


class Linear(layers_fa.Linear):
    """
    Method from [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/pdf/1510.05067.pdf)

    Fixed Random Magnitude Sign-concordant Feedbacks (frSF):
    weight_backward = M ◦ sign(weight), where M is initialized once and fixed throughout each experiment

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        return LinearGrad.apply(x,
                                self.weight,
                                self.weight_backward * torch.sign(self.weight),
                                self.bias,
                                None)
