import torch
import biotorch.layers.fa as layers_fa

from biotorch.autograd.fa.linear import LinearGrad


class Linear(layers_fa.Linear):
    """
    Method from [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/pdf/1510.05067.pdf)

    Batchwise Random Magnitude Sign-concordant Feedbacks (brSF):
    weight_backward = M ◦ sign(weight), where M is redrawn after each update of W (i.e., each mini-batch).

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        wb = torch.Tensor(self.weight.size()).to(self.weight.device)
        torch.nn.init.xavier_uniform_(wb)
        self.weight_backward = torch.nn.Parameter(wb * torch.sign(self.weight), requires_grad=False)
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, None)
