import torch.nn as nn

from biotorch.autograd.sign.linear import LinearGrad


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 1)

    def forward(self, x):
        # Linear Sign Weight Transport Backward
        return LinearGrad.apply(x, self.weight, self.bias)
