import torch
import numpy as np
import torch.nn as nn

from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.functions import LinearFA


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_fa)
        self.bias_fa = None
        if self.bias is not None:
            self.bias_fa = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_fa, 1)

    def forward(self, x):
        if self.bias is None:
            return LinearFA.apply(x, self.weight, self.weight_fa, None)
        else:
            return LinearFA.apply(x, self.weight, self.weight_fa, self.bias, self.bias_fa)
