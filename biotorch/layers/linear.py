import torch
import numpy as np
import torch.nn as nn

from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.linear import LinearFA


class LinearFA(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearFA, self).__init__(in_features, out_features, bias)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_fa)
        self.bias_fa = None
        if self.bias is not None:
            self.bias_fa = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_fa, 1)

    def forward(self, x):
        return LinearFAGrad.apply(x, self.weight, self.weight_fa, self.bias, self.bias_fa)


class LinearDFA(nn.Linear):
    def __init__(self, in_features: int, out_features: int, y_dim: int, bias: bool = True) -> None:
        super(LinearDFA, self).__init__(in_features, out_features, bias)
        self.weight_dfa = nn.Parameter(torch.Tensor(y_dim, self.weight.size()[1]), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_fa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(y_dim, self.bias.size()[1]), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)
        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        return LinearFAGrad.apply(x, self.weight, self.weight_dfa, self.bias, self.bias_dfa)

    def dfa_backward_hook(self, grad_input, grad_output):
        return self.loss_gradient.mm(self.weight_dfa)
