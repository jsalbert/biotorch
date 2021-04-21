import torch
import numpy as np
import torch.nn as nn

from torch import autograd
from torch.autograd import Variable
from biotorch.autograd.linear import LinearFAGrad, LinearBPDFA


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
    def __init__(self, in_features: int, out_features: int, output_dim: int, bias: bool = True) -> None:
        super(LinearDFA, self).__init__(in_features, out_features, bias)
        self.weight_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)
        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        return LinearBPDFA.apply(x, self.weight, self.bias)

    def dfa_backward_hook(self, module, grad_input, grad_output):
        # If initial layer don't have grad w.r.t input
        if grad_input[0] is None:
            return grad_input
        else:
            grad_dfa = module.loss_gradient.mm(module.weight_dfa)
            # If no bias term
            if len(grad_input) == 2:
                return grad_dfa, grad_input[1]
            else:
                return grad_dfa, grad_input[1], grad_input[2]
