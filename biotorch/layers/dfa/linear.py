import torch
import torch.nn as nn

from biotorch.autograd.dfa.linear import LinearGrad


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, output_dim: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)
        # Will use gradients computed in the backward hook
        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        # Regular BackPropagation Forward-Backward
        return LinearGrad.apply(x, self.weight, self.bias)

    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        # If layer don't have grad w.r.t input
        if grad_input[0] is None:
            return grad_input
        else:
            grad_dfa = module.loss_gradient.mm(module.weight_dfa)
            # If no bias term
            if len(grad_input) == 2:
                return grad_dfa, grad_input[1]
            else:
                return grad_dfa, grad_input[1], grad_input[2]
