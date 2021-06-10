import torch
import torch.nn as nn

from biotorch.layers.metrics import compute_angle
from biotorch.autograd.fa.linear import LinearGrad


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_backward)
        self.bias_backward = None
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_backward, 1)
        # self.register_backward_hook(self.angle_hook)

    def forward(self, x):
        # Linear Feedback Alignment Backward
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, self.bias_backward)

    # @staticmethod
    # def angle_hook(module, grad_input, grad_output):
    #     module.weight_angle = compute_angle(module.weight, module.weight_backward)
    #     print(module, module.weight_angle)
    #     return grad_output
