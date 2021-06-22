import torch
import torch.nn as nn

from biotorch.autograd.fa.linear import LinearGrad
from biotorch.layers.metrics import compute_matrix_angle


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_backward)
        self.bias_backward = None
        self.alignment = 0
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_backward, 1)

    def forward(self, x):
        # Linear Feedback Alignment Backward
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, self.bias_backward)

    def compute_alignment(self):
        self.alignment = compute_matrix_angle(self.weight_backward, self.weight)
        return self.alignment
