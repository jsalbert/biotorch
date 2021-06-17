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
        self.bias_fa = None
        if self.bias is not None:
            self.bias_fa = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_fa, 1)

        self.alignment = 0

    def forward(self, x):
        # Linear Feedback Alignment Backward
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, self.bias_fa)

    def update_B(self,
                 x,
                 y,
                 mirror_learning_rate=0.01,
                 growth_control=False,
                 damping_factor=0.5):

        dB = mirror_learning_rate * torch.matmul(y.T, x)
        self.weight_backward += nn.Parameter(dB)

        if growth_control:
            device = x.device
            # Prevent feedback weights growing too large
            x = torch.randn(x.size()).to(device)
            y = torch.matmul(x, self.weight_backward.T)
            y_std = torch.mean(torch.std(y, axis=0))
            self.weight_backward = nn.Parameter(damping_factor * self.weight_backward / y_std, requires_grad=False)

    def compute_alignment(self):
        self.alignment = compute_matrix_angle(self.weight_backward, self.weight)
        return self.alignment
