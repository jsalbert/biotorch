import torch
import torch.nn as nn

from biotorch.autograd.fa.linear import LinearGrad


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
        # Linear Feedback Alignment Backward
        return LinearGrad.apply(x, self.weight, self.weight_fa, self.bias, self.bias_fa)

    def update_B(self, weight_update, damping_factor=0.5):
        self.weight_fa += weight_update
        # Prevent feedback weights growing too large
        x = torch.randn(self.weight_fa.size())
        y = torch.matmul(self.weight_fa, x)
        y_std = torch.mean(np.std(y, axis=0))
        self.weight_fa = damping_factor * self.weight_fa / y_std
