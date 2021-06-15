import torch
import torch.nn as nn

from biotorch.autograd.fa.linear import LinearGrad


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

    def forward(self, x):
        # Linear Feedback Alignment Backward
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, self.bias_fa)

    def update_B(self, weight_update, damping_factor=0.5):
        self.weight_backward += weight_update
        print('weight_backward: ', self.weight_backward.size())
        # Prevent feedback weights growing too large
        x = torch.randn(self.weight_backward.size()[1])
        print('x: ', x.size())
        y = torch.matmul(self.weight_backward, x)
        print('y: ', y.size())
        y_std = torch.mean(torch.std(y, axis=0))
        print('y_std: ', y_std.size())
        self.weight_backward = nn.Parameter(damping_factor * self.weight_backward / y_std, requires_grad=False)
