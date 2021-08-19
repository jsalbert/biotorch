import math
import torch
import torch.nn as nn

from biotorch.autograd.dfa.linear import LinearGrad


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, output_dim: int, bias: bool = True,
                 layer_config: dict = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

        self.layer_config = layer_config

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "gradient_clip": False,
                "init": "xavier"
            }

        self.options = self.layer_config["options"]
        self.init = self.options["init"]
        self.loss_gradient = None

        self.weight_backward = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
        self.bias_backward = None
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)

        self.init_parameters()

        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            with torch.no_grad():
                self.norm_initial_weights = torch.linalg.norm(self.weight)

        # Will use gradients computed in the backward hook
        self.register_backward_hook(self.dfa_backward_hook)
        self.weight_ratio = 0

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_backward)
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_backward, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self, x):
        # Regular BackPropagation Forward-Backward
        with torch.no_grad():
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights / torch.linalg.norm(self.weight))

        return LinearGrad.apply(x, self.weight, self.bias)

    def compute_weight_ratio(self):
        with torch.no_grad():
            self.weight_diff = torch.linalg.norm(self.weight_backward) / torch.linalg.norm(self.weight)
        return self.weight_diff

    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        # If layer don't have grad w.r.t input
        if grad_input[0] is None:
            return grad_input
        else:
            grad_dfa = module.loss_gradient.mm(module.weight_backward)
            # If no bias term
            if len(grad_input) == 2:
                return grad_dfa, grad_input[1]
            else:
                return grad_dfa, grad_input[1], grad_input[2]
