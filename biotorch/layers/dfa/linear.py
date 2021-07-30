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
                "scaling_factor": False,
                "gradient_clip": None
            }

        self.options = self.layer_config["options"]

        self.weight_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dfa)
        self.bias_dfa = None
        self.loss_gradient = None
        if self.bias is not None:
            self.bias_dfa = nn.Parameter(torch.Tensor(size=(output_dim, self.in_features)), requires_grad=False)
            nn.init.constant_(self.bias, 1)
            nn.init.constant_(self.bias_dfa, 1)

        if self.options["constrain_weights"]:
            with torch.no_grad():
                self.norm_initial_weights = torch.linalg.norm(self.weight)

        if self.options["scaling_factor"]:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))

        # Will use gradients computed in the backward hook
        self.register_backward_hook(self.dfa_backward_hook)

    def forward(self, x):
        # Regular BackPropagation Forward-Backward
        with torch.no_grad():
            if self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights / torch.linalg.norm(self.weight))

            if self.options["scaling_factor"]:
                self.weight_dfa = torch.nn.Parameter(self.scaling_factor * self.weight_dfa, requires_grad=False)

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
