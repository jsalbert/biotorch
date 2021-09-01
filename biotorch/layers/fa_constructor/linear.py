import math
import torch
import torch.nn as nn
from torch import Tensor

from biotorch.autograd.fa.linear import LinearGrad
from biotorch.layers.metrics import compute_matrix_angle


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "fa"
            }

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "gradient_clip": False,
                "init": "xavier"
            }

        self.options = self.layer_config["options"]
        self.type = self.layer_config["type"]
        self.init = self.options["init"]

        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

        if self.type == "frsf":
            self.weight_backward = nn.Parameter(torch.abs(self.weight_backward), requires_grad=False)

        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)

        if self.type == "usf" or self.type == "brsf":
            # Initialize backward weight for alignment computation
            with torch.no_grad():
                self.weight_backward = torch.nn.Parameter(self.scaling_factor * torch.sign(self.weight),
                                                          requires_grad=False)

        self.alignment = 0
        self.weight_ratio = 0

        if "gradient_clip" in self.options and self.options["gradient_clip"]:
            self.register_backward_hook(self.gradient_clip)

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

    def forward(self, x: Tensor) -> Tensor:
        weight_backward = None
        with torch.no_grad():
            # Based on "Feedback alignment in deep convolutional networks" (https://arxiv.org/pdf/1812.06488.pdf)
            # Constrain weight magnitude
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights / torch.linalg.norm(self.weight))

            # Backward using weight_backward matrix
            if self.type == "usf":
                weight_backward = torch.nn.Parameter(torch.sign(self.weight), requires_grad=False)
                # To avoid Exploding Gradients, we scale the sign of the weights by a scaling factor
                # given by our layer initialization as in "Biologically-Plausible Learning Algorithms Can
                # Scale to Large Datasets" (https://arxiv.org/pdf/1811.03567.pdf)
                weight_backward = torch.nn.Parameter(self.scaling_factor * weight_backward, requires_grad=False)
                self.weight_backward = weight_backward
            elif self.type == "brsf":
                wb = torch.Tensor(self.weight.size()).to(self.weight.device)
                if self.init == "xavier":
                    torch.nn.init.xavier_uniform_(wb)
                else:
                    init.kaiming_uniform_(wb, a=math.sqrt(5))
                weight_backward = torch.nn.Parameter(torch.abs(wb) * torch.sign(self.weight), requires_grad=False)
                self.weight_backward = weight_backward
            elif self.type == "frsf":
                weight_backward = torch.nn.Parameter(torch.abs(self.weight_backward) * torch.sign(self.weight),
                                                     requires_grad=False)
            # Vanilla FA
            if weight_backward is None:
                weight_backward = self.weight_backward

        return LinearGrad.apply(x, self.weight, weight_backward, self.bias, self.bias_backward)

    def compute_alignment(self):
        self.alignment = compute_matrix_angle(self.weight_backward, self.weight)
        return self.alignment

    def compute_weight_ratio(self):
        with torch.no_grad():
            self.weight_diff = torch.linalg.norm(self.weight_backward) / torch.linalg.norm(self.weight)
        return self.weight_diff

    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
        return tuple(grad_input)
