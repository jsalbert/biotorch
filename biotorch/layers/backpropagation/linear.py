import math
import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "backpropagation"
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
        self.init_parameters()

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
