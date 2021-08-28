import torch
import torch.nn as nn
import biotorch.layers.fa_constructor as fa_constructor


from biotorch.autograd.fa.linear import LinearGrad
from biotorch.layers.metrics import compute_matrix_angle


class Linear(fa_constructor.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        if layer_config is None:
            layer_config = {}
        layer_config["type"] = "fa"

        super(Linear, self).__init__(in_features, out_features, bias, layer_config)
