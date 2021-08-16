import torch
import biotorch.layers.fa_constructor as fa_constructor

from biotorch.autograd.fa.linear import LinearGrad


class Linear(fa_constructor.Linear):
    """
    Method from [How Important Is Weight Symmetry in Backpropagation?](https://arxiv.org/pdf/1510.05067.pdf)

    Uniform Sign-concordant Feedbacks (uSF):
    weight_backward = sign(weight)

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        if layer_config is None:
            layer_config = {}
        layer_config["type"] = "usf"

        super(Linear, self).__init__(in_features, out_features, bias, layer_config)
