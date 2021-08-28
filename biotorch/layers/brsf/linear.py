import torch
import biotorch.layers.fa_constructor as fa_constructor

from biotorch.autograd.fa.linear import LinearGrad


class Linear(fa_constructor.Linear):
    """
    Implements the method from How Important Is Weight Symmetry in Backpropagation?
    with the modification of taking the absolute value of the Backward Matrix

    Batchwise Random Magnitude Sign-concordant Feedbacks (brSF):
    weight_backward = |M| ◦ sign(weight), where M is redrawn after each update of W (i.e., each mini-batch).

    (https://arxiv.org/pdf/1510.05067.pdf)

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        if layer_config is None:
            layer_config = {}
        layer_config["type"] = "brsf"

        super(Linear, self).__init__(in_features, out_features, bias, layer_config)
