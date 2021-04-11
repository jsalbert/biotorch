import torch
import torch.nn as nn


from torch.autograd import Variable
from biotorch.autograd.functions import LinearFA, Conv2dFA


def add_fa_weight_matrices(layer):
    # fixed random weight and bias for FA backward pass do not need gradient
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
        layer.weight_fa = Variable(torch.Tensor(layer.weight.size()), requires_grad=False)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(layer.weight_fa)
        layer.bias_fa = None
        if layer.bias is not None:
            layer.bias_fa = Variable(torch.Tensor(layer.bias.size()), requires_grad=False)
            torch.nn.init.constant_(layer.bias, 1)
            torch.nn.init.constant_(layer.bias_fa, 1)


def override_backward(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        def forward_conv(x):
            if layer.bias is None:
                return Conv2dFA.apply(x,
                                      layer.weight,
                                      layer.weight_fa,
                                      None,
                                      None,
                                      layer.stride,
                                      layer.padding,
                                      layer.dilation,
                                      layer.groups)
            else:
                return Conv2dFA.apply(x,
                                      layer.weight,
                                      layer.weight_fa,
                                      layer.bias,
                                      layer.bias_fa,
                                      layer.stride,
                                      layer.padding,
                                      layer.dilation,
                                      layer.groups)
        layer.forward = forward_conv
    elif isinstance(layer, nn.Linear):
        def forward_fc(x):
            if layer.bias is None:
                return LinearFA.apply(x, layer.weight, layer.weight_fa, None)
            else:
                return LinearFA.apply(x, layer.weight, layer.weight_fa, layer.bias, layer.bias_fa)
        layer.forward = forward_fc
