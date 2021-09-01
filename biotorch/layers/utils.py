import torch.nn as nn

import biotorch.layers.fa_constructor as fa_constructor
import biotorch.layers.backpropagation as bp_layers
import biotorch.layers.dfa as dfa_layers


def convert_layer(layer, mode, copy_weights, layer_config=None, output_dim=None):
    # Initialize variables
    layer_bias, bias_weight = False, None
    # Save original weights and biases
    if 'weight' in layer.__dict__['_parameters'] and copy_weights:
        weight = layer.weight

    if 'bias' in layer.__dict__['_parameters'] and layer.bias is not None:
        bias_weight = layer.bias
        layer_bias = True

    new_layer = None
    if layer_config is None:
        layer_config = {}

    layer_config["type"] = mode
    if isinstance(layer, nn.Conv2d):
        if mode in ["fa", "usf", "brsf", "frsf"]:
            new_layer = fa_constructor.Conv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer_bias,
                layer.padding_mode,
                layer_config
            )
        elif mode == 'dfa':
            new_layer = dfa_layers.Conv2d(
                layer.in_channels,
                layer.out_channels,
                output_dim,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer_bias,
                layer.padding_mode,
                layer_config
            )
        elif mode == 'backpropagation':
            new_layer = bp_layers.Conv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer_bias,
                layer.padding_mode,
                layer_config
            )

    elif isinstance(layer, nn.Linear):
        if mode in ["fa", "usf", "brsf", "frsf"]:
            new_layer = fa_constructor.Linear(
                layer.in_features,
                layer.out_features,
                layer_bias,
                layer_config
            )
        elif mode == 'dfa':
            new_layer = dfa_layers.Linear(
                layer.in_features,
                layer.out_features,
                output_dim,
                layer_bias,
                layer_config
            )
        elif mode == 'backpropagation':
            new_layer = bp_layers.Linear(
                layer.in_features,
                layer.out_features,
                layer_bias,
                layer_config
            )

    if new_layer is not None and copy_weights:
        new_layer.weight = weight
        new_layer.bias = bias_weight

    return new_layer
