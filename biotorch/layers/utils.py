import torch.nn as nn

import biotorch.layers.fa as fa_layers
import biotorch.layers.dfa as dfa_layers
import biotorch.layers.sign as sign_layers


def convert_layer(layer, mode, copy_weights, output_dim=None):
    # Initialize variables
    layer_bias, bias_weight = False, None
    # Save original weights and biases
    if 'weight' in layer.__dict__['_parameters'] and copy_weights:
        weight = layer.weight

    if 'bias' in layer.__dict__['_parameters'] and layer.bias is not None:
        bias_weight = layer.bias
        layer_bias = True

    new_layer = None
    if isinstance(layer, nn.Conv2d):
        if mode == 'FA':
            new_layer = fa_layers.Conv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer_bias,
                layer.padding_mode
            )
        elif mode == 'DFA':
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
                layer.padding_mode
            )
        elif mode == 'sign':
            new_layer = sign_layers.Conv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer_bias,
                layer.padding_mode
            )

    elif isinstance(layer, nn.Linear):
        if mode == 'FA':
            new_layer = fa_layers.Linear(
                layer.in_features,
                layer.out_features,
                layer_bias,
            )
        elif mode == 'DFA':
            new_layer = dfa_layers.Linear(
                layer.in_features,
                layer.out_features,
                output_dim,
                layer_bias,
            )
        elif mode == 'sign':
            new_layer = sign_layers.Linear(
                layer.in_features,
                layer.out_features,
                layer_bias,
            )

    if new_layer is not None and copy_weights:
        new_layer.weight = weight
        new_layer.bias = bias_weight

    return new_layer
