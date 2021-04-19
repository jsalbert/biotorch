import torch
import torch.nn as nn


from collections import defaultdict
from biotorch.layers import Linear, Conv2d


def add_fa_weights(layer):
    # fixed random weight and bias for FA backward pass do not need gradient
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
        layer.weight_fa = nn.Parameter(torch.Tensor(layer.weight.size()), requires_grad=False)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(layer.weight_fa)
        layer.bias_fa = None
        if layer.bias is not None:
            layer.bias_fa = nn.Parameter(torch.Tensor(layer.bias.size()), requires_grad=False)
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


def replace_layers_recursive(module, copy_weights, replaced_layers):
    # Go through all of module nn.module (e.g. network or layer)
    for module_name in module._modules.keys():
        # Get layer
        layer = getattr(module, module_name)
        # Initialize variables
        layer_bias, bias_weight = False, None
        # Save original weights and biases
        if 'weight' in layer.__dict__['_parameters'] and copy_weights:
            weight = layer.weight

        if 'bias' in layer.__dict__['_parameters'] and layer.bias is not None:
            bias_weight = layer.bias
            layer_bias = True

        if isinstance(layer, nn.Conv2d):
            new_layer = Conv2d(
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
            if copy_weights:
                new_layer.weight = weight
                new_layer.bias = bias_weight
            replaced_layers[str(type(layer))] += 1
            setattr(module, module_name, new_layer)

        elif isinstance(layer, nn.Linear):
            new_layer = Linear(
                layer.in_features,
                layer.out_features,
                layer_bias,
            )
            if copy_weights:
                new_layer.weight = weight
                new_layer.bias = bias_weight
            replaced_layers[str(type(layer))] += 1
            setattr(module, module_name, new_layer)

    # Iterate through immediate child modules
    for name, child_module in module.named_children():
        replace_layers_recursive(child_module, copy_weights, replaced_layers)


def convert_module(module, copy_weights=False):
    # Compute original model layer counts
    layer_counts = count_layers(module)

    # Replace layers
    replaced_layers_counts = defaultdict(lambda: 0)
    replace_layers_recursive(module, copy_weights, replaced_layers_counts)

    # Sanity Check
    for layer, count in replaced_layers_counts.items():
        if layer_counts[layer] != count:
            print('There were originally {} {} layers and {} were converted'.format(layer_counts[layer], layer, count))
        else:
            print('All the {} {} layers were converted successfully'.format(count, layer))


def count_layers(module):
    layer_counts = defaultdict(lambda: 0)
    for layer in module.modules():
        layer_counts[str(type(layer))] += 1

    return layer_counts
