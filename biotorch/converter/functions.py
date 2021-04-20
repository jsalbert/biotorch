import torch
import torch.nn as nn


from collections import defaultdict
from biotorch.layers.utils import convert_layer


def convert_module(module, mode='FA', copy_weights=False):
    # Compute original model layer counts
    layer_counts = count_layers(module)
    # Replace layers
    replaced_layers_counts = defaultdict(lambda: 0)
    replace_layers_recursive(module, mode, copy_weights, replaced_layers_counts)
    # Sanity Check
    for layer, count in replaced_layers_counts.items():
        if layer_counts[layer] != count:
            print('There were originally {} {} layers and {} were converted'.format(layer_counts[layer], layer, count))
        else:
            print('All the {} {} layers were converted successfully'.format(count, layer))

    return module


def replace_layers_recursive(module, mode, copy_weights, replaced_layers):
    # Go through all of module nn.module (e.g. network or layer)
    for module_name in module._modules.keys():
        # Get layer
        layer = getattr(module, module_name)
        # Convert layer
        new_layer = convert_layer(layer, mode, copy_weights)
        if new_layer is not None:
            replaced_layers[str(type(layer))] += 1
            setattr(module, module_name, new_layer)
    # Iterate through immediate child modules
    for name, child_module in module.named_children():
        replace_layers_recursive(child_module, mode, copy_weights, replaced_layers)


def count_layers(module):
    layer_counts = defaultdict(lambda: 0)
    for layer in module.modules():
        layer_counts[str(type(layer))] += 1

    return layer_counts
