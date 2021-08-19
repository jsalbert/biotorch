import torch
from biotorch.module.biomodule import ModuleConverter


def test_module_converter_convert_dummy_net(dummy_net, mode_types):
    with torch.no_grad():
        converter = ModuleConverter()
        layers_to_convert = {'nn.Conv2d': 1, 'nn.Linear': 1}
        w1 = dummy_net.conv1.weight
        w2 = dummy_net.fc.weight
        for mode in mode_types:
            converted = converter.convert(dummy_net, mode)

            for layer, count in converter.replaced_layers_counts.items():
                assert layers_to_convert[layer] == count