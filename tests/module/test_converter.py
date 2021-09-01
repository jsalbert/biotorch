import numpy as np
from biotorch.module.biomodule import ModuleConverter


def test_module_converter_convert_dummy_net(dummy_net_constructor, mode_types):
    for mode in mode_types:
        dummy_net = dummy_net_constructor()
        layers_to_convert = {str(type(dummy_net.conv1)): 1, str(type(dummy_net.fc)): 1}
        w1 = dummy_net.conv1.weight.data
        w2 = dummy_net.fc.weight.data
        output_dim = None
        converter = ModuleConverter(mode=mode)
        if mode == 'dfa':
            output_dim = 10
        converted = converter.convert(dummy_net, output_dim=output_dim)
        for layer, count in converter.replaced_layers_counts.items():
            assert layers_to_convert[layer] == count

    assert not np.testing.assert_array_almost_equal(w1, converted.conv1.weight.data)
    assert not np.testing.assert_array_almost_equal(w2, converted.fc.weight.data)


def test_module_converter_convert_dummy_net_copy_weights(dummy_net_constructor, mode_types):
    for mode in mode_types:
        dummy_net = dummy_net_constructor()
        layers_to_convert = {str(type(dummy_net.conv1)): 1, str(type(dummy_net.fc)): 1}
        w1 = dummy_net.conv1.weight.data
        w2 = dummy_net.fc.weight.data
        output_dim = None
        converter = ModuleConverter(mode=mode)
        if mode == 'dfa':
            output_dim = 10
        converted = converter.convert(dummy_net, copy_weights=True, output_dim=output_dim)
        for layer, count in converter.replaced_layers_counts.items():
            assert layers_to_convert[layer] == count

        np.testing.assert_array_almost_equal(w1, converted.conv1.weight.data)
        np.testing.assert_array_almost_equal(w2, converted.fc.weight.data)


def test_module_converter_convert_dummy_net_layer_config(dummy_net_constructor, mode_types):
    for mode in mode_types:
        dummy_net = dummy_net_constructor()
        layers_to_convert = {str(type(dummy_net.conv1)): 1, str(type(dummy_net.fc)): 1}
        w1 = dummy_net.conv1.weight.data
        w2 = dummy_net.fc.weight.data
        output_dim = None
        converter = ModuleConverter(mode=mode)
        layer_config = {'options': {'init': 'kaiming'}}
        if mode == 'dfa':
            output_dim = 10
        converted = converter.convert(dummy_net, copy_weights=True, output_dim=output_dim, layer_config=layer_config)
        for layer, count in converter.replaced_layers_counts.items():
            assert layers_to_convert[layer] == count

        np.testing.assert_array_almost_equal(w1, converted.conv1.weight.data)
        np.testing.assert_array_almost_equal(w2, converted.fc.weight.data)
        assert converted.conv1.init == 'kaiming'
        assert converted.fc.init == 'kaiming'
