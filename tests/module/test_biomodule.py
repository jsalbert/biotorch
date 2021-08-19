import pytest
from biotorch.module.biomodule import BioModule


def test_biomodule_convert(dummy_net, mode_types):
    for mode in mode_types:
        if mode == 'dfa':
            with pytest.raises(ValueError,
                               match='Model `output_dim` is required for Direct Feedback Alignment (dfa) mode'):
                BioModule(dummy_net, mode)
            biomodule = BioModule(dummy_net, mode, output_dim=10)
        else:
            biomodule = BioModule(dummy_net, mode)
