import pytest
from biotorch.module.biomodule import BioModule


def test_biomodule_convert(dummy_net_constructor, mode_types):
    for mode in mode_types:
        dummy_net = dummy_net_constructor()
        if mode == 'dfa':
            with pytest.raises(ValueError,
                               match="Model `output_dim` is required for Direct Feedback Alignment \\(dfa\\) mode"):
                BioModule(dummy_net, mode)
            _ = BioModule(dummy_net, mode, output_dim=10)
        else:
            _ = BioModule(dummy_net, mode)
