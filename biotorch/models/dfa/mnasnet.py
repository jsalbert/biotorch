import torchvision.models as models
from torchvision.models.mnasnet import MNASNet
from biotorch.models.utils import create_torchvision_biomodel


MODE = 'dfa'
MODE_STRING = 'Direct Feedback Alignment'


def mnasnet0_5(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting MNASNet 0.5 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.mnasnet0_5, MODE, layer_config, pretrained, progress, num_classes)


def mnasnet0_75(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting MNASNet 0.75 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.mnasnet0_75, MODE, layer_config, pretrained, progress, num_classes)


def mnasnet1_0(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting MNASNet 1.0 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.mnasnet1_0, MODE, layer_config, pretrained, progress, num_classes)


def mnasnet1_3(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting MNASNet 1.3 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.mnasnet1_3, MODE, layer_config, pretrained, progress, num_classes)
