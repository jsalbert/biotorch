import torchvision.models as models


from torchvision.models.densenet import DenseNet
from biotorch.models.utils import create_torchvision_biomodel


MODE = 'frsf'
MODE_STRING = 'Sign Alignment: Fixed Random Magnitude Sign-concordant Feedbacks (frSF)'


def densenet121(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Densenet-121 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.densenet121, MODE, layer_config, pretrained, progress, num_classes)


def densenet161(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Densenet-161 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.densenet161, MODE, layer_config, pretrained, progress, num_classes)


def densenet169(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Densenet-169 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.densenet169, MODE, layer_config, pretrained, progress, num_classes)


def densenet201(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Densenet-201 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.densenet201, MODE, layer_config, pretrained, progress, num_classes)
