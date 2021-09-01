import torchvision.models as models
import biotorch.models.small_resnet as small_resnet

from torchvision.models.resnet import ResNet
from biotorch.models.utils import create_torchvision_biomodel


MODE = 'backpropagation'
MODE_STRING = 'Backpropagation'


def resnet18(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-18 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnet18, MODE, layer_config, pretrained, progress, num_classes)


def resnet20(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-20 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet20, MODE, layer_config, pretrained, progress, num_classes)


def resnet32(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-32 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-32 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet32, MODE, layer_config, pretrained, progress, num_classes)


def resnet34(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-34 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnet34, MODE, layer_config, pretrained, progress, num_classes)


def resnet44(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-44 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-44 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet44, MODE, layer_config, pretrained, progress, num_classes)


def resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-50 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnet50, MODE, layer_config, pretrained, progress, num_classes)


def resnet56(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-56 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet56, MODE, layer_config, pretrained, progress, num_classes)


def resnet101(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-101 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnet101, MODE, layer_config, pretrained, progress, num_classes)


def resnet110(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-110 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-110 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet110, MODE, layer_config, pretrained, progress, num_classes)


def resnet152(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-152 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnet152, MODE, layer_config, pretrained, progress, num_classes)


def resnet1202(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config: dict = None) -> ResNet:
    r"""ResNet-1202 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNet-1202 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(small_resnet.resnet1202, MODE, layer_config, pretrained, progress, num_classes)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """

    print('Converting ResNext-50 32x4d to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnext50_32x4d, MODE, layer_config, pretrained, progress, num_classes)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting ResNext-101 32x8d to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.resnext101_32x8d, MODE, layer_config, pretrained, progress, num_classes)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Wide ResNet-50-2 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.wide_resnet50_2, MODE, layer_config, pretrained, progress, num_classes)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting Wide ResNet-101-2 to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.wide_resnet101_2, MODE, layer_config, pretrained, progress, num_classes)
