import biotorch.models.le_net as le_net
from biotorch.models.utils import create_le_net_biomodel


MODE = 'fa'
MODE_STRING = 'Feedback Alignment'


def le_net_mnist(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config=None):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
    """
    print('Converting LeNet CNN MNIST to {} mode'.format(MODE_STRING))
    return create_le_net_biomodel(le_net.le_net_mnist, MODE, layer_config, pretrained, progress, num_classes)


def le_net_cifar(pretrained: bool = False, progress: bool = True, num_classes: int = 10, layer_config=None):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
    """
    print('Converting LeNet CNN CIFAR to {} mode'.format(MODE_STRING))
    return create_le_net_biomodel(le_net.le_net_cifar, MODE, layer_config, pretrained, progress, num_classes)
