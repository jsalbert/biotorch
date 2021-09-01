import torchvision.models as models
from torchvision.models.alexnet import AlexNet
from biotorch.models.utils import create_torchvision_biomodel


MODE = 'fa'
MODE_STRING = 'Feedback Alignment'


def alexnet(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting AlexNet to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(models.alexnet, MODE, layer_config, pretrained, progress, num_classes)
