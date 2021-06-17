from biotorch.datasets.cifar import CIFAR10, CIFAR100
from biotorch.datasets.mnist import MNIST
from biotorch.datasets.fashion_mnist import FashionMNIST
from biotorch.datasets.tiny_imagenet import TinyImageNet
from biotorch.datasets.imagenet import ImageNet


DATASETS_AVAILABLE = ['mnist', 'cifar10', 'cifar100', 'fashion-mnist', 'tiny-imagenet', 'imagenet']


class DatasetSelector:
    def __init__(self, dataset_name):
        if dataset_name not in DATASETS_AVAILABLE:
            raise ValueError('Dataset name specified: {} not in the list of available datasets {}'.format(
                dataset_name, DATASETS_AVAILABLE)
            )
        self.dataset_name = dataset_name

    def get_dataset(self):
        if self.dataset_name == 'cifar10':
            return CIFAR10
        elif self.dataset_name == 'cifar100':
            return CIFAR100
        elif self.dataset_name == 'mnist':
            return MNIST
        elif self.dataset_name == 'fashion-mnist':
            return FashionMnist
        elif self.dataset_name == 'tiny-imagenet':
            return TinyImageNet
        elif self.dataset_name == 'imagenet':
            return ImageNet
