import pytest


@pytest.fixture(scope='session')
def datasets_available():
    return ['mnist', 'cifar10', 'cifar10_benchmark', 'cifar100', 'fashion_mnist', 'imagenet']
