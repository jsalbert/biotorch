import pytest


@pytest.fixture(scope='session')
def mode_types():
    return ['fa', 'dfa', 'usf', 'brsf', 'frsf']


@pytest.fixture(scope='session')
def model_architectures():
    return [
        ('le_net_mnist', (1, 32, 32)),
        ('le_net_cifar', (3, 32, 32)),
        ('resnet18', (3, 128, 128)),
        ('resnet20', (3, 128, 128)),
        ('resnet56', (3, 128, 128))
    ]
