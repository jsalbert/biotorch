import pytest


@pytest.fixture(scope='session')
def model_architectures():
    # Tuple containing (architecture, input size)
    return [
        ('le_net_mnist', (1, 1, 32, 32)),
        ('le_net_cifar', (1, 3, 32, 32)),
        ('resnet18', (1, 3, 128, 128)),
        ('resnet20', (1, 3, 128, 128)),
        ('resnet56', (1, 3, 128, 128))
    ]
