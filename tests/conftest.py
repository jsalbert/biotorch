import pytest
import torch
import torch.nn as nn
import torch.functional as F


@pytest.fixture(scope='session')
def mode_types():
    return ['backpropagation', 'fa', 'dfa', 'usf', 'brsf', 'frsf']


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = F.avg_pool2d(out, out.size()[3])
        return self.fc(out)


@pytest.fixture(scope='function')
def dummy_net():
    return Model()


@pytest.fixture(scope='function')
def dummy_net_constructor():
    return Model
