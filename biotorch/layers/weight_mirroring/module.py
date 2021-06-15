import torch.nn as nn
from torch import Tensor


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def mirror_weights(self,
                       x: Tensor,
                       mirror_learning_rate: float = 0.01,
                       noise_amplitude: float = 0.1,
                       damping_factor: float = 0.5) -> Tensor:
        for module in self:
            x = module.mirror_weights(x, mirror_learning_rate, noise_amplitude, damping_factor)
        return x

