import torch.nn as nn
from torch.autograd import grad


from biotorch.converter.functions import convert_model


class BioModel(nn.Module):
    def __init__(self, model, loss_function, copy_weights=False, mode='FA', output_dim=None):
        super(BioModule, self).__init__()
        self.model = convert_model(model, mode, copy_weights)
        self.loss_function = loss_function
        self.output_grad = None
        self.copy_weights = copy_weights
        self.mode = mode
        self.output_dim = output_dim

    def forward(self, x, targets=None):
        output = self.model.forward(x)
        if self.mode == 'DFA':
            if targets is None:
                raise ValueError('Targets missing for DFA mode')
            loss = loss_function(output, targets)
            loss_gradient = grad(loss, output)
            self.loss_gradient_broadcast(loss_gradient)
        return output

    def loss_gradient_broadcast(self, loss_gradient):
        for layer in self.model.modules():
            layer.loss_gradient = loss_gradient
