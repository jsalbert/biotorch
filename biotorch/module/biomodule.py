import torch.nn as nn
from torch.autograd import grad


from biotorch.module.converter import ModuleConverter


class BioModel(nn.Module):
    def __init__(self, model, mode='FA', copy_weights=False, output_dim=None):
        super(BioModel, self).__init__()
        module_converter = ModuleConverter(mode)
        self.mode = mode
        self.output_dim = output_dim
        self.copy_weights = copy_weights

        if self.mode == 'DFA':
            if self.output_dim is None:
                raise ValueError('You need to introduce the `output_dim` of your model for DFA mode')

        self.model = module_converter.convert(model, copy_weights, output_dim)

    def forward(self, x, targets=None, loss_function=None):
        output = self.model(x)
        if self.mode == 'DFA' and self.model.training:
            if targets is None:
                raise ValueError('Targets missing for DFA mode')
            if loss_function is None:
                raise ValueError('You need to introduce your `loss_function` for DFA mode')
            loss = loss_function(output, targets)
            loss_gradient = grad(loss, output, retain_graph=True)[0]
            # Broadcast gradient of the loss to every layer
            for layer in self.model.modules():
                layer.loss_gradient = loss_gradient
        return output
