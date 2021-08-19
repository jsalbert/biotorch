import torch.nn as nn
from torch.autograd import grad


from biotorch.module.converter import ModuleConverter


class BioModule(nn.Module):
    def __init__(self, module, mode='fa', copy_weights=True, layer_config=None, output_dim=None):
        super(BioModule, self).__init__()
        self.module = module
        self.mode = mode
        self.output_dim = output_dim
        self.copy_weights = copy_weights
        if layer_config is None:
            layer_config = {"type": mode}
        self.layer_config = layer_config
        if self.mode == 'dfa':
            if self.output_dim is None:
                raise ValueError("Model `output_dim` is required for Direct Feedback Alignment (dfa) mode")

        module_converter = ModuleConverter(mode=self.mode)
        self.module = module_converter.convert(self.module, self.copy_weights, self.layer_config, self.output_dim)

    def forward(self, x, targets=None, loss_function=None):
        output = self.module(x)
        if self.mode == 'dfa' and self.module.training:
            if targets is None:
                raise ValueError('Targets missing for Direct Feedback Alignment mode')
            if loss_function is None:
                raise ValueError('You need to introduce your `loss_function` for Direct Feedback Alignment mode')
            loss = loss_function(output, targets)
            loss_gradient = grad(loss, output, retain_graph=True)[0]
            # Broadcast gradient of the loss to every layer
            for layer in self.module.modules():
                layer.loss_gradient = loss_gradient
        return output
