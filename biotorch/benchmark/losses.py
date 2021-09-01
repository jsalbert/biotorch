import torch


def select_loss_function(loss_function_config):
    if loss_function_config['name'] == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
