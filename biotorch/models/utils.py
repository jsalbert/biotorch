import torch.nn as nn

from biotorch.module.biomodule import BioModule


def create_torchvision_biomodel(model_architecture,
                                mode,
                                layer_config: dict = None,
                                pretrained: bool = False,
                                progress: bool = True,
                                num_classes: int = 1000) -> BioModule:
    if not pretrained:
        copy_weights = False
        model = model_architecture(pretrained, progress, num_classes=num_classes)
    else:
        copy_weights = True
        model = model_architecture(pretrained, progress, num_classes=1000)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return BioModule(model, mode=mode, copy_weights=copy_weights, layer_config=layer_config, output_dim=num_classes)


def create_le_net_biomodel(model_architecture,
                           mode,
                           layer_config: dict = None,
                           pretrained: bool = False,
                           progress: bool = True,
                           num_classes: int = 10) -> BioModule:

    model = model_architecture(pretrained, progress, num_classes=num_classes)

    return BioModule(model, mode=mode, copy_weights=False, layer_config=layer_config, output_dim=num_classes)


def apply_xavier_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
