import torch.nn as nn

from biotorch.module.biomodule import BioModule


def create_resnet_biomodel(model_architecture,
                           mode,
                           pretrained: bool = False,
                           progress: bool = True,
                           num_classes: int = 1000) -> BioModule:
    if not pretrained:
        model = model_architecture(pretrained, progress, num_classes=num_classes)
    else:
        model = model_architecture(pretrained, progress, num_classes=1000)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return BioModule(model, mode=mode, output_dim=num_classes)
