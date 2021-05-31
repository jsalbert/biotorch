import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from torchvision.models import resnet
from biotorch.module.biomodule import BioModel
from biotorch.datasets.cifar import CIFAR10_Dataset
from biotorch.training.functions import train, test

from biotorch.utils.validator import validate_config, read_yaml


class Benchmark:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.config_file = read_yaml(config_file)

    def run(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        epochs = 10
        batch_size = 32
        # validate_config(self.config, 'auto', defaults=True)

        cudnn.benchmark = True

        dataset = CIFAR10_Dataset()
        train_dataloader = dataset.create_train_dataloader(batch_size)
        test_dataloader = dataset.create_test_dataloader(batch_size)

        mode = 'DFA'
        model = resnet.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 10)
        output_dim = 10
        model = BioModel(model, mode=mode, output_dim=output_dim)
        model.to(device)
        self.model = model
        crossentropy_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=0.)

        print('\nBenchmarking model on {}'.format(str(dataset)))
        for epoch in range(epochs):
            train(
                model=model,
                mode=mode,
                loss_function=crossentropy_loss,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                epoch=epoch
            )
            acc, loss = test(
                model=model,
                loss_function=crossentropy_loss,
                test_dataloader=test_dataloader,
                device=device,
            )

            # Remember best acc@1 and save checkpoint
            acc > best_acc
            best_acc = max(acc, best_acc)


def __main__():
    parser = argparse.ArgumentParser(
        description='BioTorch'
    )
    parser.add_argument('config_file')

    try:
        benchmark = Benchmark(args.config_file)
        benchmark.run()

    except Exception as e:
        message = 'an unexpected error occurred: {}: {}'.format(
            type(e).__name__,
            (e.message if hasattr(e, 'message') else '') or str(e)
        )
        raise ValueError(message)
