import os
import torch
import shutil
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn


from biotorch.module.biomodule import BioModel
from biotorch.datasets.cifar import CIFAR10_Dataset
from biotorch.training.functions import train, test
from biotorch.utils.validator import validate_config, read_yaml


DATASETS_AVAILABLE = ['mnist', 'cifar10']


class Benchmark:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.config_file_path = config_file
            self.config_file = read_yaml(config_file)
            # Validate config file
            validate_config(self.config_file, 'benchmark', defaults=True)
            # Parse config file
            self.n_gpus = self.config_file['infrastructure']['num_gpus']
            self.hyperparameters = self.config_file['training']['hyperparameters']
            self.model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                                      and callable(models.__dict__[name]))
            self.dataset_names = DATASETS_AVAILABLE
            self.model_config = self.config_file['model']
            self.dataset_config = self.config_file['dataset']
            self.multi_gpu = False
            self.output_dir = self.config_file['experiment']['output_dir']
            # shutil.copy2(self.config_file_path, os.path.join(self.output_dir, 'config.yaml'))

    def run(self):
        if self.n_gpus == 0:
            device = 'cpu'
        elif self.n_gpus > 0 and not torch.cuda.is_available():
            raise ValueError('You selected {} GPUs but there are no GPUs available'.format(self.n_gpus))
        else:
            device = 'cuda'

        cudnn.benchmark = True

        self.best_acc = 0  # best test accuracy
        self.epochs = self.hyperparameters['epochs']
        self.batch_size = self.hyperparameters['batch_size']

        # Create dataset
        if self.dataset_config['name'] in self.dataset_names:
            self.dataset = CIFAR10_Dataset()
            self.n_classes = 10
            self.train_dataloader = self.dataset.create_train_dataloader(self.batch_size)
            self.test_dataloader = self.dataset.create_test_dataloader(self.batch_size)

        self.mode = self.model_config['mode']
        self.output_dim = self.n_classes

        # Create model
        if self.model_config['architecture'] is not None and self.model_config['architecture'] in self.model_names:
            if self.model_config['pretrained']:
                print("=> Using pre-trained model '{}'".format(self.model_config['architecture']))
                self.model = models.__dict__[self.model_config['architecture']](pretrained=True)
            else:
                print("=> Creating model '{}'".format(self.model_config['architecture']))
                self.model = models.__dict__[self.model_config['architecture']]()
            self.model.fc = nn.Linear(512, self.output_dim)

        elif self.model_config['checkpoint'] is not None:
            self.model = torch.load(self.model_config['checkpoint'])

        self.model = BioModel(self.model, mode=self.mode, output_dim=self.output_dim)
        self.model.to(device)

        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model, list(range(0, n_gpus)))
            self.multi_gpu = True

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4, weight_decay=0.)

        print('\nBenchmarking model on {}'.format(str(self.dataset)))

        trainer = Trainer()
        trainer.run()

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
