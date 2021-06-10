import os
import torch
import shutil
import torch.nn as nn
import torch.backends.cudnn as cudnn

from types import ModuleType
import biotorch.models as models
from biotorch.datasets.selector import DatasetSelector
from biotorch.training.trainer import Trainer
from biotorch.utils.validator import validate_config
from biotorch.utils.utils import read_yaml, mkdir


DATASETS_AVAILABLE = ['mnist', 'cifar10', 'cifar100']


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
            self.dataset_names = DATASETS_AVAILABLE
            self.model_config = self.config_file['model']
            self.dataset_config = self.config_file['dataset']

            self.mode_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                                     and isinstance(models.__dict__[name], ModuleType))

            self.mode = self.model_config['mode']
            if self.mode not in self.mode_names:
                raise ValueError("Mode not supported")

            options = models.__dict__[self.mode].__dict__
            self.model_names = sorted(name for name in options if name.islower() and not name.startswith("__")
                                      and callable(options[name]))
            self.multi_gpu = False
            self.output_dir = self.config_file['experiment']['output_dir']
            mkdir(self.output_dir)
            shutil.copy2(self.config_file_path, os.path.join(self.output_dir, 'config.yaml'))

    def run(self):
        if self.n_gpus == 0:
            self.device = 'cpu'
        elif self.n_gpus > 0 and not torch.cuda.is_available():
            raise ValueError('You selected {} GPUs but there are no GPUs available'.format(self.n_gpus))
        else:
            self.device = 'cuda'

        cudnn.benchmark = True

        self.best_acc = 0  # best test accuracy
        self.epochs = self.hyperparameters['epochs']
        self.batch_size = self.hyperparameters['batch_size']
        self.target_size = self.hyperparameters['target_size']

        # Create dataset
        if self.dataset_config['name'] in self.dataset_names:
            self.dataset_creator = DatasetSelector(self.dataset_config['name']).get_dataset()
            self.dataset = self.dataset_creator(self.target_size)
            self.train_dataloader = self.dataset.create_train_dataloader(self.batch_size)
            self.test_dataloader = self.dataset.create_test_dataloader(self.batch_size)
            self.num_classes = self.dataset.num_classes

        # Create model
        if self.model_config['architecture'] is not None and self.model_config['architecture'] in self.model_names:
            if self.model_config['pretrained']:
                print("=> Using pre-trained model '{}'".format(self.model_config['architecture']))
                self.model = models.__dict__[self.model_config['mode']].__dict__[
                    self.model_config['architecture']](pretrained=True, )
            else:
                print("=> Creating model from sratch'{}'".format(self.model_config['architecture']))
                self.model = models.__dict__[self.model_config['mode']].__dict__[
                    self.model_config['architecture']]()

        elif self.model_config['checkpoint'] is not None:
            self.model = torch.load(self.model_config['checkpoint'])

        self.model.to(self.device)

        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model, list(range(0, self.n_gpus)))
            self.multi_gpu = True

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4, weight_decay=0.)

        print('\nBenchmarking model on {}'.format(str(self.dataset)))

        trainer = Trainer(model=self.model,
                          mode=self.mode,
                          loss_function=self.loss_function,
                          optimizer=self.optimizer,
                          train_dataloader=self.train_dataloader,
                          test_dataloader=self.test_dataloader,
                          device=self.device,
                          epochs=self.epochs,
                          multi_gpu=self.multi_gpu,
                          display_iterations=500)
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
