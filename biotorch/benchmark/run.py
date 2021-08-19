import os
import torch
import random
import shutil
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import biotorch.models as models
import torch.backends.cudnn as cudnn


from types import ModuleType
from biotorch.training.trainer import Trainer
from biotorch.evaluation.evaluator import Evaluator
from biotorch.utils.utils import read_yaml, mkdir
from biotorch.models.utils import apply_xavier_init
from biotorch.utils.validator import validate_config
from biotorch.datasets.selector import DatasetSelector
from biotorch.benchmark.optimizers import create_optimizer
from biotorch.benchmark.lr_schedulers import create_lr_scheduler
from biotorch.benchmark.losses import select_loss_function


class Benchmark:
    def __init__(self, config_file):
        self.config_file_path = config_file
        self.config_file = read_yaml(config_file)
        # Validate config file
        validate_config(self.config_file, 'benchmark', defaults=True)
        # Set seed for reproducibility
        torch.manual_seed(self.config_file['experiment']['seed'])
        random.seed(self.config_file['experiment']['seed'])
        np.random.seed(self.config_file['experiment']['seed'])

        # Reproducibility
        self.deterministic = self.config_file['experiment']['deterministic']
        if self.deterministic:
            cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            cudnn.benchmark = True

        # Parse config file
        self.gpus = self.config_file['infrastructure']['gpus']
        self.model_config = self.config_file['model']

        if 'training' not in self.config_file:
            self.benchmark_mode = 'evaluation'
        else:
            self.benchmark_mode = 'training'
            self.hyperparameters = self.config_file['training']['hyperparameters']
            self.metrics_config = self.config_file['training']['metrics']
            self.optimizer_config = self.config_file['training']['optimizer']
            self.lr_scheduler_config = self.config_file['training']['lr_scheduler']
            self.mode_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                                     and isinstance(models.__dict__[name], ModuleType))

            self.mode = self.model_config['mode']['type']

            self.layer_config = {}
            self.layer_config = {"type": self.mode}
            if 'options' in self.model_config['mode']:
                self.mode_options = self.model_config['mode']['options']
                self.layer_config["options"] = self.mode_options

            if self.mode not in self.mode_names:
                raise ValueError("Mode not {} supported".format(self.mode))

            options = models.__dict__[self.mode].__dict__
            self.model_names = sorted(name for name in options if name.islower() and not name.startswith("__")
                                      and callable(options[name]))

        self.loss_function_config = self.config_file['model']['loss_function']
        self.data_config = self.config_file['data']
        self.num_workers = self.config_file['data']['num_workers']
        self.multi_gpu = False

        if isinstance(self.gpus, int) and self.gpus <= -1:
            self.device = 'cpu'
        else:
            if not torch.cuda.is_available():
                raise ValueError('You selected {} GPUs but there are no GPUs available'.format(self.gpus))
            self.device = 'cuda'
            if isinstance(self.gpus, int):
                self.device += ':' + str(self.gpus)
            elif isinstance(self.gpus, list):
                self.device += ':' + str(self.gpus[0])
                self.multi_gpu = True

        self.output_dir = os.path.join(self.config_file['experiment']['output_dir'],
                                       self.config_file['experiment']['name'])
        mkdir(self.output_dir)
        shutil.copy2(self.config_file_path, os.path.join(self.output_dir, 'config.yaml'))

    def run(self):
        self.epochs = self.hyperparameters['epochs']
        self.batch_size = self.hyperparameters['batch_size']
        self.target_size = self.data_config['target_size']

        # Create dataset
        self.dataset_creator = DatasetSelector(self.data_config['dataset']).get_dataset()
        if self.data_config['dataset_path'] is not None:
            self.dataset = self.dataset_creator(self.target_size, dataset_path=self.data_config['dataset_path'])
        else:
            self.dataset = self.dataset_creator(self.target_size)

        self.train_dataloader = self.dataset.create_train_dataloader(self.batch_size,
                                                                     deterministic=self.deterministic,
                                                                     num_workers=self.num_workers)
        self.val_dataloader = self.dataset.create_val_dataloader(self.batch_size,
                                                                 deterministic=self.deterministic,
                                                                 num_workers=self.num_workers)
        self.num_classes = self.dataset.num_classes

        # Create model
        if self.model_config['architecture'] is not None and self.model_config['architecture'] in self.model_names:
            arch = self.model_config['architecture']
            if self.model_config['pretrained']:
                print("=> Using pre-trained model '{}'".format(self.model_config['architecture']))
            else:
                print("=> Creating model from scratch '{}'".format(self.model_config['architecture']))

                self.model = models.__dict__[self.mode].__dict__[arch](
                    pretrained=self.model_config['pretrained'],
                    num_classes=self.num_classes,
                    layer_config=self.layer_config
                )

        elif self.model_config['checkpoint'] is not None:
            print('Loading model checkpoint from ', self.model_config['checkpoint'])
            self.model = torch.load(self.model_config['checkpoint'], map_location=self.device)

        self.model.to(self.device)

        if isinstance(self.gpus, list):
            self.model = nn.DataParallel(self.model, self.gpus)

        self.loss_function = select_loss_function(self.loss_function_config)
        self.optimizer = create_optimizer(self.optimizer_config, self.model)
        self.lr_scheduler = create_lr_scheduler(self.lr_scheduler_config, self.optimizer)

        print('\nBenchmarking model on {}'.format(str(self.dataset)))
        print(self.metrics_config)

        trainer = Trainer(model=self.model,
                          mode=self.mode,
                          loss_function=self.loss_function,
                          optimizer=self.optimizer,
                          lr_scheduler=self.lr_scheduler,
                          train_dataloader=self.train_dataloader,
                          val_dataloader=self.val_dataloader,
                          device=self.device,
                          epochs=self.epochs,
                          output_dir=self.output_dir,
                          metrics_config=self.metrics_config,
                          multi_gpu=self.multi_gpu
                          )
        trainer.run()

        if self.config_file['evaluation']:
            self.model = torch.load(os.path.join(self.output_dir, 'model_best_acc.pth'))
            self.test_dataloader = self.dataset.create_test_dataloader(self.batch_size,
                                                                       deterministic=self.deterministic,
                                                                       num_workers=self.num_workers
                                                                       )

            self.evaluator = Evaluator(
                self.model,
                self.mode,
                self.loss_function,
                self.test_dataloader,
                self.device,
                self.output_dir
            )

        self.evaluate(self.evaluator)

    def run_eval(self):
        if self.model_config['checkpoint'] is not None:
            print('Loading model checkpoint from ', self.model_config['checkpoint'])
            self.model = torch.load(self.model_config['checkpoint'], map_location=self.device)
        else:
            raise ValueError('A model checkpoint must be specified')
        self.target_size = self.data_config['target_size']
        self.batch_size = self.data_config['batch_size']
        # Create dataset
        self.dataset_creator = DatasetSelector(self.data_config['dataset']).get_dataset()
        if self.data_config['dataset_path'] is not None:
            self.dataset = self.dataset_creator(self.target_size, dataset_path=self.data_config['dataset_path'])
        else:
            self.dataset = self.dataset_creator(self.target_size)

        self.test_dataloader = self.dataset.create_test_dataloader(self.batch_size,
                                                                   deterministic=self.deterministic,
                                                                   num_workers=self.num_workers)

        self.loss_function = select_loss_function(self.loss_function_config)

        self.evaluator = Evaluator(
            self.model,
            None,
            self.loss_function,
            self.test_dataloader,
            self.device,
            self.output_dir
        )

        self.evaluate(self.evaluator)

    def evaluate(self, evaluator):
        self.test_acc, self.test_loss = evaluator.run()
        self.results_df = pd.DataFrame({
            'model_name': [self.config_file['experiment']['name']],
            'dataset': [self.data_config['dataset']],
            'accuracy': [float(self.test_acc)],
            'error': [100.0 - float(self.test_acc)],
            'loss': [self.test_loss]
        })
        print('Test Results')
        print(self.results_df)
        csv_results = os.path.join(self.output_dir, 'results.csv')
        json_results = os.path.join(self.output_dir, 'results.json')
        print('Test Results saved in: \n{}\n{}'.format(csv_results, json_results))

        self.results_df.to_csv(csv_results)
        self.results_df.to_json(json_results, indent=2, orient='records')


def __main__():
    parser = argparse.ArgumentParser(
        description='BioTorch'
    )
    parser.add_argument('--config_file', help="Path to the configuration file")

    try:
        args = parser.parse_args()
        benchmark = Benchmark(args.config_file)
        if benchmark.benchmark_mode == 'training':
            benchmark.run()
        else:
            benchmark.run_eval()

    except Exception as e:
        message = 'an unexpected error occurred: {}: {}'.format(
            type(e).__name__,
            (e.message if hasattr(e, 'message') else '') or str(e)
        )
        raise ValueError(message)
