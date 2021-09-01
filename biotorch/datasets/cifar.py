import random
import numpy as np

from torchvision import datasets, transforms
from biotorch.datasets.dataset import Dataset


class CIFAR100(Dataset):
    def __str__(self):
        return 'CIFAR-100 Dataset'

    def __init__(self, target_size, dataset_path='./datasets/cifar100', train_transforms=None, test_transforms=None):
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.num_classes = 100

        super(CIFAR100, self).__init__(target_size=target_size,
                                       dataset_path=dataset_path,
                                       mean=self.mean,
                                       std=self.std,
                                       train_transforms=train_transforms,
                                       test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        self.train_dataset = datasets.CIFAR100(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms)
        )

        self.val_dataset = datasets.CIFAR100(self.dataset_path,
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose(self.test_transforms)
                                             )

        self.test_dataset = datasets.CIFAR100(self.dataset_path,
                                              train=False,
                                              download=True,
                                              transform=transforms.Compose(self.test_transforms)
                                              )


class CIFAR10(Dataset):
    def __str__(self):
        return 'CIFAR-10 Dataset'

    def __init__(self, target_size, dataset_path='./datasets/cifar10', train_transforms=None, test_transforms=None):
        self.mean = (0.4914, 0.4821, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        self.num_classes = 10

        super(CIFAR10, self).__init__(target_size=target_size,
                                      dataset_path=dataset_path,
                                      mean=self.mean,
                                      std=self.std,
                                      train_transforms=train_transforms,
                                      test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        self.train_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms)
        )

        self.val_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )

        # Test dataset is 10k
        self.test_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )


class CIFAR10Benchmark(Dataset):
    def __str__(self):
        return 'CIFAR-10 Benchmark Dataset'

    def __init__(self, target_size, dataset_path='./datasets/cifar10', train_transforms=None, test_transforms=None):
        self.mean = (0.4914, 0.4821, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        self.num_classes = 10

        super(CIFAR10Benchmark, self).__init__(target_size=target_size,
                                               dataset_path=dataset_path,
                                               mean=self.mean,
                                               std=self.std,
                                               train_transforms=train_transforms,
                                               test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        # Train is 45k and validation is 5k as in (https://arxiv.org/pdf/1512.03385.pdf)

        # Seed is fixed
        random.seed(0)
        self.train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]

        self.train_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms)
        )

        val_indices = random.sample(range(0, len(self.train_dataset.data)), 5000)
        self.train_dataset.data = np.delete(self.train_dataset.data, val_indices, axis=0)
        self.train_dataset.targets = np.delete(self.train_dataset.targets, val_indices, axis=0)

        self.val_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )

        self.val_dataset.data = self.val_dataset.data[val_indices]
        self.val_dataset.targets = list(np.array(self.val_dataset.targets)[val_indices])

        # Test dataset is 10k
        self.test_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )
