from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


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
