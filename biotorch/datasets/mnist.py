from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


class MNIST(Dataset):
    def __str__(self):
        return 'MNIST Dataset'

    def __init__(self, target_size, dataset_path='./datasets/mnist', train_transforms=None, test_transforms=None):
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.num_classes = 10
        super(MNIST, self).__init__(target_size=target_size,
                                    dataset_path=dataset_path,
                                    mean=self.mean,
                                    std=self.std,
                                    train_transforms=train_transforms,
                                    test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        self.train_dataset = datasets.MNIST(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms)
        )

        self.val_dataset = datasets.MNIST(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )

        self.test_dataset = datasets.MNIST(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )
