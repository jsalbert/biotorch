from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


class FashionMNIST(Dataset):
    def __str__(self):
        return 'Fashion MNIST Dataset'

    def __init__(self, target_size, dataset_path='./datasets/fashion-mnist', train_transforms=None, test_transforms=None):
        self.mean = (0.2859,)
        self.std = (0.3530,)
        self.num_classes = 10
        super(FashionMNIST, self).__init__(target_size=target_size,
                                           dataset_path=dataset_path,
                                           mean=self.mean,
                                           std=self.std,
                                           train_transforms=train_transforms,
                                           test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        self.train_dataset = datasets.FashionMNIST(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms)
        )

        self.val_dataset = datasets.FashionMNIST(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )

        self.test_dataset = datasets.FashionMNIST(
            self.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms)
        )
