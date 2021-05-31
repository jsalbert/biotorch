from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


class CIFAR10_Dataset(Dataset):
    def __str__(self):
        return 'CIFAR10 Dataset'

    def __init__(self, dataset_path='./data'):
        self.dataset_path = dataset_path
        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))
        self.train_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
        )

        self.test_dataset = datasets.CIFAR10(self.dataset_path,
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
