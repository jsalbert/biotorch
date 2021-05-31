from torch.utils.data import DataLoader
from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


class MNIST_Dataset(Dataset):
    def __init__(self, dataset_path='./data'):
        self.dataset_path = dataset_path
        self.train_dataset = datasets.MNIST(self.dataset_path, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
        self.test_dataset = datasets.MNIST(self.dataset_path, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

    def _create_dataloader(self, mode, batch_size):
        if mode == 'train':
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'test':
            return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def create_train_dataloader(self, batch_size):
        return self._create_dataloader('train', batch_size)

    def create_test_dataloader(self, batch_size):
        return self._create_dataloader('test', batch_size)
