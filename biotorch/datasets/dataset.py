from torch.utils.data import DataLoader


class Dataset(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def _create_dataloader(self, mode, batch_size):
        if mode == 'train':
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'test':
            return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def create_train_dataloader(self, batch_size):
        return self._create_dataloader('train', batch_size)

    def create_test_dataloader(self, batch_size):
        return self._create_dataloader('test', batch_size)
