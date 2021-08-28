import torch
import numpy
import random

from torchvision import transforms
from torch.utils.data import DataLoader


class Dataset(object):
    def __init__(self,
                 target_size,
                 dataset_path,
                 mean=None,
                 std=None,
                 train_transforms=None,
                 test_transforms=None):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        default_transforms = [
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
        ]

        if self.mean is not None and self.std is not None:
            default_transforms.append(transforms.Normalize(self.mean, self.std))

        if self.train_transforms is None:
            self.train_transforms = default_transforms

        if self.test_transforms is None:
            self.test_transforms = default_transforms

    # For reproducibility
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def _create_dataloader(self, mode, batch_size, deterministic=False, shuffle=True, drop_last=True, num_workers=0):
        # For reproducibility (deterministic dataloader sampling)
        gen = None
        worker_init_fn = None
        if deterministic and num_workers > 0:
            worker_init_fn = self.seed_worker
            gen = torch.Generator()
            gen.manual_seed(0)

        if mode == 'train':
            return DataLoader(self.train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              generator=gen
                              )
        elif mode == 'val':
            return DataLoader(self.val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              generator=gen
                              )
        elif mode == 'test':
            return DataLoader(self.test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              generator=gen
                              )

    def create_train_dataloader(self, batch_size, deterministic=False, num_workers=0):
        return self._create_dataloader('train', batch_size, deterministic=deterministic, num_workers=num_workers)

    def create_val_dataloader(self, batch_size, deterministic=False, num_workers=0):
        return self._create_dataloader('val', batch_size, deterministic=deterministic, num_workers=num_workers)

    def create_test_dataloader(self, batch_size, deterministic=False, num_workers=0):
        return self._create_dataloader('test', batch_size, deterministic=deterministic, num_workers=num_workers)
