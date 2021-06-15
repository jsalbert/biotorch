from biotorch.datasets.dataset import Dataset
from torchvision import datasets, transforms


class TinyImageNet(Dataset):
    def __str__(self):
        return 'Tiny Imagenet Dataset'

    def __init__(self, target_size, dataset_path='./datasets/tiny_imagenet', train_transforms=None, test_transforms=None):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 100
        super(TinyImageNet, self).__init__(target_size=target_size,
                                           dataset_path=dataset_path,
                                           mean=self.mean,
                                           std=self.std,
                                           train_transforms=train_transforms,
                                           test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        train_dir = os.path.join(self.dataset_path, 'train')

        self.train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transforms.Compose(self.train_transforms)
        )

        val_dir = os.path.join(self.dataset_path, 'val')
        self.val_dataset = datasets.ImageFolder(root=val_dir,
                                                transform=transforms.Compose(self.test_transforms)
                                                )

        test_dir = os.path.join(self.dataset_path, 'test')
        self.test_dataset = datasets.ImageFolder(root=test_dir,
                                                 transform=transforms.Compose(self.test_transforms)
                                                 )

