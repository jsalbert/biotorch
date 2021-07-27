import os


from torchvision import datasets, transforms
from biotorch.datasets.dataset import Dataset


class ImageNet(Dataset):
    def __str__(self):
        return 'Imagenet Dataset'

    def __init__(self, target_size, dataset_path='./datasets/imagenet', train_transforms=None, test_transforms=None):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 1000
        super(ImageNet, self).__init__(target_size=target_size,
                                       dataset_path=dataset_path,
                                       mean=self.mean,
                                       std=self.std,
                                       train_transforms=train_transforms,
                                       test_transforms=test_transforms)

        print('Preparing {} and storing data in {}'.format(str(self), dataset_path))

        self.train_dataset = datasets.ImageFolder(
            os.path.join(self.dataset_path, 'train'),
            transform=transforms.Compose(self.train_transforms)
        )

        self.val_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'val'),
                                                transform=transforms.Compose(self.test_transforms)
                                                )

        self.test_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'val'),
                                                 transform=transforms.Compose(self.test_transforms)
                                                 )
