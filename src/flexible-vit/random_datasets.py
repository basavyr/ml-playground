from torch.utils.data import Dataset, DataLoader
import torch


from typing import Tuple
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_type: str
    num_samples: int
    img_size: int
    in_channels: int
    num_classes: int
    train: bool = True


class ClassificationData(Dataset):
    def __init__(self, num_samples: int,  img_size: int, in_channels: int, num_classes: int):
        self.samples = torch.randn(
            (num_samples, in_channels, img_size, img_size))
        self.labels = torch.randint(1, num_classes, (num_samples,))
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = self.samples[idx]
        y_true = self.labels[idx]
        return x, y_true


def get_dataloader_and_config(dataset_type: str, num_samples: int, batch_size: int, train: bool = True) -> Tuple[DataLoader, DataConfig]:
    if dataset_type == "mnist":
        data_config = DataConfig(dataset_type=dataset_type,
                                 num_samples=num_samples,
                                 img_size=28,
                                 in_channels=3,
                                 num_classes=10,
                                 train=True)
    elif dataset_type == "cifar10":
        data_config = DataConfig(dataset_type=dataset_type,
                                 num_samples=num_samples,
                                 img_size=32,
                                 in_channels=3,
                                 num_classes=10,
                                 train=True)
    elif dataset_type == "tiny" or dataset_type == "tiny-imagenet-200":
        data_config = DataConfig(dataset_type=dataset_type,
                                 num_samples=num_samples,
                                 img_size=64,
                                 in_channels=3,
                                 num_classes=200,
                                 train=True)
    elif dataset_type == "cifar100":
        data_config = DataConfig(dataset_type=dataset_type,
                                 num_samples=num_samples,
                                 img_size=32,
                                 in_channels=3,
                                 num_classes=100,
                                 train=True)
    elif dataset_type == "imagenet-100":
        data_config = DataConfig(dataset_type=dataset_type,
                                 num_samples=num_samples,
                                 img_size=224,
                                 in_channels=3,
                                 num_classes=100,
                                 train=True)
    else:
        raise ValueError(f"Incorrect dataset type: {dataset_type}")

    dataset = ClassificationData(
        num_samples, data_config.img_size, data_config.in_channels, data_config.num_classes)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, data_config
