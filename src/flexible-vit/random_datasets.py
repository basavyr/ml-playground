from torch.utils.data import Dataset, DataLoader
import torch

from torchvision.transforms import Compose, ToTensor
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
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


def get_dataloader(num_samples: int, batch_size: int, img_size: int, in_channels: int, num_classes: int, train: bool = True):
    dataset = ClassificationData(
        num_samples, img_size, in_channels, num_classes)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
