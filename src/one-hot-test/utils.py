from typing import Union, List
import torch
from torch.types import Device
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize, ToTensor, Compose

import numpy as np
from matplotlib import pyplot as plt

from typing import Tuple


def get_optimal_device() -> Device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def plot_cifar_like_tensor(t: torch.Tensor) -> None:
    tt = t.permute(1, 2, 0).detach().cpu().numpy()
    print(f't: {t.shape}')
    print(f'tt: {tt.shape}')
    plt.imshow(tt)
    plt.show()


def denorm(t: torch.Tensor, mean: Union[Tuple, List, torch.Tensor], std: Union[Tuple, List, torch.Tensor]):
    """
    - xn = (x-mu)/std
    - x = xn * std + mu
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    t_denorm = t*std_tensor + mean_tensor
    return t_denorm.clamp(0.0, 1.0)


def get_cifar10(data_dir: str, batch_size: int, train: bool = True) -> Tuple[DataLoader, int]:

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    tf = Compose([ToTensor(), Normalize(mean=cifar10_mean, std=cifar10_std)])
    dataset = CIFAR10(data_dir, train=train, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader, 10


class RandomDataset(Dataset):
    def __init__(self, num_samples: int, in_channels: int, img_size: int, num_classes: int):
        self.num_samples = num_samples
        self.x = torch.randn(num_samples, in_channels, img_size, img_size)
        self.y_true = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y_true = self.y_true[idx]
        return x, y_true
