import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Tuple
import os
import sys

DEFAULT_DATA_DIR: str | None = os.environ.get("DEFAULT_DATA_DIR", None)
assert DEFAULT_DATA_DIR is not None, "Environment variable < DEFAULT_DATA_DIR > is not set."

DEFAULT_SEED = 1137


def test_plot(dataloader: DataLoader):
    from utils import plot_cifar_like_tensor, denorm

    x, _ = next(iter(dataloader))
    x0 = x[0]
    xdn = denorm(x0, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    plot_cifar_like_tensor(xdn)


def get_cifar10(data_dir: str, batch_size: int, train: bool = True) -> Tuple[DataLoader, int]:
    from torchvision.datasets import CIFAR10
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    tf = Compose([ToTensor(), Normalize(mean=cifar10_mean, std=cifar10_std)])
    dataset = CIFAR10(data_dir, train=train, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader, 10


def test_inference(dataloader: DataLoader, num_classes: int,  device: torch.types.Device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.to(device)

    x, y_true = next(iter(dataloader))
    x, y_true = x.to(device), y_true.to(device)
    x0 = x[0].unsqueeze(dim=0)
    yt0 = y_true[0].unsqueeze(dim=0)
    yt0_one_hot = F.one_hot(yt0, num_classes=num_classes)
    print(f'x0: {x0.shape}')
    print(f'yt0: {yt0.shape}')
    print(f'yt0_oh: {yt0_one_hot.shape}')

    loss_fn = nn.CrossEntropyLoss()
    was_training = model.training
    with torch.inference_mode():
        model.eval()
        y = model(x0)
        loss = loss_fn(y, yt0)
        print(loss.item())

    if was_training:
        model.train()

    print(f'data: {x.shape, y_true.shape}')
    print(f'idx=0: {x0.shape, yt0_one_hot}')


def main():
    torch.manual_seed(DEFAULT_SEED)
    cifar10, num_classes = get_cifar10(DEFAULT_DATA_DIR, 128)
    test_inference(cifar10, num_classes, torch.device("mps"))
    test_plot(cifar10)


if __name__ == "__main__":
    main()
