import os
import sys
from dataclasses import dataclass
from tqdm import tqdm

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
# local imports
from models import VisionTransformer
from utils import get_optimal_device, set_deterministic_behavior
from random_datasets import get_dataloader, DataConfig

DEFAULT_DATA_DIR: str = str(os.getenv("DEFAULT_DATA_DIR", None))
assert DEFAULT_DATA_DIR is not None, "Environment variable < DEFAULT_DATA_DIR > is not set."


@dataclass
class TrainingConfig:
    device: torch.types.Device
    batch_size: int
    epochs: int
    lr: float = 0.01
    seed: int | None = None


def train_vit(training_config: TrainingConfig, data_config: DataConfig, trainloader: DataLoader):
    vit = VisionTransformer(img_size=data_config.img_size,
                            patch_size=4,
                            in_channels=data_config.in_channels,
                            num_classes=data_config.num_classes)
    vit.to(training_config.device)
    vit.train()

    optimizer = torch.optim.SGD(vit.parameters(
    ), lr=training_config.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.epochs, eta_min=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(training_config.epochs):
        epoch_loss = 0.0
        for x, y_true in tqdm(trainloader, desc=f'Epoch {epoch+1}: training dataset'):
            x, y_true = x.to(training_config.device), y_true.to(
                training_config.device)
            optimizer.zero_grad()

            y = vit(x)
            loss = loss_fn(y, y_true)
            epoch_loss += loss.item()*x.shape[0]

            loss.backward()
            optimizer.step()

        epoch_loss /= len(trainloader.dataset)

        print(f'Epoch loss: {epoch_loss}')
        scheduler.step()


def main():
    device = get_optimal_device()
    training_config = TrainingConfig(
        device=device,
        batch_size=128,
        epochs=2,
        seed=1137)
    set_deterministic_behavior(training_config.seed)
    data_config = DataConfig(num_samples=1000,
                             img_size=32,
                             in_channels=3,
                             num_classes=10,
                             train=True)
    trainloader = get_dataloader(num_samples=data_config.num_samples,
                                 batch_size=training_config.batch_size,
                                 img_size=data_config.img_size,
                                 in_channels=data_config.in_channels,
                                 num_classes=data_config.num_classes,
                                 train=data_config.train)
    train_vit(training_config, data_config, trainloader)


if __name__ == "__main__":
    main()
