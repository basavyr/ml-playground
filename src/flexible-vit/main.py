import os
import sys
from dataclasses import dataclass
from tqdm import tqdm, trange

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
# local imports
from models import VisionTransformer, get_patch_size_vit
from utils import get_optimal_device, set_deterministic_behavior
from random_datasets import get_dataloader_and_config, DataConfig

DEFAULT_DATA_DIR: str = str(os.getenv("DEFAULT_DATA_DIR", None))
assert DEFAULT_DATA_DIR is not None, "Environment variable < DEFAULT_DATA_DIR > is not set."


@dataclass
class TrainingConfig:
    device: torch.types.Device
    batch_size: int
    epochs: int
    lr: float = 0.01
    seed: int | None = None


def train_vit(vit_model: nn.Module, patch_size: int, training_config: TrainingConfig, data_config: DataConfig, trainloader: DataLoader):
    vit = vit_model
    vit.to(training_config.device)
    vit.train()

    optimizer = torch.optim.SGD(vit.parameters(
    ), lr=training_config.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.epochs, eta_min=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    pbar = tqdm(total=training_config.epochs *
                data_config.num_samples, desc=f'Training ViT on {data_config.dataset_type}', dynamic_ncols=True)
    epoch_loss = None
    for _ in range(training_config.epochs):
        num_samples = 0
        epoch_loss = 0.0
        for x, y_true in trainloader:
            x, y_true = x.to(training_config.device), y_true.to(
                training_config.device)
            optimizer.zero_grad()

            y = vit(x)
            loss = loss_fn(y, y_true)
            epoch_loss += loss.item()*x.shape[0]
            num_samples += x.shape[0]

            loss.backward()
            optimizer.step()
            pbar.update(x.shape[0])

        epoch_loss = epoch_loss/num_samples
        scheduler.step()

    pbar.close()
    os.makedirs("./models", exist_ok=True)
    model_pth = f'vit-img{data_config.img_size}-p{patch_size}_{training_config.epochs}epochs.pt'
    model = {
        "state_dict": vit.state_dict(),
        "training_config": training_config,
        "data_config": data_config,
    }
    torch.save(model, f'models/{model_pth}')
    print(f'Final loss: {epoch_loss} | Checkpoint -> {model_pth}')


def finetune_vit(model_checkpoint: str, finetuning_config: TrainingConfig, checkpoint_data_config: DataConfig, new_data_config: DataConfig):
    raise NotADirectoryError("TBD")


def main():
    device = get_optimal_device()
    training_config = TrainingConfig(
        device=device,
        batch_size=128,
        epochs=1,
        seed=1137)
    set_deterministic_behavior(training_config.seed)

    dataset_type = "mnist"
    trainloader, data_config = get_dataloader_and_config(
        dataset_type=dataset_type,
        num_samples=1000,
        batch_size=128,
        train=True)

    patch_size = get_patch_size_vit(data_config.img_size)
    vit = VisionTransformer(img_size=data_config.img_size,
                            patch_size=patch_size,
                            in_channels=data_config.in_channels,
                            num_classes=data_config.num_classes,
                            dynamic_img_size=True)
    train_vit(vit, patch_size, training_config, data_config, trainloader)


if __name__ == "__main__":
    main()
