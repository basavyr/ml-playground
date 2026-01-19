import os
import sys
from dataclasses import dataclass


import torch
from torch import nn as nn
import torch.nn.functional as F

# local imports
from utils import get_optimal_device, set_deterministic_behavior
from models import VisionTransformer


DEFAULT_DATA_DIR: str = str(os.getenv("DEFAULT_DATA_DIR", None))
assert DEFAULT_DATA_DIR is not None, "Environment variable < DEFAULT_DATA_DIR > is not set."


@dataclass
class TrainingConfig:
    device: torch.types.Device
    batch_size: int
    epochs: int
    seed: int | None = None


def train_vit(training_config: TrainingConfig):
    x = torch.randn((training_config.batch_size, 3, 28, 28))
    y_true = torch.randint(0, 10, (training_config.batch_size,))
    vit = VisionTransformer(img_size=x.shape[-1],
                            patch_size=4,
                            in_channels=3,
                            num_classes=10)
    vit.to(training_config.device)
    vit.train()

    # test_inference
    x = x.to(training_config.device)
    y_true = y_true.to(training_config.device)
    y = vit(x)
    print(x.shape, y.shape, y.device, y)


def main():
    device = get_optimal_device()
    training_config = TrainingConfig(
        device=device,
        batch_size=128,
        epochs=1,
        seed=1137)
    set_deterministic_behavior(training_config.seed)
    train_vit(training_config=training_config)


if __name__ == "__main__":
    main()
