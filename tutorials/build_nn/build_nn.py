# source: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.functional


def get_device():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def main():
    # in order to have a model we need to import nn from pytorch
    model = nn.Sequential(nn.Linear(10, 10))
    return model


if __name__ == "__main__":
    device = get_device()
