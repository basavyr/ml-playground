# source: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.functional


import model as m
import data_loader as dl


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


def get_model(input_size, output_size, params, device):
    model = m.Model(input_size, output_size, params).to(device)
    return model


def init_model_parameters():
    n_samples = 1000
    input_size = 250
    output_size = 3
    params = [1, 1]
    return n_samples, input_size, output_size, params


def test_tensor(model):
    device = get_device()
    n_samples, input_size, output_size, params = init_model_parameters()
    X = torch.rand(n_samples, input_size, input_size, device=device)

    # evaluate the model
    Y_pred = model(X)

    print(f"Predicted class: {Y_pred}")

    print(f"Model structure: {model}\n\n")


def train(train_dataloader, model, loss_function, optimizer):
    print(model)

    exit(1)


def main():
    # get device
    device = get_device()
    n_samples, input_size, output_size, params = init_model_parameters()

    # automatically download the data sets and create the loaders for both training and test data
    train_dataloader, test_dataloader = dl.get_loaders(batch_size=128)

    model = get_model(input_size, output_size, params, device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    train(train_dataloader, model, loss_fn, optimizer)


if __name__ == "__main__":
    main()
