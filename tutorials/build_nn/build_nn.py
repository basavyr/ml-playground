# source: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.functional


import model as m
import data_loader as dl


def init_model_parameters():
    n_samples = 1000
    input_size = 28
    output_size = 10
    params = [1, 1]
    batch_size = 128
    return n_samples, input_size, output_size, params, batch_size


def train(train_dataloader, device, model, loss_function, optimizer, batch_size=128):

    train_dataloader, _ = dl.get_loaders(batch_size)
    size = len(train_dataloader.dataset)  # size of data

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    # device
    device = m.MPS_DEVICE

    # params
    n_samples, input_size, output_size, params, batch_size = init_model_parameters()
    # data sets
    train_dataloader, test_dataloader = dl.get_loaders(batch_size=batch_size)

    # model
    model = m.Model(input_size, output_size, params).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    train(train_dataloader, device, model, loss_fn, optimizer, batch_size)


if __name__ == "__main__":
    main()
