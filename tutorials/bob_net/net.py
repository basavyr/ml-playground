# source: https://www.youtube.com/watch?v=JRlyw6LO5qo&list=PL_LG-VOo0eUykJPa7r8zyJq7pq9MKJNXQ

import os
import torch

import torch.nn as nn

import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import torchvision.transforms as transforms

import torch.optim as optim


import matplotlib.pyplot as plt


import tqdm


class Net(nn.Module):
    def __init__(self, input_size: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.tensor):
        x = x.view(x.size(0), -1)  # flatten the input
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        logits = self.fc3(x)
        return logits


def train_model(model: nn.Module, device: torch.device, data_loader: DataLoader, num_epochs: int) -> None:
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []
    epoch_accs = []
    once = False
    for epoch in range(num_epochs):
        running_loss = 0
        correct_labels_per_epoch = 0
        total_labels = 0
        for idx, batch in enumerate(data_loader):
            X, Y = batch

            if once:
                img = X[0].squeeze()
                plt.imshow(img, cmap="gray")
                plt.show()
                once = False

            if isinstance(Y, torch.Tensor):
                Y = Y.clone().detach()
            else:
                Y = torch.tensor(Y, dtype=torch.long)

            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            y_pred = model(X)

            correct_labels_per_batch = (
                torch.argmax(y_pred, dim=1) == Y).sum().item()
            correct_labels_per_epoch += correct_labels_per_batch
            total_labels += X.shape[0]

            loss = loss_fn(y_pred, Y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_losses.append(running_loss / len(data_loader))
        epoch_accs.append(float(correct_labels_per_epoch/total_labels))
    for idx, data in enumerate(zip(epoch_losses, epoch_accs)):
        loss, acc = data
        print(f'Epoch: {idx + 1}')
        print(f'Loss: {loss}')
        print(f'Acc: {acc}\n')

    torch.save(model, 'model.pth')


def test_model(model: nn.Module, device: torch.device, data_loader: DataLoader, iter_stop: int = 10) -> None:
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            X, Y = batch
            if isinstance(Y, torch.Tensor):
                Y = Y.clone().detach()
            else:
                Y = torch.tensor(Y, dtype=torch.long)

            X, Y = X.to(device), Y.to(device)

            y_pred = model(X)

            preds = torch.argmax(y_pred, dim=1)
            # acc_perc = (preds == Y).float().sum()/(preds == Y).numel()
            # print(f'Acc -> {acc_perc}')

            print(f'Batch # {idx}')
            loss = loss_fn(y_pred, Y)
            print(f'Loss -> {loss.item()}')
            acc_perc = (preds == Y).float().mean()
            print(f'Acc -> {acc_perc}')
            if idx == iter_stop and iter_stop != -1:
                break


def main():
    height = 28
    width = 28
    n_channels = 1
    input_size = height*width
    n_samples = 1
    batch_size = 128

    device = torch.device("mps")

    model = Net(input_size)
    model.to(device)

    x = torch.randint(0, 256, size=(
        n_samples, n_channels, height, width), dtype=torch.float32).to(device)

    train_dataset = datasets.MNIST(
        "./", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(
        "./", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    if not os.path.exists("model.pth"):
        print('Training the model')
        train_model(model, device, train_loader, 10)
    else:
        print('Model is already trained')

        model = torch.load('model.pth')
        test_model(model, device, test_loader, -1)


if __name__ == "__main__":
    main()
