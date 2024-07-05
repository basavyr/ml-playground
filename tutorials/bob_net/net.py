# source: https://www.youtube.com/watch?v=JRlyw6LO5qo&list=PL_LG-VOo0eUykJPa7r8zyJq7pq9MKJNXQ

import os
import torch

import copy

import torch.nn as nn

import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import torchvision.transforms as transforms

import torch.optim as optim


import matplotlib.pyplot as plt

import sys
from tqdm import trange


class Net(nn.Module):
    def __init__(self, input_size: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)
        self.fc23 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.tensor):
        x = x.view(x.size(0), -1)  # flatten the input
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc21(x)
        x = torch.relu(x)
        # x = self.fc22(x)
        # x = torch.relu(x)
        # x = self.fc23(x)
        # x = torch.relu(x)
        logits = self.fc3(x)
        return logits


def train_model(model: nn.Module, device: torch.device, data_loader: DataLoader, num_epochs: int) -> None:
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []
    epoch_accs = []
    once = False
    for epoch in (t := trange(num_epochs)):
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
        epoch_loss = round(float(running_loss / len(data_loader)), 3)
        epoch_acc = round(float(correct_labels_per_epoch/total_labels), 3)
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        t.set_description(f"loss {epoch_loss} acc {epoch_acc}")

    plt.plot(epoch_losses)
    plt.plot(epoch_accs)
    plt.savefig("loss_acc.pdf", dpi=300, bbox_inches="tight")
    # for idx, data in enumerate(zip(epoch_losses, epoch_accs)):
    #     loss, acc = data
    #     print(f'Epoch: {idx + 1}')
    #     print(f'Loss: {loss}')
    #     print(f'Acc: {acc}\n')

    torch.save(model, 'model.pth')


def test_model(model: nn.Module, device: torch.device, data_loader: DataLoader, iter_stop: int = 10) -> None:
    loss_fn = nn.CrossEntropyLoss()

    model.eval()

    best_acc = -torch.inf
    best_loss = torch.inf
    best_weights = None
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
            # acc = (preds == Y).float().sum()/(preds == Y).numel()

            loss = loss_fn(y_pred, Y)
            acc = (preds == Y).float().mean()
            if acc >= best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())

            if loss.item() <= best_loss:
                best_loss = loss.item()
            if idx == iter_stop and iter_stop != -1:
                break

    print(f'Best accuracy: {best_acc}')
    print(f'Best loss: {best_loss}')
    model.load_state_dict(best_weights)

    print(model.state_dict())


def main(device_as_input_string: str):
    height = 28
    width = 28
    n_channels = 1
    input_size = height*width
    n_samples = 1
    batch_size = 128
    num_epochs = 20

    device = torch.device(device_as_input_string)

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
        train_model(model, device, train_loader, num_epochs)
    else:
        print('Model is already trained')

        model = torch.load('model.pth')
        test_model(model, device, test_loader, -1)


if __name__ == "__main__":
    print('Run the script as such:')
    print('$ python3 net.py <device>')
    print('$ python3 net.py <device> clean')
    if len(sys.argv) > 1:
        device_as_input_string = sys.argv[1]
    if len(sys.argv) == 3:
        os.remove("model.pth")
    main(device_as_input_string)
