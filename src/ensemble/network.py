import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import sys


# LeNet Model definition
class Net(nn.Module):
    def __init__(self, features: int, labels: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 10)
        self.fc6 = nn.Linear(10, 100)
        self.fc7 = nn.Linear(100, 10)
        self.fc8 = nn.Linear(10, labels)
        self.bn8 = nn.BatchNorm1d(labels)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.bn8(self.fc8(x))
        x = torch.softmax(x, dim=1)
        return x


def gen_input_and_labels(device: torch.device, batch_size: int, features: int, label_size: int) -> tuple:
    """
    Generate a batch of input data and corresponding output labels.

    Args:
        device (torch.device): The device to which the tensors will be moved.
        batch_size (int): Number of samples in the batch.
        features (int): Number of input features per sample.
        label_size (int): Number of output classes for the labels.

    Returns:
        tuple: A tuple containing:
            - inputs (torch.Tensor): A tensor of shape (batch_size, features) with random inputs.
            - labels (torch.Tensor): A tensor of shape (batch_size,) with random class labels.
    """
    inputs = torch.randn(batch_size, features).to(device)
    labels = torch.randint(0, label_size, (batch_size,)).to(device)
    return inputs, labels


def gen_model(features: int, labels: int, aarch: str):
    print(f'Device: < {aarch} >\n')
    model = Net(features, labels)
    device = torch.device(aarch)
    model.to(device=device)
    return model, device


def train(l_optimizer: optim, n_epochs: int = 100) -> None:
    print(f'Training using optimizer: < {l_optimizer} >\n')

    model.train()
    for epoch in range(n_epochs):  # Example: 10 epochs
        l_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        l_optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    print(f'#############################################')


if __name__ == "__main__":

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])),
        batch_size=1, shuffle=True)

    if len(sys.argv) > 1:
        aarch = sys.argv[1]
    else:
        aarch = "cpu"
    n_labels = 12
    n_features = 900
    batch_size = 1024
    epochs = 1000
    model, device = gen_model(n_features, n_labels, aarch)
    inputs, labels = gen_input_and_labels(
        device, batch_size, n_features, n_labels)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_adam = optim.Adam(model.parameters())

    train(optimizer_adam, epochs)
    train(optimizer_sgd, epochs)
