import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

import torch.optim as optim

import torch.nn.functional as F

import os

import plotter as plt

MODEL_FILE_PATH = "model.pth"


class Model0(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, input_size)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)  # Reshape back to the original image size
        return x


class Model(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, input_size)

    def forward(self, x):
        # alternative
        # x = x.view(-1, 28*28)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.log_softmax(x, dim=1)
        x = x.view(-1, 1, 28, 28)  # Reshape back to the original image size

        return x


def print_params(model, epoch, batch_id):
    print(f"Epoch {epoch+1}, Batch {batch_id+1}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Grad: {param.grad}")


def data_loader(M: int, batch_size: int) -> DataLoader:
    # alternative using cat
    # features = torch.cat([torch.rand([1, 1, 28, 28]) for _ in range(M)])
    features = torch.stack([torch.rand([1, 28, 28]) for _ in range(M)])
    labels = torch.randint(10, (M,))
    tensor_data = TensorDataset(features, labels)
    train_dataloader = DataLoader(
        tensor_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def synthetic_linear_data(M: int, batch_size: int) -> DataLoader:
    """
    - generates a loader containing linear mapping y=ax+b
    """
    X = torch.rand((M, 1, 28, 28))
    labels = 2 * X+1
    tensor_data = TensorDataset(X, labels)
    train_data_loader = DataLoader(
        tensor_data, batch_size=batch_size, shuffle=True)
    return train_data_loader


def train_model(model: nn.Module, device: torch.device, num_epochs: int, data_loader: DataLoader):
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    model.train()

    print_interval = 300

    for epoch in range(num_epochs):
        print(f'Training on epoch {epoch+1}')

        running_loss = 0.0
        for batch_id, mini_batch in enumerate(data_loader):

            X, y = mini_batch
            X, y = X.to(device), y.to(device)

            # set the gradients back to zero
            model.zero_grad()

            # pass the input through the NN -> obtain the output
            y_pred = model(X)

            # calculate the loss function for the current batch
            loss = loss_fn(y_pred, y)

            # perform back propagation
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if batch_id % print_interval == print_interval-1:
                average_loss = running_loss / print_interval
                print(f'[{epoch + 1}, {batch_id + 1:5d}] loss: {average_loss:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), "model.pth")


def logical_mse(t1: torch.tensor, t2: torch.tensor):
    """
    MSE loss
    Logical implementation for the loss function
    Source: https://www.kaggle.com/code/basavyr/pytorch-basics-linear-regression-from-scratch/edit
    """
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


def test_model(model: nn.Module, device: torch.device, data_loader: DataLoader):

    model.eval()

    loss_mean = nn.MSELoss()  # Sum the squared errors
    loss_sum = nn.MSELoss(reduction='sum')  # Sum the squared errors

    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    total_samples = 0
    num_batches = 0

    once = True

    with torch.no_grad():
        for _, mini_batch in enumerate(data_loader):
            X, y = mini_batch
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss_1 = loss_mean(y_pred, y)
            loss_2 = loss_sum(y_pred, y)
            loss_3 = logical_mse(y_pred, y)

            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()
            total_loss_3 += loss_3.item()
            total_samples += X.numel()
            num_batches += 1
            if once:
                plt.plot_tensors(X, y, y_pred)
                once = False

    loss_mean = total_loss_1/num_batches
    loss_sum = total_loss_2/total_samples
    loss_logical = total_loss_2/total_samples

    return loss_mean, loss_sum, loss_logical


def main():
    M_train = 60000
    M_test = 10000
    batch_size = 64
    num_epochs = 50
    train_dataloader = synthetic_linear_data(M_train, batch_size)
    test_dataloader = synthetic_linear_data(M_test, batch_size)

    device = torch.device("mps")

    # generate a local tensor that can be feed into the network
    _t = torch.rand((28, 28), device=device).view(-1,
                                                  28*28)  # needs to be flattened out
    model = Model(28*28).to(device)

    if not os.path.exists(MODEL_FILE_PATH):
        train_model(model, device, num_epochs, train_dataloader)
    else:
        model.load_state_dict(torch.load(MODEL_FILE_PATH))
        loss_mean, loss_sum, loss_logical = test_model(
            model, device, test_dataloader)
        print(f'Using mean ->{loss_mean}')
        print(f'Using sum ->{loss_sum}')
        print(f'Using logical mse ->{loss_logical}')


if __name__ == "__main__":
    main()
