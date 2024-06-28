import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x


def data_loader(M: int, batch_size: int) -> DataLoader:
    # alternative using cat
    # features = torch.cat([torch.rand([1, 1, 28, 28]) for _ in range(M)])
    features = torch.stack([torch.rand([1, 28, 28]) for _ in range(M)])
    labels = torch.randint(10, (M,))
    tensor_data = TensorDataset(features, labels)
    train_dataloader = DataLoader(
        tensor_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def main():
    M_test = 60000
    M_train = 10000
    batch_size = 64
    train_dataloader = data_loader(M_train, batch_size)
    test_dataloader = data_loader(M_test, batch_size)

    device = torch.device("mps")

    model = Model(28*28).to(device)

    # generate a local tensor that can be feed into the network
    _t = torch.rand((28, 28), device=device).view(-1,
                                                  28*28)  # needs to be flattened out

    for batch_id, mini_batch in enumerate(train_dataloader):
        X, y = mini_batch
        X, y = X.to(device), y.to(device)
        break


if __name__ == "__main__":
    main()
