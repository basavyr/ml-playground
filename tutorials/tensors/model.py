
from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
import os

import loaders


class ImprovedModel(nn.Module):
    def __init__(self, n_samples, batch_size: int):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        logits = self.fc5(x)
        return logits


class Model(nn.Module):
    def __init__(self, n_samples, batch_size: int):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 5)
        self.fc6 = nn.Linear(5, 1)

    def forward(self, x: torch.tensor):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        x = self.fc5(x)
        x = torch.relu(x)
        logits = self.fc6(x)
        return logits


def train_model(model: nn.Module, device, data_loader, num_epochs):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(X).squeeze()

            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader):.4f}")

    # save the model locally
    torch.save(model.state_dict(), "model.pth")


def test_model(model: nn.Module, device, test_data):
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(test_data).squeeze()
        y_pred_prob = torch.sigmoid(y_pred).item()
        y_pred_label = 1 if y_pred_prob > 0.5 else 0

    print(f"Predicted logit: {y_pred.item()}")
    print(f"Predicted probability: {y_pred_prob}")
    print(f"Predicted label: {y_pred_label}")


def generate_test_tensor(M_test: int, device) -> torch.tensor:
    # Create a single tensor of ones
    X_ones_test = torch.ones((1, 1, 28, 28)).to(device)
    X_rand_test = torch.rand((1, 1, 28, 28)).to(device)

    rand_num = np.random.randint(0, 2)
    if rand_num > 0:
        print('Test tensor is a <ones>')
        test_tensor = X_ones_test
    else:
        print('Test tensor is a <rand>')
        test_tensor = X_rand_test

    return test_tensor


def main():
    # params
    M = 1024
    M_test = 1
    batch_size = 8
    num_epochs = 5

    # device
    device = torch.device("mps")

    # load model to device
    model = Model(M, batch_size)

    # create data loader
    mnist_like = loaders.MNISTLike(M, batch_size)
    data_loader = mnist_like.loader()

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        print("Model loaded from model.pth")
    else:
        train_model(model, device, data_loader, num_epochs)

    test_tensor = generate_test_tensor(M_test, device)
    test_model(model, device, test_tensor)


if __name__ == "__main__":
    main()
