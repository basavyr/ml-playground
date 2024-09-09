import torch

import torch.utils
from torch.utils.data import Dataset, DataLoader, TensorDataset

import model as m

import torch.nn as nn
import torch.optim as optim


def valid_tensor_shape(t: torch.Tensor):
    if len(t.shape) != 3:
        print(
            'Invalid tensor shape. The tensor must be of size [batch_size, N, M]')
        print(f'Actual tensor shape: {t.shape}')
        exit(1)


def sum_tensor_by_columns(t: torch.Tensor):
    valid_tensor_shape(t)
    a = t[:, :, 0]
    b = t[:, :, 1]
    return (a + b).view(t.shape[0], -1, 1)


def generate_tensor(n_samples: int, highs: int, lows: int = 0):
    dim_t = (n_samples, 1, 2)
    t = torch.randint(lows, highs, dim_t, dtype=torch.float)
    return t


def generate_input_data(n_samples: int):
    t = generate_tensor(n_samples, 10)
    st = sum_tensor_by_columns(t)
    return torch.cat((t, st), dim=-1)


def generate_features_and_labels(t: torch.tensor):
    valid_tensor_shape(t)
    features = t[:, :, :2]
    labels = t[:, :, 2].view(t.shape[0], -1, 1)
    return features, labels


def train(model: nn.Module, loss_fn, dataloader: DataLoader):
    optimizer = optim.SGD(model.parameters())

    model.train()
    train_loss = []
    for idx, batch in enumerate(dataloader):
        x, y_true = batch
        y = model(x)

        loss = loss_fn(y, y_true)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

    torch.save(model, "model.pth")


def eval(model_path: str, loss_fn, eval_dataloader: DataLoader):
    model = torch.load(model_path)

    model.eval()

    eval_losses = []

    total_predictions = []
    correct_predictions = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            x, y_true = batch

            y = model(x)

            loss = loss_fn(y, y_true)
            eval_losses.append(loss.item())

            prediction = torch.allclose(y, y_true, atol=0.1)
            total_predictions.append(prediction)
            if prediction is True:
                correct_predictions.append(prediction)

    print(
        f'Correct predictions: {len(correct_predictions)}/{len(total_predictions)}')
    print(
        f'Model accuracy: {len(correct_predictions)/len(total_predictions)*100:.3f} %')


if __name__ == "__main__":
    batch_size = 32
    n_samples = 100000

    train_data = generate_input_data(n_samples)
    train_features, train_labels = generate_features_and_labels(train_data)

    test_data = generate_input_data(1000)
    test_features, test_labels = generate_features_and_labels(test_data)

    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(
        [[60, 9]], dtype=torch.float), torch.tensor([[5]], dtype=torch.float))
    test_loader = DataLoader(test_set)

    loss_fn = nn.MSELoss()
    model = m.Adnet()

    train(model, loss_fn, train_loader)

    eval("model.pth", loss_fn, test_loader)
