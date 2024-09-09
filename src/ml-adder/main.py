import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset

import model as m

import torch.nn as nn

torch.manual_seed(42)


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
    features = data[:, :, :2]
    labels = data[:, :, 2].view(data.shape[0], -1, 1)
    return features, labels


if __name__ == "__main__":
    batch_size = 128
    n_samples = 10000
    data = generate_input_data(n_samples)

    features, labels = generate_features_and_labels(data)

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
