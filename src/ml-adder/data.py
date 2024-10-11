import torch

import torch.utils
from torch.utils.data import DataLoader, TensorDataset


# a sever downgrade in model performance is observed for values >20
MAX_TENSOR_VAL = 10
# MAX_TENSOR_VAL = 25
# Correct predictions: 53/2500
# Model accuracy: 2.120 %
# MAX_TENSOR_VAL = 20
# Correct predictions: 2500/2500
# Model accuracy: 100.000 %


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


def generate_input_data(n_samples: int, max_tensor_val: int):
    t = generate_tensor(n_samples, highs=max_tensor_val)
    st = sum_tensor_by_columns(t)
    return torch.cat((t, st), dim=-1)


def generate_features_and_labels(t: torch.tensor):
    valid_tensor_shape(t)
    features = t[:, :, :2]
    labels = t[:, :, 2].view(t.shape[0], -1, 1)
    return features, labels


def generate_train_data(n_samples: int, batch_size: int) -> DataLoader:
    train_data = generate_input_data(n_samples, MAX_TENSOR_VAL)
    train_features, train_labels = generate_features_and_labels(train_data)

    train_set = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    return train_loader


def generate_test_data(n_samples) -> DataLoader:
    test_data = generate_input_data(n_samples, 69)
    test_features, test_labels = generate_features_and_labels(test_data)

    test_set = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_loader
