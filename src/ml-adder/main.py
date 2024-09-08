import torch


from torch.nn import functional as F


relu = F.relu

torch.manual_seed(42)


def generate_x_and_y(n: int, batch_size: int, highs: int, lows: int = 0):
    x_y = torch.randint(lows, highs, (batch_size, n, 2))
    return x_y


def add_x_and_y(x_y: torch.Tensor):
    x_values = x_y[:, :, 0]
    y_values = x_y[:, :, 1]
    return x_values+y_values


if __name__ == "__main__":
    N = 512
    batch_size = 32
    x_y = generate_x_and_y(N, batch_size, 100)

    sum_x_y = add_x_and_y(x_y)

    print(x_y.shape)
    print(sum_x_y.shape)

    print(x_y[0])
    print(sum_x_y[0])