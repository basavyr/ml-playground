import torch


from torch.nn import functional as F


relu = F.relu

torch.manual_seed(1137)


def generate_x_and_y(n: int, batch_size: int, highs: int, lows: int = 0):
    x_y = torch.randint(lows, highs, (batch_size, n, 2))
    return x_y


if __name__ == "__main__":
    N = 512
    batch_size = 32
    x_y = generate_x_and_y(N, batch_size, 100)
