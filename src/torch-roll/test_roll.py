import torch
import torch.nn as nn


def add_noise(t: torch.tensor, strength: float = 0.01):
    t_noise = strength*torch.randn_like(t)
    t += t_noise
    return t


def main():
    device = torch.device("mps")

    torch.manual_seed(1137)

    dims = (3, 2)
    alpha = 0.001
    t1 = torch.arange(12, dtype=torch.float)
    # t1 = add_noise(t1, 0.01)

    roll = torch.roll(t1, 2)

    print(roll, dims)


if __name__ == "__main__":
    main()
