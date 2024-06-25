import torch
from torch import nn
import torch.functional as F
from torch.linalg import inv

import math
import numpy as np


def init_params():
    M, N, H, W = (100, 3, 28, 28)
    return M, N, H, W


def generate_tensor(M, N, H, W):
    t0 = torch.tensor([1, 1])
    t = torch.rand([2, 2])
    t_np = torch.from_numpy(generate_numpy_array(3))
    # create a diagonal tensor (here is just a 2D matrix) that will have on the diagonal the initial fixed-length tensor (i.e., here it is denoted by `t0`)
    t2 = torch.diag(t0, diagonal=0)
    print(f't -> {t}')
    print(f't_np -> {t_np}')
    print(f't2 -> {t2}')
    print(f't + t2 -> {t+t2}')
    print(f't + t2 -> {t*t2}')
    # print(f't / t2 -> {t.mm(inv(t2))}')


def generate_numpy_array(size: int) -> np.ndarray:
    return np.random.rand(size, size)


def main():
    M, N, H, W = init_params()
    generate_tensor(M, N, H, W)


if __name__ == "__main__":
    main()
