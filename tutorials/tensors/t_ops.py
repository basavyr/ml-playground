import torch
from torch import nn
import torch.functional as F
from torch.linalg import inv

import math
import numpy as np


class TensorEnvironments:
    def __init__(self, param):
        self.param = param

    def generate_numpy_array(self, M, N, H, W) -> np.ndarray:
        return np.random.rand(M, N, H, W)

    def test_tensors(self):
        t0 = torch.tensor([1, 1])
        t = torch.rand([2, 2])
        t_np = torch.from_numpy(self.generate_numpy_array(3, 3, 3, 3))
        # create a diagonal tensor (here is just a 2D matrix) that will have on the diagonal the initial fixed-length tensor (i.e., here it is denoted by `t0`)
        t2 = torch.diag(t0, diagonal=0)
        print(f't -> {t}')
        print(f't_np -> {t_np}')
        print(f't2 -> {t2}')
        print(f't + t2 -> {t+t2}')
        print(f't + t2 -> {t*t2}')
        # print(f't / t2 -> {t.mm(inv(t2))}')


def init_params():
    M, N, H, W = (100, 3, 28, 28)
    return M, N, H, W


def generate_tensor(device, M, N, H, W):
    """
    - generate a tensor that will be also allocated on a specific device
    """
    return torch.rand([M, N, H, W]).to(device)


def generate_tensor_np(device, M, N, H, W):
    t_np = np.random.rand(M, N, H, W)
    return torch.from_numpy(t_np).to(device)
