import torch

from torch.types import Device


def get_optimal_device(force_cuda: bool = False):
    if force_cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError(
            "Cannot use CUDA device! Make sure CUDA is available on the system.")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
