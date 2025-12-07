import torch
from torch.types import Device
from torch.utils.flop_counter import FlopCounterMode
from typing import Tuple

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()


def get_optimal_device(force_cuda: bool = False) -> Device:
    if force_cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError(
            "Cannot use CUDA device! Make sure CUDA is available on the system.")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()
