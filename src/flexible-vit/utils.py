import torch
import random
import numpy as np


def get_optimal_device() -> torch.types.Device:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f'Detected {num_gpus} CUDA devices. Using cuda:0 by default')
        return torch.device("cuda:0")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_deterministic_behavior(seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
