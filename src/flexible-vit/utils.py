import torch


def get_optimal_device() -> torch.types.Device:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f'Detected {num_gpus} CUDA devices. Using cuda:0 by default')
        return torch.device("cuda:0")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
