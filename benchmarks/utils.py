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



def get_flops(
    model: torch.nn.Module,
    device: Device,
    custom_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    is_custom_transformer: bool = False,
    with_backward: bool = False
) -> float:
    model.to(device)
    is_training = model.training
    model.eval()

    if is_custom_transformer:
        tgt, mem, tgt_mask = custom_input
        tgt, mem, tgt_mask = tgt.to(device), mem.to(
            device), tgt_mask.to(device)
    else:
        tgt, _, _ = custom_input
        tgt = tgt.to(device)
        mem = None
        tgt_mask = None

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        if is_custom_transformer:
            if with_backward:
                model(tgt, mem, tgt_mask).sum().backward()
            else:
                model(tgt, mem, tgt_mask)
        else:
            if with_backward:
                model(tgt).sum().backward()
            else:
                model(tgt)
    if is_training:
        model.train()
    return flop_counter.get_total_flops()
