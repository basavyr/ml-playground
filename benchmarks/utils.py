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


def get_flops_approx(
    d_model: int,
    n_layers: int,
    n_heads: int,
    sequence_length: int,
    vocab_size: int,
    num_samples: int,
    num_epochs: int,
    d_ffn: int = 2048,
) -> float:
    # decoder
    """
    in terms of param count, the attentions are:
    3 projections (QKV) with [ sequence_length * d_model^2 ] -> 3 * sequence_length * d_model^2
    1 output projection (O) with [ sequence_length * d_model^2] -> 1 * sequence_length * d_model^2
    total for param count: 4 * sequence_length * d_model^2

    in terms of flops:
    2 * p
    """
    flops_per_attention = 8 * sequence_length * \
        (d_model**2) + 4 * (sequence_length**2) * \
        d_model + 3 * n_heads * (sequence_length**2)
    flops_per_ffn = 2 * (2 * sequence_length * d_model * d_ffn)

    # lm head
    flops_per_lm_head = 2 * sequence_length * \
        d_model*vocab_size  # the 2*M*N*K rule

    # per 1 forward pass
    flops_model_1fp = n_layers * \
        (flops_per_attention + flops_per_ffn)+flops_per_lm_head

    # per data and backward (1xfp + 2xbp = 3x)
    flops_iter = 3 * num_samples * flops_model_1fp

    return num_epochs*flops_iter


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
