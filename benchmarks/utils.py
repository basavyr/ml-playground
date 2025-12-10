import torch
from torch.types import Device
from torch.utils.flop_counter import FlopCounterMode

import urllib
import shutil
import zipfile
from typing import Tuple
import os
import sys
from datetime import datetime


def generate_log_file(model_type: str) -> Tuple[str, str]:
    """
    Returns a tuple with a name for a log file and the timestamp at which the log name has been generated
    """
    os.makedirs("./logs", exist_ok=True)
    platform = os.uname().nodename
    generated_at = datetime.now().isoformat(
        sep="-", timespec='seconds').replace(":", "")
    log_name = f'./logs/M{model_type}_{generated_at}_{platform}.log'
    return log_name, generated_at


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


def download_and_prepare_tiny_imagenet(root_dir: str = "data") -> str:
    """
    - Utility for downloading the Tiny ImageNet 200 dataset as a zip file. It automatically extracts the archive and then moves original images into a dedicated directory with per-class label sub-folders.

    .. note::
    The method requires only a path where the entire process will be executed. As a result, a `./root_dir/tiny-imagenet-200` directory will be created. This path will be returned by default.
    """
    base_name = "tiny-imagenet-200"
    url = f"https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root_dir, "tiny-imagenet-200.zip")

    # Move train and val images into a single ImageFolder-style directory
    tiny_imagenet_dir = os.path.join(root_dir, base_name)
    os.makedirs(tiny_imagenet_dir, exist_ok=True)

    if not os.path.isfile(zip_path):
        print(f"Downloading Tiny ImageNet to {zip_path} ...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(
            f"Skipping download since .zip file already exists -> {zip_path}")

    extract_path = f"{root_dir}/extracted"
    os.makedirs(extract_path, exist_ok=True)

    try:
        zip_content = os.listdir(f'{extract_path}/{base_name}')
    except FileNotFoundError or FileExistsError:
        zip_content = None
    unzip = True
    if os.path.isdir(extract_path) and zip_content is not None and ('test' in zip_content and 'train' in zip_content and 'val' in zip_content):
        unzip = False

    if unzip:
        print(f"Extracting to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        print(
            f'Skipping zip extraction since files already exist -> {extract_path}/{base_name}')

    if not validate_tiny_imagenet_image_folder(tiny_imagenet_dir):
        # Move train images
        train_dir = os.path.join(f'{extract_path}/{base_name}', "train")
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name, "images")
            target_class_dir = os.path.join(tiny_imagenet_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            for img in os.listdir(class_dir):
                shutil.copy(os.path.join(class_dir, img),
                            os.path.join(target_class_dir, img))
        # Move val images
        val_dir = os.path.join(f'{extract_path}/{base_name}', "val")
        val_annotations = os.path.join(val_dir, "val_annotations.txt")
        val_img_dir = os.path.join(val_dir, "images")
        with open(val_annotations, 'r') as f:
            for line in f:
                img, class_name, *_ = line.strip().split('\t')
                target_class_dir = os.path.join(tiny_imagenet_dir, class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                shutil.copy(os.path.join(val_img_dir, img),
                            os.path.join(target_class_dir, img))
    else:
        print(
            f'The path < {tiny_imagenet_dir} > is already a valid ImageFolder.\nSkipping .JPEG move & copy.')
    print(f"Tiny ImageNet prepared at {tiny_imagenet_dir}")
    return tiny_imagenet_dir


def get_dir_size(path):
    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_dir_size(entry.path)
    return total_size


def validate_tiny_imagenet_image_folder(image_folder_path: str) -> bool:
    """
    - Checks if the directory structure is valid and if all the samples are present
    """
    num_classes = 200
    minimum_required_disk_size = 201.0
    dir_size = get_dir_size(image_folder_path) / (1024**2)
    if len(os.listdir(image_folder_path)) == num_classes and dir_size >= minimum_required_disk_size:
        return True
    return False


def get_model_flops(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.types.Device, with_backward: bool = True):
    model.to(device)

    x, _ = next(iter(train_loader))
    x = x.to(device)
    B, C, H, W = x.shape
    print(f'B = {B} | C = {C} | H = {H} | W = {W} |')
    flop_counter = FlopCounterMode(display=True)
    with flop_counter:
        model(x)
        if with_backward:
            model(x).sum().backward()

    return flop_counter.get_total_flops()
