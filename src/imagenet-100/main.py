import torch
from torch import nn as nn
from torchvision.transforms import Compose, ToTensor, Resize

from PIL import Image
from io import BytesIO
import pandas as pd


import pathlib
from tqdm import tqdm
from typing import List
import os


def parquet_to_torch(parquet_file: pathlib.PurePath):
    data = pd.read_parquet(parquet_file)
    images = []
    labels = []
    # start = time.monotonic_ns()
    for img, label in tqdm(data.iloc(0), desc='Processing parquet file', total=len(data), unit='images', dynamic_ncols=True):
        img_bytes = img['bytes']
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        image_tensor = Compose([Resize((224, 224)), ToTensor()])(image)
        label = torch.tensor(label, dtype=torch.long)
        images.append(image_tensor)
        labels.append(label)
    # finish = (time.monotonic_ns() - start)/1e9
    # print(f'Processed {len(data)} samples in {finish:.3f}s')
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def get_parquet_files(root_dir: str, n_files: int, split: str) -> List[pathlib.Path]:
    assert split in ["validation", "train"], f"Incorrect split type: {split}"

    max_parquet = f'000{n_files}' if n_files >= 10 else f'0000{n_files}'
    parquet_files = []
    for idx in range(n_files):
        pq_idx = f'000{idx}' if idx >= 10 else f'0000{idx}'
        parquet_path = pathlib.Path(
            f'{root_dir}/{split}-{pq_idx}-of-{max_parquet}.parquet')
        parquet_files.append(parquet_path)
    return parquet_files


def convert_parquet_data_to_torch(root_dir, n_files: int, split: str):
    assert split in ["validation", "train"], f"Incorrect split type: {split}"

    pq_files = get_parquet_files(root_dir, n_files, split)
    for idx, pq_file in enumerate(pq_files):
        images, labels = parquet_to_torch(pq_file)
        pt_data = {"images": images, "labels": labels}
        pt_idx = f'{idx+1}' if idx+1 >= 10 else f'0{idx+1}'
        data_file = f'data/imagenet100_{split}-{pt_idx}.pt'
        torch.save(pt_data, data_file)
        print(f'{split} data {idx+1}/{n_files} -> {data_file}')


def main():
    # initial data taken from HF: https://huggingface.co/datasets/basavyr/imagenet-100
    PARQUET_PATH = os.getenv("PARQUET_PATH", None)
    assert PARQUET_PATH is not None, "Environment variable < PARQUET_PATH > is not set"

    os.makedirs("data", exist_ok=True)
    convert_parquet_data_to_torch(PARQUET_PATH, 17, "train")
    convert_parquet_data_to_torch(PARQUET_PATH, 1, "validation")


if __name__ == "__main__":
    main()
