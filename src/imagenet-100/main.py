from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch

import pandas as pd
from PIL import Image
from io import BytesIO

import time
import pathlib
import os
import sys
from tqdm import tqdm
from typing import List, Callable, Optional


PT_DATA_DIR = "data"


def parquet_to_torch(parquet_file: pathlib.PurePath):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    try:
        data = pd.read_parquet(parquet_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file {parquet_file}: {e}")

    if len(data) == 0:
        raise ValueError(f"Empty parquet file: {parquet_file}")

    images = []
    labels = []
    transform = Compose([Resize((224, 224)), ToTensor()])

    for row in tqdm(data.itertuples(index=False), desc='Processing parquet file', total=len(data), unit='images', dynamic_ncols=True):
        try:
            img_bytes = row[0]['bytes']
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            image_tensor = transform(image)
            label = torch.tensor(row[1], dtype=torch.long)
            images.append(image_tensor)
            labels.append(label)
        except Exception as e:
            print(
                f"Warning: Skipping corrupted image at index {len(images)}: {e}")
            continue

    if not images:
        raise ValueError(f"No valid images found in {parquet_file}")

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


def convert_parquet_data_to_torch(root_dir: str, n_files: int, split: str):
    if split not in ["validation", "train"]:
        raise ValueError(f"Incorrect split type: {split}")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    os.makedirs(PT_DATA_DIR, exist_ok=True)
    pq_files = get_parquet_files(root_dir, n_files, split)

    for idx, pq_file in enumerate(pq_files):
        if not os.path.exists(pq_file):
            print(f"Warning: Parquet file not found, skipping: {pq_file}")
            continue

        try:
            images, labels = parquet_to_torch(pq_file)
            pt_data = {"images": images, "labels": labels}
            pt_idx = f'{idx+1}' if idx+1 >= 10 else f'0{idx+1}'
            data_file = f'{PT_DATA_DIR}/imagenet100_{split}-{pt_idx}.pt'
            torch.save(pt_data, data_file)
            print(f'{split} data {idx+1}/{n_files} -> {data_file}')
        except Exception as e:
            print(f"Error processing {pq_file}: {e}")
            continue


class Imagenet100(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[Callable] = None, max_files: int = -1):
        assert max_files != 0, "Cannot have max_files=0 to retrieve."
        self.root_dir = root_dir
        self.transform = transform

        pt_files = [f for f in os.listdir(root_dir)
                    if (("train" in f) if train else ("validation" in f)) and f.endswith('.pt')]
        pt_files.sort()
        if max_files < 0:
            self.pt_files = pt_files
        else:
            self.pt_files = pt_files[:max_files]

        self.data = []
        for pt_file in self.pt_files:
            file_path = os.path.join(root_dir, pt_file)
            data = torch.load(file_path, map_location='cpu')
            for i in range(len(data['images'])):
                self.data.append((data['images'][i], data['labels'][i]))

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class Model(torch.nn.Module):
    def __init__(self, use_pretrained_resnet: bool = False):
        super(Model, self).__init__()
        self.num_classes = 100
        self.use_pretrained_resnet = use_pretrained_resnet
        if self.use_pretrained_resnet:
            from torchvision.models import resnet18
            from torchvision.models.resnet import ResNet18_Weights
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.fc = torch.nn.Linear(512, self.num_classes)
        else:
            self.conv1 = torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.conv2 = torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1)
            self.bn2 = torch.nn.BatchNorm2d(128)
            self.relu = torch.nn.ReLU()
            self.fc = torch.nn.Linear(128*220*220, self.num_classes)

    def forward(self, x: torch.Tensor):
        if self.use_pretrained_resnet:
            x = self.model(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        return x


def main():
    PARQUET_PATH = os.getenv("PARQUET_PATH", None)
    if PARQUET_PATH is None:
        raise ValueError("Environment variable PARQUET_PATH is not set")

    os.makedirs(PT_DATA_DIR, exist_ok=True)

    train_dataset = Imagenet100(PT_DATA_DIR, train=True, max_files=2)
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(f"Train dataset size: {len(train_dataset)}")

    # test_dataset = Imagenet100(PT_DATA_DIR, train=False)
    # testloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    # print(f"Test dataset size: {len(test_dataset)}")

    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("mps")
    model = Model(use_pretrained_resnet=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    train_loss = 0
    total_tensor_size_mb = 0
    start = time.monotonic_ns()
    for x, y_true in tqdm(trainloader, desc='Training data'):
        x, y_true = x.to(device), y_true.to(device)
        optimizer.zero_grad()

        total_tensor_size_mb += float(x.numel()*x.element_size())/(1024**2)
        y = model(x)
        loss = loss_fn(y, y_true)
        train_loss += loss.item()*x.shape[0]

        loss.backward()
        optimizer.step()
    train_loss /= len(trainloader.dataset)

    duration = float((time.monotonic_ns() - start)/1e9)
    bandwidth = round(total_tensor_size_mb/duration, 4)
    print(f'Total bandwidth: {bandwidth} MBps')
    print(f'Train Loss: {train_loss}')


if __name__ == "__main__":
    main()
