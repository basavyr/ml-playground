from torch.utils.data import Dataset
import torch

import glob
import os

from typing import Callable, List

import numpy as np


class Cifar5m(Dataset):
    DEFAULT_IMG_SIZE_NP = (32, 32, 3)
    DEFAULT_IMG_SIZE_TORCH = torch.Size([3, 32, 32])  # ToPILImage
    N_TRAIN_SAMPLES: int = 5002240
    N_TEST_SAMPLES: int = 1000448

    def __init__(self, root: str | None, transform: Callable | None, train: bool, integrity_check: bool = False):
        self.integrity = None
        self.train = train
        self.root = root
        self.transform = transform
        self.x = []
        self.y_true = []

        # get the .npz files
        npz_files = self.is_cifar5m_available(root)
        npz_files = npz_files[:5] if train else npz_files[5:]

        for npz_file in npz_files:
            npz_data = np.load(npz_file)
            _X = npz_data['X']
            _Y = npz_data['Y']
            self.x.append(_X)
            self.y_true.extend(_Y)
        self.x = np.vstack(self.x)
        if integrity_check:
            self.integrity = self._check_integrity()

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y_true = self.y_true[idx]

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).float()
        y_true = torch.tensor(y_true, dtype=torch.long)
        return x, y_true

    def __len__(self):
        return len(self.x)

    def _check_integrity(self):
        total_samples = len(self.x)
        img_dim = self.x[0].shape
        if self.train:
            assert total_samples == self.N_TRAIN_SAMPLES, f"Invalid number of training samples. Expected: {self.N_TRAIN_SAMPLES} / Real: {total_samples}"
        else:
            assert total_samples == self.N_TEST_SAMPLES, f"Invalid number of test samples. Expected: {self.N_TRAIN_SAMPLES} / Real: {total_samples}"
        assert img_dim == self.DEFAULT_IMG_SIZE_NP, f"Invalid tensor shape. Expected: {self.DEFAULT_IMG_SIZE_NP} / Real: {img_dim}"
        return 1

    @staticmethod
    def is_cifar5m_available(root: str | None) -> List[str]:
        assert root is not None, f"Environment variable <CIFAR_5M_FULL_PATH> is not set."
        assert os.path.exists(
            root), f"Invalid CIFAR-5M root directory provided: root={root}"
        npz_files = []
        npz_files = glob.glob(
            "*part[0-9].npz", root_dir=root, recursive=True)
        npz_files = sorted(npz_files, key=lambda x: x[-5])
        npz_files = [f'{root}/{_file}' for _file in npz_files]
        assert len(
            npz_files) == 6, "The CIFAR-5M root directory is incomplete or corrupt."
        return npz_files
