"""
ImageNet-1K PyTorch Dataset backed by Parquet shards.

Designed for the Hugging Face ILSVRC/imagenet-1k repository layout where
parquet files live under a ``data/`` directory with the naming convention::

    data/train-00000-of-00294.parquet
    data/train-00001-of-00294.parquet
    ...
    data/val-00000-of-00013.parquet
    ...

Each parquet file contains rows with two columns:
    - image : dict with key "bytes" holding raw JPEG data
    - label : int (0-999 for train/val, -1 for test)

Usage
-----
>>> from imagenet_dataset import ImageNetParquetDataset
>>> from torch.utils.data import DataLoader
>>>
>>> train_ds = ImageNetParquetDataset("./imagenet-1k/data", split="train")
>>> val_ds   = ImageNetParquetDataset("./imagenet-1k/data", split="val")
>>>
>>> train_loader = DataLoader(
...     train_ds, batch_size=256, shuffle=True,
...     num_workers=8, pin_memory=True,
... )
>>> for images, labels in train_loader:
...     # images: [B, 3, 224, 224]  labels: [B]
...     pass
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time
from collections import OrderedDict
from typing import Callable, Literal, Optional, Tuple

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Standard ImageNet normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet training augmentation."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def default_val_transform(
    image_size: int = 224, resize_size: int = 256
) -> transforms.Compose:
    """Standard ImageNet validation transform (resize + center-crop)."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# LRU cache for parquet shard tables
# ---------------------------------------------------------------------------
class _ShardLRUCache:
    """
    A simple least-recently-used cache that keeps at most *maxsize* parquet
    tables in memory.  Each table is a ``pyarrow.Table`` read from one shard
    file.  When the cache is full the oldest entry is evicted.

    Parameters
    ----------
    maxsize : int
        Maximum number of shards to keep in memory simultaneously.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self._maxsize = max(1, maxsize)
        self._cache: OrderedDict[str, pq.ParquetFile] = OrderedDict()

    def get(self, path: str) -> pq.ParquetFile:
        """Return the table for *path*, reading it from disk if necessary."""
        if path in self._cache:
            # Move to end (most recently used).
            self._cache.move_to_end(path)
            return self._cache[path]

        # Evict oldest if at capacity.
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        table = pq.ParquetFile(path)
        self._cache[path] = table
        return table


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------
class ImageNetParquetDataset(Dataset):
    """
    A memory-efficient PyTorch ``Dataset`` that reads ImageNet images from
    parquet shard files on disk.

    Only shard metadata (row counts) is loaded at init time.  Image bytes are
    decoded lazily in ``__getitem__`` with an LRU cache so that recently
    accessed shards stay in memory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the parquet files
        (e.g. ``"./imagenet-1k/data"``).
    split : ``"train"`` or ``"val"``
        Which split to load.
    transform : callable, optional
        A torchvision transform (or composition) applied to each PIL image.
        If *None* the standard ImageNet transform for the requested split is
        used (``RandomResizedCrop`` for train, ``Resize+CenterCrop`` for val).
    image_size : int
        Target spatial resolution (default 224).
    cache_shards : int
        Number of parquet shard files to keep in the LRU cache.  Each shard
        is ~500 MB, so set this according to available RAM.  With
        ``num_workers=N`` each worker gets its own cache, so total memory is
        roughly ``num_workers * cache_shards * 500 MB``.  Default is 2.
    """

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "valildation"] = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        cache_shards: int = 2,
    ) -> None:
        super().__init__()

        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got '{split}'")

        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.image_size = image_size

        # --- Discover shard files -------------------------------------------
        pattern = os.path.join(self.data_dir, f"{split}-*.parquet")
        self._shard_paths: list[str] = sorted(glob.glob(pattern))
        if not self._shard_paths:
            raise FileNotFoundError(
                f"No parquet shards found for split='{split}' in {self.data_dir!r}.\n"
                f"Expected files matching: {pattern}\n"
                f"Make sure you cloned the HuggingFace repo and the 'data/' "
                f"directory contains files like '{split}-00000-of-NNNNN.parquet'."
            )

        # --- Build cumulative row-count index --------------------------------
        # We only read parquet metadata here (no pixel data).
        self._shard_row_counts: list[int] = []
        self._cumulative_rows: list[int] = []  # exclusive upper bounds
        total = 0
        for path in self._shard_paths:
            pf = pq.ParquetFile(path)
            n = pf.metadata.num_rows
            self._shard_row_counts.append(n)
            total += n
            self._cumulative_rows.append(total)

        self._length = total

        # --- Transform -------------------------------------------------------
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = default_train_transform(image_size)
        else:
            self.transform = default_val_transform(image_size)

        # --- Shard cache (per-worker; see worker_init_fn) --------------------
        self._cache = _ShardLRUCache(maxsize=cache_shards)

        print(
            f"[ImageNetParquetDataset] split={split}  "
            f"shards={len(self._shard_paths)}  "
            f"images={self._length:,}  "
            f"cache_shards={cache_shards}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate(self, index: int) -> Tuple[int, int]:
        """Map a global index to ``(shard_index, row_within_shard)``."""
        # Binary search over cumulative row counts.
        lo, hi = 0, len(self._cumulative_rows) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_rows[mid] <= index:
                lo = mid + 1
            else:
                hi = mid
        shard_idx = lo
        row_offset = index - (self._cumulative_rows[shard_idx - 1] if shard_idx > 0 else 0)
        return shard_idx, row_offset

    def _read_row(self, shard_idx: int, row_offset: int) -> Tuple[bytes, int]:
        """
        Read one row from a shard file, returning ``(image_bytes, label)``.

        Uses the LRU cache to avoid re-opening recently used shards.
        We read a single row group slice so memory stays bounded.
        """
        path = self._shard_paths[shard_idx]
        pf = self._cache.get(path)

        # Read just the two columns we need for the target row.
        # pyarrow can read a slice efficiently.
        table = pf.read_row_groups(
            [self._find_row_group(pf, row_offset)],
            columns=["image", "label"],
        )

        # Adjust row_offset to be relative to the start of the row group.
        rg_start = self._row_group_start(pf, row_offset)
        local_offset = row_offset - rg_start

        image_col = table.column("image")
        label_col = table.column("label")

        # The image column is a struct with a "bytes" field.
        image_struct = image_col[local_offset].as_py()
        if isinstance(image_struct, dict):
            image_bytes = image_struct["bytes"]
        else:
            # Fallback: some versions store raw bytes directly.
            image_bytes = image_struct

        label = label_col[local_offset].as_py()
        return image_bytes, label

    @staticmethod
    def _find_row_group(pf: pq.ParquetFile, row_offset: int) -> int:
        """Return the row-group index that contains *row_offset*."""
        cumulative = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            cumulative += pf.metadata.row_group(rg_idx).num_rows
            if row_offset < cumulative:
                return rg_idx
        return pf.metadata.num_row_groups - 1

    @staticmethod
    def _row_group_start(pf: pq.ParquetFile, row_offset: int) -> int:
        """Return the first global row index of the row group containing *row_offset*."""
        cumulative = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            num = pf.metadata.row_group(rg_idx).num_rows
            if row_offset < cumulative + num:
                return cumulative
            cumulative += num
        return cumulative

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < 0 or index >= self._length:
            raise IndexError(
                f"index {index} out of range for dataset of size {self._length}"
            )

        shard_idx, row_offset = self._locate(index)
        image_bytes, label = self._read_row(shard_idx, row_offset)

        # Decode JPEG bytes -> PIL Image.
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure RGB (some images might be grayscale or RGBA).
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transform (returns a tensor).
        image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# DataLoader worker init -- each worker gets its own shard cache
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id: int) -> None:
    """
    Called once per DataLoader worker process.  Ensures each worker has its
    own independent shard cache (they are separate processes, so this happens
    automatically via fork, but we reinitialize to be safe with spawn).
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, ImageNetParquetDataset):
            # Reinitialize the cache in this worker.
            dataset._cache = _ShardLRUCache(maxsize=dataset._cache._maxsize)


# ---------------------------------------------------------------------------
# Convenience: build train & val DataLoaders in one call
# ---------------------------------------------------------------------------
def create_imagenet_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    cache_shards: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader`` objects ready for a standard
    ImageNet training loop.

    Parameters
    ----------
    data_dir : str
        Path to the directory with parquet shard files.
    batch_size : int
        Mini-batch size (default 256).
    num_workers : int
        Number of data-loading worker processes (default 8).
    image_size : int
        Crop / resize target (default 224).
    cache_shards : int
        Number of parquet shards each worker keeps in memory (default 2).
    pin_memory : bool
        Whether to use pinned (page-locked) memory for faster GPU transfer.

    Returns
    -------
    train_loader, val_loader : tuple[DataLoader, DataLoader]
    """
    train_ds = ImageNetParquetDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        cache_shards=cache_shards,
    )
    val_ds = ImageNetParquetDataset(
        data_dir=data_dir,
        split="validation",
        image_size=image_size,
        cache_shards=cache_shards,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the ImageNet parquet dataset loader."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing parquet shards "
        "(e.g. ./imagenet-1k/data).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which split to test (default: validation).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the test DataLoader (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="How many batches to iterate for the test (default: 5).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (default: 224).",
    )
    parser.add_argument(
        "--cache-shards",
        type=int,
        default=2,
        help="Number of shard files to keep in memory per worker (default: 2).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ImageNet Parquet Dataset - Sanity Check")
    print("=" * 60)

    ds = ImageNetParquetDataset(
        data_dir=args.data_dir,
        split=args.split,
        image_size=args.image_size,
        cache_shards=args.cache_shards,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )

    print(f"\nIterating {args.num_batches} batches (batch_size={args.batch_size})...\n")

    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        if i >= args.num_batches:
            break
        elapsed = time.perf_counter() - t0
        print(
            f"  batch {i + 1:3d}  |  images: {list(images.shape)}  "
            f"dtype={images.dtype}  |  labels: {list(labels.shape)}  "
            f"min={labels.min().item()} max={labels.max().item()}  |  "
            f"time: {elapsed:.2f}s"
        )
        t0 = time.perf_counter()

    # Quick single-sample check.
    print("\nSingle-sample check (index 0):")
    img, lbl = ds[0]
    print(f"  image shape: {list(img.shape)}  dtype: {img.dtype}")
    print(f"  pixel range:  [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"  label: {lbl}")
    print("\nDone.")


if __name__ == "__main__":
    main()
