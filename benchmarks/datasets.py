import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
import os

from utils import download_and_prepare_tiny_imagenet, validate_tiny_imagenet_image_folder


class RandomEmbeddings(Dataset):
    def __init__(self, num_samples: int, sequence_length: int, embedding_dim: int, vocab_size: int, torch_dtype: torch.dtype | None):
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.x = torch.randn(num_samples, sequence_length,
                             embedding_dim, dtype=torch_dtype)
        self.y = torch.randint(
            1, vocab_size, (num_samples, sequence_length), dtype=torch.long)

    def __len__(self,):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class StandardDatasets:
    """
    .. note::
        Before using this class with Tiny Imagenet or Imagenet, it is required to download and prepare the datasets in the root directory, such that they can be used through `ImageFolder`.
    """
    SUPPORTED_DATASETS = ["mnist", "fashion",
                          "cifar10", "cifar100", "tiny", "imagenet1k"]
    DATASETS_MAPPING = {"mnist": {"mean": (0.1307,),
                                  "std": (0.3081,),
                                  },
                        "fashion": {"mean": (0.2860,),
                                    "std": (0.3530,),
                                    },
                        "cifar10": {"mean": (0.4914, 0.4822, 0.4465),
                                    "std": (0.2470, 0.2435, 0.2616),
                                    },
                        "cifar100": {"mean": (0.5071, 0.4867, 0.4408),
                                     "std": (0.2675, 0.2565, 0.2761),
                                     },
                        "tiny": {"mean": (0.4804, 0.4482, 0.3976),
                                 "std": (0.2764, 0.2689, 0.2817),
                                 },
                        "imagenet1k": {"mean": (0.485, 0.456, 0.406),
                                       "std": (0.229, 0.224, 0.225),
                                       },
                        }

    def __init__(self, default_data_dir: str):
        os.makedirs(default_data_dir, exist_ok=True)
        self.default_data_dir = default_data_dir

    def get_dataset(self, dataset_name: str, custom_image_folder: str | None = None, download: bool = False) -> Dataset | None:
        assert dataset_name in self.SUPPORTED_DATASETS, f"Unsupported dataset type (Currently supported datasets: {self.SUPPORTED_DATASETS})"
        self.dataset_mapping = self.DATASETS_MAPPING[dataset_name]
        dataset_name = dataset_name.lower()
        ds = None
        if custom_image_folder is None:
            root_dir = self.default_data_dir
        else:
            root_dir = custom_image_folder
            os.makedirs(root_dir, exist_ok=True)
        if dataset_name == "mnist":
            tf = self.get_transform(force_3_channels=True, force_resize=True)
            ds = MNIST(root=root_dir, transform=tf, download=download)
        elif dataset_name == "fashion":
            tf = self.get_transform(force_3_channels=True, force_resize=True)
            ds = FashionMNIST(root=root_dir, transform=tf, download=download)
        elif dataset_name == "cifar10":
            tf = self.get_transform(force_resize=True)
            ds = CIFAR10(root=root_dir, transform=tf, download=download)
        elif dataset_name == "cifar100":
            tf = self.get_transform(force_resize=True)
            ds = CIFAR100(root=root_dir, transform=tf, download=download)
        elif dataset_name == "tiny":
            if download:
                tiny_imagenet_dir = download_and_prepare_tiny_imagenet(
                    root_dir)
                root_dir = tiny_imagenet_dir
            tf = self.get_transform(force_resize=True)
            try:
                validate_tiny_imagenet_image_folder(root_dir)
                ds = ImageFolder(root=root_dir, transform=tf)
            except FileNotFoundError or FileExistsError:
                raise FileNotFoundError(
                    'Incorrect path provided for the ImageFolder. Tiny ImageNet 200 requires the root directory as /path/to/tiny-imagenet-200')
        elif dataset_name == "imagenet1k":
            raise NotImplementedError("ImageNet-1k not supported yet.")
        return ds

    def get_transform(self, force_3_channels: bool = False, force_resize: bool = False):
        tf_list = transforms.Compose([])
        tf_list.transforms.append(transforms.ToTensor())
        tf_list.transforms.append(transforms.Normalize(
            mean=self.dataset_mapping['mean'], std=self.dataset_mapping['std']))
        if force_3_channels:
            tf_list.transforms.append(
                transforms.Grayscale(num_output_channels=3))
        if force_resize:
            tf_list.transforms.append(transforms.RandomResizedCrop(224))
        return tf_list
