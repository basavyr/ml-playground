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
    .. info:: All datasets will be transformer according to the following ruleset: `num_channels=3` ; `RandomResizedCrop(224)`
    """
    SUPPORTED_DATASETS = ["mnist", "fashion",
                          "cifar10", "cifar100", "tiny", "imagenet1k"]
    DATASETS_MAPPING = {"mnist": {"mean": (0.1307,),
                                  "std": (0.3081,),
                                  "num_classes": 10,
                                  "num_channels": 1,
                                  },
                        "fashion": {"mean": (0.2860,),
                                    "std": (0.3530,),
                                    "num_classes": 10,
                                    "num_channels": 1,
                                    },
                        "cifar10": {"mean": (0.4914, 0.4822, 0.4465),
                                    "std": (0.2470, 0.2435, 0.2616),
                                    "num_classes": 10,
                                    "num_channels": 3,
                                    },
                        "cifar100": {"mean": (0.5071, 0.4867, 0.4408),
                                     "std": (0.2675, 0.2565, 0.2761),
                                     "num_classes": 100,
                                     "num_channels": 3,
                                     },
                        "tiny": {"mean": (0.4804, 0.4482, 0.3976),
                                 "std": (0.2764, 0.2689, 0.2817),
                                 "num_classes": 200,
                                 "num_channels": 3,
                                 },
                        "imagenet1k": {"mean": (0.485, 0.456, 0.406),
                                       "std": (0.229, 0.224, 0.225),
                                       "num_classes": 1000,
                                       "num_channels": 3,
                                       },
                        }

    def __init__(self, default_data_dir: str):
        os.makedirs(default_data_dir, exist_ok=True)
        self.default_data_dir = default_data_dir

    def get_dataset(self, dataset_name: str, custom_image_folder: str | None = None, download: bool = False) -> Dataset | None:
        """
        - Helper method for preparing a standard Torchvision dataset such as MNIST, CIFAR10, etc.
        - The current implementation also supports custom datasets such as [Tiny ImageNet 200](https://gist.github.com/basavyr/be29dde85dbd9623b3e41188ed0e0592), which can be provided as a path.

        The method can be used in the following way(s)::

            dataset = dataset_helper.get_dataset(
                        dataset_name="tiny", custom_image_folder="./data/tiny-imagenet-200", download=False)
            dataset = dataset_helper.get_dataset(
                        dataset_name="mnist", download=True)

        .. note:: Datasets such as **Tiny Imagenet 200** requires argument `custom_image_folder` to be provided by the user. The path must point to the directory in which all the samples are structured in the class/label-specific subfolder. This is required by `ImageFolder` wrapper.

        .. info:: For **Tiny Imagenet 200**, we provide support for download when `download=True`, which is assured through `utils.download_and_prepare_tiny_imagenet`
        """
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
            assert custom_image_folder is not None, KeyError(
                "Tiny ImageNet 200 requires `custom_image_folder` argument to be provided.")
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
        if force_3_channels:
            tf_list.transforms.append(
                transforms.Grayscale(num_output_channels=3))
        tf_list.transforms.append(transforms.ToTensor())
        tf_list.transforms.append(transforms.Normalize(
            mean=self.dataset_mapping['mean'], std=self.dataset_mapping['std']))
        if force_resize:
            tf_list.transforms.append(transforms.RandomResizedCrop(224))
        return tf_list
