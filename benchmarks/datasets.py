import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder


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
    DATASET_MAPPING = {
        "mnist": {"mean": (0.1307,),
                  "std": (0.3081,)},
        "fashion": {"mean": (0.2860,),
                    "std": (0.3530,)},
        "cifar10": {"mean": (0.4914, 0.4822, 0.4465),
                    "std": (0.2470, 0.2435, 0.2616)},
        "cifar100": {"mean": (0.5071, 0.4867, 0.4408),
                     "std": (0.2675, 0.2565, 0.2761)},
        "imagenet1k": {"mean": (0.485, 0.456, 0.406),
                       "std": (0.229, 0.224, 0.225)},
        "tiny": {"mean": (0.4804, 0.4482, 0.3976),
                 "std": (0.2764, 0.2689, 0.2817)}
    }

    def __init__(self, dataset_name: str, train_batch_size: int, test_batch_size: int, shuffle_train: bool = True):
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle_train = shuffle_train
