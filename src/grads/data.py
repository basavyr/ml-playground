import torch.utils
import torch.utils.data
import torchvision

from torch.utils.data import DataLoader
import torch


class MNIST:
    DEFAULT_BATCH_SIZE = 256
    TF = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self, eval_only: bool = False):
        self.eval_only = eval_only

        self.mnist = self.get_mnist()

    def get_mnist(self) -> tuple[DataLoader | None, DataLoader]:
        mnist_eval = torchvision.datasets.MNIST(
            "./data", download=True, train=False, transform=self.TF)
        eval_loader = DataLoader(
            mnist_eval, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=False)

        if self.eval_only == True:
            return None, eval_loader

        mnist_train = torchvision.datasets.MNIST(
            "./data", download=True, train=True, transform=self.TF)
        train_loader = DataLoader(
            mnist_train, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=True)
        return train_loader, eval_loader


class CIFAR10:
    DEFAULT_BATCH_SIZE = 256
    TF_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    TF_EVAL = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, eval_only: bool = False):
        self.eval_only = eval_only

        self.cifar10 = self.get_cifar10()

    def get_cifar10(self) -> tuple[DataLoader | None, DataLoader]:
        cifar10_eval = torchvision.datasets.CIFAR10(
            "./data", download=True, train=False, transform=self.TF_EVAL)
        eval_loader = torch.utils.data.DataLoader(
            cifar10_eval, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=False)

        if self.eval_only == True:
            return None, eval_loader

        cifar10_train = torchvision.datasets.CIFAR10(
            "./data", download=True, train=True, transform=self.TF_TRAIN)
        train_loader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=True)
        return train_loader, eval_loader


class CIFAR100:
    DEFAULT_BATCH_SIZE = 256

    TF_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    TF_EVAL = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, eval_only: bool = False):
        self.eval_only = eval_only

        self.cifar100 = self.get_cifar100()

    def get_cifar100(self) -> tuple[DataLoader | None, DataLoader]:
        eval_dataset = torchvision.datasets.CIFAR100(
            "./data", download=True, train=False, transform=self.TF_EVAL)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=False)

        if self.eval_only:
            return None, eval_loader

        train_dataset = torchvision.datasets.CIFAR100(
            "./data", download=True, train=True, transform=self.TF_TRAIN)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=True)

        return train_loader, eval_loader
