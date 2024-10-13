import torchvision

from torch.utils.data import DataLoader, Dataset
import torch


class MNIST:
    DEFAULT_BATCH_SIZE = 256
    TF = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self, eval_only: bool = False):
        self.eval_only = eval_only

        self.mnist = self.get_mnist()

    def get_mnist(self) -> tuple[DataLoader, DataLoader]:
        mnist_eval = torchvision.datasets.MNIST(
            "./data", download=True, train=False, transform=self.TF)
        eval_loader = DataLoader(
            mnist_eval, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=False)

        if self.eval_only:
            return None, eval_loader

        mnist_train = torchvision.datasets.MNIST(
            "./data", download=True, train=True, transform=self.TF)
        train_loader = DataLoader(
            mnist_train, batch_size=self.DEFAULT_BATCH_SIZE, shuffle=True)
        return train_loader, eval_loader
