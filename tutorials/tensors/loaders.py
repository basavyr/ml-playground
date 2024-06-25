import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data: torch.tensor, labels: np.ndarray):
        self.data = data
        self.true_labels = labels

        # manually transform the labels into binary classification
        self.labels = torch.tensor(
            list(map(lambda val: val == "ones", labels)), dtype=torch.bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class MNISTLike:
    """
    A class to generate and provide a data loader for MNIST-like data. Each sample 
    is a 28x28 image tensor of either all ones or random values. The labels are 
    binary, indicating whether the tensor is all ones (label 0) or random (label 1).

    Attributes:
    - TENSOR_TYPE_LABELS (list): List of possible tensor types ("ones" or "rands").
    - height (int): Height of the generated images (default 28).
    - width (int): Width of the generated images (default 28).
    - n_samples (int): Number of samples to generate.
    - batch_size (int): Batch size for the data loader.
    """

    TENSOR_TYPE_LABELS = ["ones", "rands"]
    height = 28
    width = 28

    def __init__(self, n_samples: int, batch_size: int):
        """
        Initializes the MNISTLike instance with the number of samples and batch size.

        Parameters:
        - n_samples (int): Number of samples to generate.
        - batch_size (int): Batch size for the data loader.
        """
        self.n_samples = n_samples
        self.batch_size = batch_size

    def generate_data(self, labels: np.ndarray) -> torch.Tensor:
        """
        Generates a tensor of tensors, where each tensor is an image of size `height x width`.

        Parameters:
        - labels (np.ndarray): Array of labels with values "ones" or "rands".

        Returns:
        - torch.Tensor: A tensor containing the generated tensors.
        """
        def generate_image(tensor_type):
            if tensor_type == "ones":
                return torch.ones([1, self.height, self.width])
            elif tensor_type == "rands":
                return torch.rand([1, self.height, self.width])
            else:
                raise ValueError(f"Unknown tensor type: {tensor_type}")

        tensor_list = [generate_image(tensor_type) for tensor_type in labels]
        return torch.cat(tensor_list, dim=0)

    def loader(self) -> DataLoader:
        """
        Creates a DataLoader for the generated MNIST-like data.

        The labels are randomly assigned as "ones" or "rands". Each label is converted
        to a binary value: 0 for "ones" and 1 for "rands".

        Returns:
        - DataLoader: A data loader for the generated dataset.
        """
        labels = np.random.choice(self.TENSOR_TYPE_LABELS, self.n_samples)
        data = self.generate_data(labels)

        # Transform labels into binary classification tensor
        binary_labels = torch.tensor(
            list(map(lambda val: val == "ones", labels)), dtype=torch.long)

        custom_dataset = CustomDataset(data, binary_labels)
        # Create a data loader
        data_loader = DataLoader(
            custom_dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader


def main():

    mnist = MNISTLike(64, 8)
    data_loader = mnist.loader()

    # Example of using the data loader
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to("mps"), y.to("mps")
        print(f"Batch {batch_idx+1}/{len(data_loader)}")
        print(f"Data: {x.shape}")
        # print(f"Data: {x}")
        print(f"Labels: {y.shape}")
        # print(f"Labels: {y}")


if __name__ == "__main__":
    main()
