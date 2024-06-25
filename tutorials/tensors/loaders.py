import torch

from torch.utils.data import Dataset, DataLoader


def generate_image(height: int, width: int) -> torch.tensor:
    """
    Generates a single image tensor of shape [1, height, width] with random values.
    """
    return torch.rand([1, height, width])


def generate_data(m: int, size: tuple[int, int]) -> torch.tensor:
    """
    Generates a tensor of size `m`, where each tensor is an image of size `height x width`.
    """
    height, width = size

    # Generate a list of `m` image tensors and stack them into a single tensor
    t = torch.stack([generate_image(height, width) for _ in range(m)])
    return t


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


def get_mnist_samples(n_samples: int) -> torch.tensor:
    # Generate 3 images of size 2x2 and print the resulting tensor
    height = 64
    width = 64
    data = generate_data(n_samples, (height, width))
    return data


def main():
    M = 64
    mnist = get_mnist_samples(M)
    labels = torch.tensor([i for i in range(M)], dtype=torch.float64)

    # Create an instance of the custom dataset
    custom_dataset = CustomDataset(mnist, labels)

    # Create a data loader
    data_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

    # Example of using the data loader
    for batch_idx, (x, y) in enumerate(data_loader):
        print(f"Batch {batch_idx+1}")
        print(f"Data: {x.shape}")
        print(f"Labels: {y.shape}")


if __name__ == "__main__":
    main()
