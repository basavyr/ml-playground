import torch
import torchvision
from torchvision import transforms, datasets


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


def main():
    # Generate 3 images of size 2x2 and print the resulting tensor
    data = generate_data(64, (28, 28))
    print(data.shape)


if __name__ == "__main__":
    main()
