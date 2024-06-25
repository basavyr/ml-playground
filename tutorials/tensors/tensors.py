# source: https://www.youtube.com/watch?v=IC0_FRiX-sw

import torch
from torch import nn
import torch.functional as F

import t_ops as tops


class Conv(nn.Module):
    def __init__(self, image_size: tuple[int, int]):
        super(Conv, self).__init__()
        self.height, self.width = image_size  # we work with squared images
        self.in_channels = 3  # for RGB images
        self.out_channels = 3  # can be any number you decide
        self.kernel_size = 3
        self.pool_size = 2

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      self.kernel_size, padding='same'),
            nn.MaxPool2d(self.pool_size),
            nn.Conv2d(self.out_channels, self.out_channels,
                      self.kernel_size, padding="same"),
            nn.MaxPool2d(self.pool_size),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_net(x)
        x = self.flatten(x)
        return x


def generate_grid(device, image_size):
    """
    - generate an image of size `height x width` which has a fixed number of channels (provided by local `n_channels` variable)
    """
    n_channels = 3
    height, width = image_size
    return tops.generate_tensor(device, 1, n_channels, height, width)


def generate_data(n_samples: int, device: str, image_size: tuple[int, int], method: str = "stack"):
    if method == "stack":
        # using stack
        return torch.stack([generate_grid(device, image_size) for _ in range(n_samples)], dim=0)
    else:
        # using cat
        return torch.cat([generate_grid(device, image_size) for _ in range(n_samples)], dim=0)


if __name__ == "__main__":

    m = 5
    image_size = (10, 10)

    device = torch.device("mps")
    images = generate_data(m, device, image_size, "cat")

    for idx, grid in enumerate(images):
        print(f'Image {idx+1}')
        print(grid)
        print(f'Flattened Image {idx+1}')
        print(grid.view(-1))

        model = Conv(image_size).to(device)
        model.train()
        output = model.forward(grid)
        print("Output shape:", output.shape)
        print("Output:", output)
        break
