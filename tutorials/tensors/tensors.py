# source: https://www.youtube.com/watch?v=IC0_FRiX-sw

import torch


def generate_grid(n_samples, image_size):
    """
    - generate an image of size `height x width` which has a fixed number of channels (provided by local `n_channels` variable)
    """
    n_channels = 3
    height, width = image_size
    return torch.rand(n_samples, n_channels, height, width)


def generate_data(n_samples: int, image_size: tuple[int, int]):
    return generate_grid(n_samples, image_size)


if __name__ == "__main__":

    m = 5
    image_size = (2, 2)

    images = generate_grid(m, image_size)
    for idx, grid in enumerate(images):
        print(f'Image {idx+1}')
        print(grid)
