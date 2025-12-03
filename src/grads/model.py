import torch
import torch.nn as nn
import math

from collections import OrderedDict


class Conv_Model(nn.Module):
    def __init__(self, num_classes: int, n_layers: int, hidden_units: int, in_channels: int, out_channels: int, image_size: tuple[int, int], maxpool: bool, logits: bool):
        super(Conv_Model, self).__init__()
        self.model_name = self._get_name()
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_units = hidden_units
        self.height, self.width = image_size
        self.logits = logits
        self.maxpool = maxpool

        # maxpool config
        self.mp2d_padding = 0
        self.mp2d_stride = 2
        self.mp2d_kernel = 2
        self.mp2d_dilation = 1

        self.new_size = math.floor(
            (self.height+2*self.mp2d_padding-self.mp2d_kernel)/self.mp2d_stride+1)

        self.conv, self.linear = self.make_layers()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.maxpool == True:
            x = nn.ReLU(inplace=True)(nn.MaxPool2d(kernel_size=self.mp2d_kernel,
                                                   stride=self.mp2d_stride,
                                                   padding=self.mp2d_padding)(x))
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        if self.logits == True:
            return x
        return nn.functional.log_softmax(x, dim=1)

    def _init_conv_layers(self):
        layers = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding='same', bias=False))]))
        layers.add_module("bn1", nn.BatchNorm2d(self.out_channels))
        for idx in range(self.n_layers-1):
            conv = nn.Conv2d(self.out_channels, self.out_channels,
                             kernel_size=3, padding="same", bias=False)
            layers.add_module(f'conv{idx+2}', conv)
            layers.add_module(f'bn{idx+2}', nn.BatchNorm2d(self.out_channels))
            layers.add_module(f'relu{idx+2}', nn.ReLU(inplace=True))
        return layers

    def _init_linear_layers(self):
        layers = nn.Sequential(OrderedDict(
            [("linear1", nn.Linear(self.out_channels*self.new_size**2,
                                   self.hidden_units, bias=False)), ("relu1", nn.ReLU(inplace=True))]))

        for idx in range(self.n_layers-1):
            layers.add_module(
                f'linear{idx+2}', nn.Linear(self.hidden_units, self.hidden_units, bias=False))
            layers.add_module(f'relu{idx+2}', nn.ReLU(inplace=True))

        layers.add_module(f'linear{self.n_layers+1}', nn.Linear(self.hidden_units, self.num_classes,
                                                                bias=False))
        return layers

    def make_layers(self):
        conv_layers = self._init_conv_layers()
        linear_layers = self._init_linear_layers()

        return conv_layers, linear_layers
