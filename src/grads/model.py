import torch
import torch.nn as nn

from collections import OrderedDict


class Conv_Model(nn.Module):
    def __init__(self, n_layers: int, hidden_units: int, out_channels: int, logits: bool):
        super(Conv_Model, self).__init__()
        self.model_name = self._get_name()
        self.n_layers = n_layers
        self.out_channels = out_channels
        self.hidden_units = hidden_units
        self.logits = logits
        self.fc0 = nn.Linear(self.out_channels*28*28,
                             self.hidden_units, bias=False)
        self.fc1 = nn.Linear(self.hidden_units, 10,
                             bias=False)  # 10 output classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.conv, self.linear = self.make_layers()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        if not self.logits:
            return self.log_softmax(x)  # apply softmax instead of raw logits
        return x

    def _init_conv_layers(self):
        layers = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(
            in_channels=1, out_channels=self.out_channels, kernel_size=3, padding='same', bias=False))]))
        for idx in range(self.n_layers-1):
            conv = nn.Conv2d(self.out_channels, self.out_channels,
                             kernel_size=3, padding="same", bias=False)
            layers.add_module(f'conv{idx+2}', conv)
        return layers

    def _init_linear_layers(self):
        layers = nn.Sequential(OrderedDict(
            [("linear1", self.fc0), ("relu1", torch.nn.ReLU(inplace=True))]))
        for idx in range(self.n_layers-1):
            if idx == self.n_layers-1:
                layers.add_module(f'relu{idx+2}', torch.nn.ReLU(inplace=True))
            else:
                layers.add_module(f'linear{idx+2}', nn.Linear(self.hidden_units,
                                                              self.hidden_units, bias=False))
        return layers

    def make_layers(self):
        conv_layers = self._init_conv_layers()
        linear_layers = self._init_linear_layers()

        return conv_layers, linear_layers
