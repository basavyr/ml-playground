import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.functional


class Model(nn.Module):
    def __init__(self, input_size: int, output_size: int, params: list):
        super().__init__()
        self.params = params
        self.input_size = input_size
        self.output_size = output_size
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(self.input_size*self.input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, self.output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        a, b = self.params  # perform ax+b
        x = a*x+b
        x = self.layer_stack(x)
        return x.argmax(1)
