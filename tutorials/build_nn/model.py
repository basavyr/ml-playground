import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.functional


MPS_DEVICE = "mps"


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
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size*input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),  # 10 output_size
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
