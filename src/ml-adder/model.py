from torch.nn import functional as F
import torch

import torch.nn as nn


class Adnet(nn.Module):
    def __init__(self):
        super(Adnet, self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = F.relu

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output
