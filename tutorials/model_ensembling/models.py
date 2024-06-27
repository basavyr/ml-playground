import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


class Model1(nn.Module):
    """
    Straightforward model, taking as input a tensor of shape `(mx1)`
    The input tensor will be flattened, to the dimension `n_channels*height*width`
    """

    def __init__(self, m: int):
        super().__init__()
        self.fc1 = nn.Linear(m, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x: torch.tensor):
        # flattened tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits


class Model2(nn.Module):
    """
    Straightforward model, similar to `Model1`, taking as input a tensor of shape `(mx1)`
    The input tensor will be flattened, to the dimension `n_channels*height*width`
    """

    def __init__(self, m: int):
        super().__init__()
        self.fc1 = nn.Linear(m, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.tensor):
        # flattened tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.fc3(x)
        return logits


data = torch.randn(100, 64, 1, 28, 28, device="mps")


for idx, data in enumerate(data):
    batch_size = data.size(0)
