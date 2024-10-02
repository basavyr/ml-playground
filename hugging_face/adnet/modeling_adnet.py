# modeling_adnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import PreTrainedModel
from .configuration_adnet import AdnetConfig


class Adnet_HF(PreTrainedModel):
    def __init__(self, config: AdnetConfig):
        super(Adnet_HF, self).__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.fc3 = nn.Linear(config.hidden_size2, config.output_size)
        self.relu = F.relu

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output
