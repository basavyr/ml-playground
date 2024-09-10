from torch.nn import functional as F
import torch

import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin


from transformers import PreTrainedModel, PretrainedConfig


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


class AdnetConfig(PretrainedConfig):
    model_type = "adnet"

    def __init__(self, input_size=2, hidden_size1=512, hidden_size2=1024, output_size=1, architectures=["adnet"], pipeline_tag="adnet", license="mit", repo_url="https://huggingface.co/basavyr/adnet", **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.architectures = architectures
        self.pipeline_tag = pipeline_tag
        self.license = license
        self.repo_url = repo_url
        self.model_name = "adnet"


class Adnet_HF(PreTrainedModel):
    def __init__(self, config: AdnetConfig):
        super(Adnet_HF, self).__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.fc3 = nn.Linear(config.hidden_size2, config.output_size)
        self.relu = F.relu
        self.model_name = config.model_name
        self.model_type = config.model_type

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output
