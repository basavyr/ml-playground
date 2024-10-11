from torch.nn import functional as F
import torch

import torch.nn as nn


from transformers import PreTrainedModel, PretrainedConfig


import os
import dotenv
dotenv.load_dotenv()

REPO_URL = os.environ.get("REPO_URL")
REPO_ID = os.environ.get("REPO_ID")
MODEL_NAME = os.environ.get("MODEL_NAME")


class AdnetConfig(PretrainedConfig):
    model_type = MODEL_NAME

    def __init__(self, input_size=2, hidden_size1=512, hidden_size2=1024, output_size=1, architectures=[MODEL_NAME], pipeline_tag=MODEL_NAME, license="mit", repo_url=REPO_URL, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.architectures = architectures
        self.pipeline_tag = pipeline_tag
        self.license = license
        self.repo_url = repo_url
        self.model_name = MODEL_NAME
        self.repo_id = REPO_ID


class Adnet(PreTrainedModel):
    def __init__(self, config: AdnetConfig):
        super(Adnet, self).__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.fc3 = nn.Linear(config.hidden_size2, config.output_size)
        self.relu = F.relu
        self.model_name = config.model_name
        self.model_type = config.model_type
        self.repo_id = config.repo_id

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output
