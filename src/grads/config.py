
import torch
import torch.nn as nn

import model as m


import hashlib


class Conv_Configs():
    def __init__(self, data: tuple, device: torch.device, epochs: int = 1, n_layers: int = 3, hidden_units: int = 32, out_channels: int = 8, logits: bool = False) -> None:
        self.model: nn.Module = m.Conv_Model(
            n_layers=n_layers, hidden_units=hidden_units, out_channels=out_channels, logits=logits)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss() if logits else torch.nn.NLLLoss()
        self.epochs = epochs
        self.train_loader, self.eval_loader = data
        self.hash = hashlib.sha256(
            f'{device}{epochs}{n_layers}{hidden_units}{out_channels}{logits}'.encode()).hexdigest()
