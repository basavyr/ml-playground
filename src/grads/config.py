
import torch
import torch.nn as nn

import model as m


import hashlib


class Conv_Configs():
    adam = torch.optim.Adam
    sgd = torch.optim.SGD
    sgd_kwargs = {
        'lr': 0.001,           # Learning rate
        'momentum': 0.9,       # Momentum value
        'weight_decay': 1e-4,  # L2 regularization (weight decay)
        'dampening': 0,        # Dampening for momentum
        'nesterov': True       # Use Nesterov momentum
    }
    adam_kwargs = {
        'lr': 0.001,            # Learning rate
        'betas': (0.9, 0.999),  # Momentum terms (first is for momentum)
        'eps': 1e-8,            # Small epsilon to avoid division by zero
        'weight_decay': 1e-4    # L2 regularization (optional)
    }

    def __init__(self, data: tuple, device: torch.device, epochs: int = 1, n_layers: int = 3, hidden_units: int = 32, out_channels: int = 8, logits: bool = False, use_adam: bool = False) -> None:
        self.model: nn.Module = m.Conv_Model(
            n_layers=n_layers, hidden_units=hidden_units, out_channels=out_channels, logits=logits)
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss() if logits == True else torch.nn.NLLLoss()
        self.optimizer = self.adam(self.model.parameters(), **self.adam_kwargs) if use_adam == True else self.sgd(
            self.model.parameters(), **self.sgd_kwargs)
        self.epochs = epochs
        self.train_loader, self.eval_loader = data
        self.hash = hashlib.sha256(
            f'{device}{epochs}{n_layers}{hidden_units}{out_channels}{logits}'.encode()).hexdigest()
