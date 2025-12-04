from typing import Callable
import torch
from torch import nn as nn

import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
# local imports
from utils import get_optimal_device


class RandomDataset(Dataset):
    def __init__(self, num_samples: int, sequence_length: int, embedding_dim: int, vocab_size: int):
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.x = torch.randn(num_samples, sequence_length,  embedding_dim)
        self.y = torch.randint(
            1, vocab_size, (num_samples, sequence_length))

    def __len__(self,):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_causal_mask(size: int, device: torch.types.Device):
    mask = nn.Transformer.generate_square_subsequent_mask(size, device)
    return mask


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, vocab_size: int):
        super(Transformer, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                        nhead=n_heads)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        decoder_output = self.decoder(tgt=tgt,
                                      memory=tgt,
                                      tgt_mask=tgt_mask)
        logits = self.lm_head(decoder_output)

        return logits


def train_model(
    model: nn.Module,
    device: torch.types.Device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        for x, y_true in tqdm(train_loader, desc=f'Epoch {epoch+1}: Training transformer'):
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            # TODO: a target mask should be applied
            y = model(x, None)  # torch.Size([128, 32, 30000])

            # shift targets and logits
            y_shifted = y[:, :-1, :].contiguous()
            y_true_shifted = y_true[:, 1:].contiguous()

            y_shifted = y_shifted.view(-1, y_shifted.shape[-1])
            y_true_shifted = y_true_shifted.view(-1)

            loss = loss_fn(y_shifted, y_true_shifted)
            epoch_loss += loss.item()*x.shape[0]

            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch+1}: Loss= {epoch_loss:.3f}')


def main():
    # training configs
    device = get_optimal_device()
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10
    learning_rate = 1e-4

    # data configs
    num_samples = 10000
    vocab_size = 30000
    batch_size = 8
    sequence_length = 8
    embedding_dim = 384

    # model configs
    d_model = embedding_dim
    n_layers = 6
    n_heads = 8

    train_loader = DataLoader(RandomDataset(
        num_samples, sequence_length, embedding_dim, vocab_size), batch_size=batch_size, shuffle=False)

    model = Transformer(d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, vocab_size=vocab_size)

    train_model(model=model,
                device=device,
                loss_fn=loss_fn,
                train_loader=train_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate)


if __name__ == "__main__":
    main()
