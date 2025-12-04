from typing import Callable
import torch
from torch import nn as nn

import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# local imports
from utils import get_optimal_device, log


class RandomDataset(Dataset):
    def __init__(self, num_samples: int, sequence_length: int, embedding_dim: int, vocab_size: int, torch_dtype: torch.dtype | None):
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.x = torch.randn(num_samples, sequence_length,
                             embedding_dim, dtype=torch_dtype)
        self.y = torch.randint(
            1, vocab_size, (num_samples, sequence_length), dtype=torch.long)

    def __len__(self,):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, vocab_size: int, torch_dtype: torch.dtype | None):
        super(Transformer, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                        nhead=n_heads, batch_first=True, dtype=torch_dtype)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, dtype=torch_dtype)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.torch_dtype = torch_dtype

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        decoder_output = self.decoder(tgt=tgt,
                                      memory=memory,
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
            memory = torch.zeros(
                x.shape[0], 0, x.shape[-1]).to(x.device, dtype=x.dtype)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                x.shape[1], x.device, dtype=torch.float32)
            optimizer.zero_grad()

            y = model(x, memory, tgt_mask)  # torch.Size([128, 32, 30000])

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
        log.info(f'Epoch {epoch+1}: Loss= {epoch_loss:.3f}')


def main():
    # training configs
    device = get_optimal_device()
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10
    learning_rate = 1e-4
    torch_dtype = torch.float32  # TODO test with bfloat16

    # model configs
    d_model = 384
    n_layers = 6
    n_heads = 8

    # data configs
    embedding_dim = d_model
    vocab_size = 30000
    num_samples = 1000
    batch_size = 24
    sequence_length = 128

    dataset = RandomDataset(num_samples=num_samples,
                            sequence_length=sequence_length,
                            embedding_dim=embedding_dim,
                            vocab_size=vocab_size,
                            torch_dtype=torch_dtype)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Transformer(d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        vocab_size=vocab_size,
                        torch_dtype=torch_dtype)

    train_model(model=model,
                device=device,
                loss_fn=loss_fn,
                train_loader=train_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate)


if __name__ == "__main__":
    main()
