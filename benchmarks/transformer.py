import torch
from torch import nn as nn
from torch.utils.data import DataLoader


from typing import Callable
import os
import sys
import time
from tqdm import tqdm

# local imports
from utils import get_optimal_device, get_flops_approx, generate_log_file
from datasets import RandomEmbeddings
from models import Transformer

import logging
log_file, generated_at = generate_log_file('transformer')
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()


def train_transformer(
    model: nn.Module,
    device: torch.types.Device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_start = time.monotonic_ns()
    training_times = []
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_start = time.monotonic_ns()
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
        epoch_time = (time.monotonic_ns()-epoch_start)*1e-9
        training_times.append(epoch_time)

        epoch_loss /= len(train_loader.dataset)
        log.info(
            f'Epoch {epoch+1}: Loss= {epoch_loss:.3f} [{epoch_time:3f} s]')

    training_time = (time.monotonic_ns()-train_start)*1e-9
    log.info(
        f'Full training finished in {training_time:.3f} s ({sum(training_times)/len(training_times):.3f} s per epoch)')


if __name__ == "__main__":

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

    # training configs
    device = get_optimal_device()
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10
    learning_rate = 1e-4
    torch_dtype = torch.float32

    dataset = RandomEmbeddings(num_samples=num_samples,
                               sequence_length=sequence_length,
                               embedding_dim=embedding_dim,
                               vocab_size=vocab_size,
                               torch_dtype=torch_dtype)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    log.info(f'Training on {device} for {num_epochs} epochs.\n<<< Config >>>\nBatch Size={batch_size} | Total samples={num_samples}\nSequence Length={sequence_length}\nN_decoder_layers={n_layers} | num_attn_heads={n_heads} | d_k={d_model}\n{"="*80}')

    model = Transformer(d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        vocab_size=vocab_size,
                        torch_dtype=torch_dtype)

    train_start = time.monotonic_ns()
    train_transformer(model=model,
                      device=device,
                      loss_fn=loss_fn,
                      train_loader=train_loader,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate)
    train_duration = (time.monotonic_ns() - train_start)*1e-9

    total_flops_approx = get_flops_approx(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        sequence_length=sequence_length,
        num_samples=num_samples,
        vocab_size=vocab_size,
        num_epochs=num_epochs)

    log.info(
        f'Achieved avg. << {(total_flops_approx/1e12)/train_duration:.4f} >> TFLOPS')

    log.info(f'System info: {os.uname().nodename}')
