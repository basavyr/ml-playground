import torch
from torch import nn as nn


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


class LinearNet(nn.Module):
    def __init__(self, num_hidden_layers: int, input_size: int, hidden_dim: int, output_size: int):
        super(LinearNet, self).__init__()
        assert num_hidden_layers > 1, "The neural network must have at least one hidden layers (num_hidden_layers > 1)"
        self.output_size = output_size
        self.num_classes = output_size
        self.input_size = input_size
        self.num_features = input_size

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_dim), nn.ReLU())
        for _ in range(num_hidden_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            self.layers.append(
                nn.ReLU())
        self.layers.append(
            nn.Linear(in_features=hidden_dim, out_features=output_size))

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)  # flatten input -> B, C*H*W (2D)
        logits = self.layers(x)
        return logits
