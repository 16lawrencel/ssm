import torch
from torch import nn, Tensor

import lightning as L

from .dss import DSSLayer


class S4(nn.Module):
    def __init__(self, depth: int, n: int, h: int, vocab_size: int, out_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, h)
        self.layers = nn.ModuleList(DSSLayer(n, h) for _ in range(depth))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, out_size))

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:
        # x: (B, L, H)
        # x_lengths: (B,)

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        # Only get the output from the last point in sequence
        x = torch.stack(
            [x[batch, length - 1, :] for batch, length in enumerate(x_lengths)],
            dim=0,
        )

        x = self.out(x)
        return x
