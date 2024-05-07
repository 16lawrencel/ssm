import torch
from torch import nn, Tensor

from .dss import DSSLayer


class S4(nn.Module):
    def __init__(self, vocab_size: int, depth: int, n: int, h: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, h)
        self.layers = nn.ModuleList(DSSLayer(n, h) for _ in range(depth))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        # Only get the output from the last point in sequence
        x = self.out(x[-1, :])

        return x
