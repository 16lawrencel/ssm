import torch
from torch import nn, Tensor

import lightning as L

from .dss import DSSLayer


class S4(L.LightningModule):
    def __init__(self, vocab_size: int, depth: int, n: int, h: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, h)
        self.layers = nn.ModuleList(DSSLayer(n, h) for _ in range(depth))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, 1))

        self.metric = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        # This is temporary until we implement batching
        assert x.size(0) == 1
        x = x[0]

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        # Only get the output from the last point in sequence
        x = self.out(x[-1, :])

        return x

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = self.metric(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        loss = self.metric(pred, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
