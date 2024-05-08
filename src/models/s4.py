import torch
from torch import nn, Tensor

import lightning as L

from .dss import DSSLayer


class S4(nn.Module):
    def __init__(self, vocab_size: int, depth: int, n: int, h: int):
        super().__init__()
        self.layers = nn.ModuleList(DSSLayer(n, h) for _ in range(depth))

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:
        # x: (B, L, H)
        # x_lengths: (B,)

        for layer in self.layers:
            x = layer(x)

        # Only get the output from the last point in sequence
        x = torch.stack(
            [x[batch, length - 1, :] for batch, length in enumerate(x_lengths)],
            dim=0,
        )

        return x


class SequenceModule(L.LightningModule):
    def __init__(self, vocab_size: int, depth: int, n: int, h: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, h)
        self.backbone = S4(vocab_size, depth, n, h)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, 10))

        self.metric = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:
        # x: (B, L)
        # x_lengths: (B,)
        x = self.embedding(x)
        x = self.backbone(x, x_lengths)
        x = self.out(x)
        return x

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, x_lengths, y = batch
        pred = self(x, x_lengths)
        loss = self.metric(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, x_lengths, y = batch
        pred = self(x, x_lengths)
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
