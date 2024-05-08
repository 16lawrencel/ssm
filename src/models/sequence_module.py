import torch
from torch import nn, Tensor
import lightning as L


class SequenceModule(L.LightningModule):
    def __init__(self, net: nn.Module):
        super().__init__()

        self.net = net
        self.metric = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:
        # x: (B, L)
        # x_lengths: (B,)
        x = self.net(x, x_lengths)
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
