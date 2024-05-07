import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path


def checkpoint_epoch(checkpoint_path: Path, epoch: int) -> Path:
    return checkpoint_path / f"epoch={epoch}.pt"


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    checkpoint_path: Path,
    load_from_epoch: int | None = None,
    device: str = "cpu",
) -> list[float]:
    losses = []
    starting_epoch = 0

    if load_from_epoch is not None:
        checkpoint_epoch_path = checkpoint_epoch(checkpoint_path, load_from_epoch)
        checkpoint = torch.load(checkpoint_epoch_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        losses = checkpoint["losses"]
        starting_epoch = load_from_epoch

    model.to(device)
    for epoch in range(starting_epoch, num_epochs):
        print(f"[Epoch {epoch}]")
        total_loss = 0
        total_count = 0
        for X, y in dataloader:
            # Only support batch size 1 for now
            assert X.shape[0] == 1

            X = X.to(device)
            y = y.to(torch.float32).to(device)

            optim.zero_grad()
            pred = model(X.squeeze())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_count += len(X)
            loss.backward()
            optim.step()

        avg_loss = total_loss / total_count
        print(f"[Epoch {epoch}] Avg loss = {avg_loss}")
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            checkpoint_epoch_path = checkpoint_epoch(checkpoint_path, epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "losses": losses,
                },
                checkpoint_epoch_path,
            )

    return losses
