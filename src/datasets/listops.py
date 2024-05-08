import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path

from src.datasets.utils import LRAUtils


LISTOPS_VOCAB: list[str] = list("0123456789") + [
    "(",
    ")",
    "[MIN",
    "[MAX",
    "[MED",
    "[SM",
    "]",
]


class ListopsDataset(Dataset):
    def __init__(self, data_dir: Path):
        super().__init__()

        self.vocab_idx_map = {
            LISTOPS_VOCAB[idx]: idx for idx in range(len(LISTOPS_VOCAB))
        }

        with data_dir.open("r") as f:
            _header = f.readline()
            data = [line.split("\t") for line in f.readlines()]

            self.inputs: list[str] = [row[0] for row in data]
            self.outputs: list[int] = [int(row[1]) for row in data]

        self.max_length = max(len(s.split()) for s in self.inputs)

    @property
    def vocab_size(self):
        return len(self.vocab_idx_map)

    def __len__(self):
        return len(self.inputs)

    def _convert_str_to_tensor(self, s: str) -> tuple[Tensor, int]:
        tensor = torch.tensor(
            [self.vocab_idx_map[word] for word in s.split()], dtype=torch.int64
        )
        length = len(tensor)
        padded_tensor = F.pad(tensor, (0, self.max_length - length), "constant", 0)
        return padded_tensor, length

    def __getitem__(self, idx: int) -> tuple[Tensor, int, int]:
        # Returns: inp, inp_length, output
        inp, inp_length = self._convert_str_to_tensor(self.inputs[idx])
        return inp, inp_length, self.outputs[idx]


class ListopsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, download: bool = False):
        super().__init__()

        if download:
            LRAUtils(data_dir).download()

        self.listops_dir = Path(data_dir) / "lra_release" / "listops-1000"
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.listops_train = ListopsDataset(self.listops_dir / "basic_train.tsv")
        self.listops_val = ListopsDataset(self.listops_dir / "basic_val.tsv")
        self.listops_test = ListopsDataset(self.listops_dir / "basic_test.tsv")

    def train_dataloader(self):
        return DataLoader(self.listops_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.listops_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.listops_test, batch_size=self.batch_size)
