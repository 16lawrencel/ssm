import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from pathlib import Path


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
    def __init__(self, listops_path: Path):

        self.vocab_idx_map = {
            LISTOPS_VOCAB[idx]: idx for idx in range(len(LISTOPS_VOCAB))
        }

        with listops_path.open("r") as f:
            _header = f.readline()
            data = [line.split("\t") for line in f.readlines()]

            self.inputs: list[str] = [row[0] for row in data]
            self.outputs: list[int] = [int(row[1]) for row in data]

        self.inputs = self.inputs[:5]
        self.outputs = self.outputs[:5]

    @property
    def vocab_size(self):
        return len(self.vocab_idx_map)

    def __len__(self):
        return len(self.inputs)

    def _convert_str_to_tensor(self, s: str) -> Tensor:
        return torch.tensor(
            [self.vocab_idx_map[word] for word in s.split()], dtype=torch.int64
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        inp_onehot = self._convert_str_to_tensor(self.inputs[idx])
        return inp_onehot, self.outputs[idx]
