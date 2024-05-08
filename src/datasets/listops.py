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
    def __init__(self, listops_path: Path, max_length: int = 6000):
        self.vocab_idx_map = {
            LISTOPS_VOCAB[idx]: idx for idx in range(len(LISTOPS_VOCAB))
        }

        with listops_path.open("r") as f:
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
