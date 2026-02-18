"""
Binary arithmetic dataset: n-bit increment and decrement.

Each instance: b_1...b_n # y_1...y_n  (length T = 2n+1, MSB-first)
Vocabulary: {0, 1, #}  →  token ids {0, 1, 2}

B-RASP causal variables (increment):
  B_prev(k) = b_k
  C_in(k)   = AND(b_{k+1}, ..., b_n)          (carry)
  Y_out(k)  = B_prev(k) XOR C_in(k)

B-RASP causal variables (decrement):
  B^D_prev(k) = b_k
  W_in(k)     = AND(NOT b_{k+1}, ..., NOT b_n) (borrow)
  Y^D_out(k)  = B^D_prev(k) XOR W_in(k)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal

TOKEN_0, TOKEN_1, TOKEN_SEP = 0, 1, 2
VOCAB_SIZE = 3


def int_to_bits(x: int, n: int) -> list[int]:
    return [(x >> (n - 1 - k)) & 1 for k in range(n)]


def bits_to_int(bits: list[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def compute_carry(bits: list[int]) -> list[int]:
    """C_in(k) = AND(b_{k+1}, ..., b_n). Empty conjunction → 1."""
    n = len(bits)
    return [
        1 if k == n - 1 else int(all(bits[j] for j in range(k + 1, n)))
        for k in range(n)
    ]


def compute_borrow(bits: list[int]) -> list[int]:
    """W_in(k) = AND(NOT b_{k+1}, ..., NOT b_n). Empty conjunction → 1."""
    n = len(bits)
    return [
        1 if k == n - 1 else int(all(bits[j] == 0 for j in range(k + 1, n)))
        for k in range(n)
    ]


class BinaryArithmeticDataset(Dataset):
    """
    Dataset for n-bit increment or decrement with causal variable labels.

    Args:
        n: Bit width.
        task: 'increment' or 'decrement'.
        split: 'train' or 'test'.
        alpha: Train fraction (default 0.5).
        max_samples: Cap for large n.
        seed: Random seed for reproducible split.
    """

    def __init__(
        self,
        n: int,
        task: Literal["increment", "decrement"] = "increment",
        split: Literal["train", "test"] = "train",
        alpha: float = 0.5,
        max_samples: int = 50_000,
        seed: int = 42,
    ):
        self.n = n
        self.task = task
        self.seq_len = 2 * n + 1

        total = 1 << n
        rng = np.random.RandomState(seed)
        indices = np.arange(total) if total <= max_samples else rng.choice(total, max_samples, replace=False)
        rng.shuffle(indices)
        cut = int(len(indices) * alpha)
        indices = indices[:cut] if split == "train" else indices[cut:]

        self.tokens_list = []
        self.causal_vars_list = []

        for x in indices:
            bits = int_to_bits(x, n)
            if task == "increment":
                out = int_to_bits((bits_to_int(bits) + 1) % (1 << n), n)
                chain = compute_carry(bits)
            else:
                out = int_to_bits((bits_to_int(bits) - 1) % (1 << n), n)
                chain = compute_borrow(bits)

            tokens = bits + [TOKEN_SEP] + out
            self.tokens_list.append(torch.tensor(tokens, dtype=torch.long))
            self.causal_vars_list.append({
                "B_prev": torch.tensor(bits, dtype=torch.float),
                "chain": torch.tensor(chain, dtype=torch.float),
                "Y_out": torch.tensor(out, dtype=torch.float),
            })

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return self.tokens_list[idx], self.causal_vars_list[idx]


def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    causal = {
        k: torch.stack([b[1][k] for b in batch])
        for k in batch[0][1]
    }
    return tokens, causal


def build_dataloaders(n, task="increment", alpha=0.5, batch_size=512,
                      seed=42, num_workers=0):
    train_ds = BinaryArithmeticDataset(n, task, "train", alpha, seed=seed)
    test_ds = BinaryArithmeticDataset(n, task, "test", alpha, seed=seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=num_workers,
                          drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=num_workers)
    return train_dl, test_dl
