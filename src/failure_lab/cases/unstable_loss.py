"""A confident wrong prediction exposes probability-space underflow."""

import torch
from torch import Tensor


def broken(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=-1)
    return -probabilities.log().gather(1, targets[:, None]).mean()


def fixture() -> tuple[Tensor, Tensor]:
    return torch.tensor([[100.0, -100.0], [-100.0, 100.0]]), torch.tensor([1, 0])
