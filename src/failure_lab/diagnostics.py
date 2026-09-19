"""Lightweight diagnostics; non-finite scalars serialize as strings, never invalid JSON."""

import math
import platform
import random
import sys
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def scalar(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def stats(tensor: Tensor) -> dict[str, Any]:
    values = tensor.detach().double()
    finite = torch.isfinite(values)
    valid = values[finite]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "count": tensor.numel(),
        "finite_count": int(finite.sum()),
        "nan_count": int(torch.isnan(values).sum()),
        "inf_count": int(torch.isinf(values).sum()),
        "all_finite": bool(finite.all()),
        "min_finite": float(valid.min()) if valid.numel() else None,
        "max_finite": float(valid.max()) if valid.numel() else None,
    }


def norm(tensors: Iterable[Tensor]) -> float:
    # Accumulate in float64 so the diagnostic itself does not overflow in float32.
    return float(
        sum((t.detach().double().square().sum() for t in tensors), torch.tensor(0.0)).sqrt()
    )


def require_same_shape(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Exact shapes required: prediction={prediction.shape}, target={target.shape}"
        )


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "seed": 7,
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
