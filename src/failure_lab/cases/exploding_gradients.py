"""An unchanged learning rate becomes unstable after a feature-unit change."""

import torch
from torch import Tensor

from failure_lab.diagnostics import norm, scalar


def dataset() -> tuple[Tensor, Tensor]:
    x = torch.linspace(-10, 10, 64)[:, None]
    return x, 3 * x


def train(lr: float, steps: int = 40) -> dict:
    x, y = dataset()
    weight = torch.zeros(1, 1, requires_grad=True)
    history = []
    for step in range(steps):
        loss = (x @ weight - y).square().mean()
        loss.backward()
        assert weight.grad is not None
        record = {
            "step": step,
            "loss": scalar(loss.item()),
            "grad_norm": scalar(norm([weight.grad])),
            "parameter_norm": scalar(norm([weight])),
            "finite_gradients": bool(torch.isfinite(weight.grad).all()),
        }
        history.append(record)
        if not torch.isfinite(loss) or not record["finite_gradients"]:
            break
        with torch.no_grad():
            weight -= lr * weight.grad
        weight.grad = None
    final_loss = (x @ weight - y).square().mean().item()
    return {
        "lr": lr,
        "history": history,
        "final_loss": scalar(final_loss),
        "final_weight": scalar(weight.item()),
        "finite_training": all(
            isinstance(r["loss"], float) and r["finite_gradients"] for r in history
        )
        and bool(torch.isfinite(weight).all())
        and bool(torch.isfinite(torch.tensor(final_loss))),
    }


def run() -> dict:
    x, _ = dataset()
    curvature = 2 * x.square().mean().item()
    bad, good = train(0.1), train(0.01)
    max_grad = max(row["grad_norm"] for row in good["history"])
    return {
        "symptom": "Growing gradient/parameter norms followed by infinite training loss.",
        "hypothesis": "A feature scaling change invalidated the existing learning rate.",
        "evidence": {
            "hessian": curvature,
            "stable_lr_upper_bound": 2 / curvature,
            "broken_error_multiplier": 1 - 0.1 * curvature,
            "repaired_error_multiplier": 1 - 0.01 * curvature,
            "normalized_feature_lr_upper_bound": 2 / (2 * (x / 10).square().mean().item()),
        },
        "root_cause": "Full-batch quadratic GD diverges when |1 - lr * Hessian| > 1.",
        "repair": "Reduce lr from 0.1 to 0.01, inside the measured stability interval.",
        "broken": {k: v for k, v in bad.items() if k != "history"},
        "repaired": {
            **{k: v for k, v in good.items() if k != "history"},
            "max_grad_norm": max_grad,
        },
        "traces": {"broken": bad["history"], "repaired": good["history"]},
        "verification": {
            "expected_failure": not bad["finite_training"],
            "finite_repair": good["finite_training"],
            "bounded_gradients": max_grad <= 207,
            "convergence": good["final_loss"] < 1e-8,
        },
    }
