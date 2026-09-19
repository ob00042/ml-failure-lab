"""A squeezed target silently converts paired regression into all-pairs regression."""

import torch
from torch import Tensor

from failure_lab.diagnostics import require_same_shape


def broken(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def fixture() -> tuple[Tensor, Tensor]:
    x = torch.linspace(-1, 1, 32)[:, None]
    return x, 2 * x + 0.5


def repaired(prediction: Tensor, target: Tensor) -> Tensor:
    require_same_shape(prediction, target)
    return (prediction - target).square().mean()


def train(use_repair: bool, steps: int = 120) -> dict:
    x, target = fixture()
    weight = torch.zeros(1, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    for _ in range(steps):
        prediction = x @ weight + bias
        loss = repaired(prediction, target) if use_repair else broken(prediction, target[:, 0])
        loss.backward()
        with torch.no_grad():
            weight -= 0.1 * weight.grad
            bias -= 0.1 * bias.grad
        weight.grad = bias.grad = None
    paired_mse = repaired(x @ weight + bias, target)
    return {
        "weight": weight.item(),
        "bias": bias.item(),
        "paired_mse": paired_mse.item(),
        "training_objective": loss.item(),
    }


def run() -> dict:
    from failure_lab.diagnostics import stats

    _, target = fixture()
    prediction = target.clone().requires_grad_()
    bad_loss = broken(prediction, target[:, 0])
    (gradient,) = torch.autograd.grad(bad_loss, prediction)
    # Explicit all-pairs calculation independently verifies broadcast semantics.
    pairwise = torch.stack(
        [
            (prediction[i, 0] - target[j, 0]).square()
            for i in range(len(target))
            for j in range(len(target))
        ]
    ).mean()
    torch.testing.assert_close(bad_loss, pairwise)
    caught = False
    try:
        repaired(prediction, target[:, 0])
    except ValueError:
        caught = True
    bad, good = train(False), train(True)
    return {
        "symptom": "Finite loss penalizes perfect predictions and training collapses to the mean.",
        "hypothesis": "A [N] target broadcasts against [N, 1] predictions into [N, N].",
        "evidence": {
            "prediction_shape": list(prediction.shape),
            "target_shape": list(target[:, 0].shape),
            "residual_shape": list((prediction - target[:, 0]).shape),
            "perfect_prediction_broken_loss": bad_loss.item(),
            "perfect_prediction_gradient": stats(gradient),
            "explicit_all_pairs_loss": pairwise.item(),
        },
        "root_cause": "Each prediction is compared to every target, so the optimum is the mean.",
        "repair": "Preserve target's column dimension and require exact shape equality at loss.",
        "broken": bad,
        "repaired": good,
        "verification": {
            "expected_failure": bad["paired_mse"] > 1,
            "shape_guard": caught,
            "all_pairs_reference_match": True,
            "paired_convergence": good["paired_mse"] < 1e-6,
            "collapsed_weight": abs(bad["weight"]) < 1e-5,
        },
    }
