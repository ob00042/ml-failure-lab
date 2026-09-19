"""A confident wrong prediction exposes probability-space underflow."""

import torch
from torch import Tensor


def broken(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=-1)
    return -probabilities.log().gather(1, targets[:, None]).mean()


def fixture() -> tuple[Tensor, Tensor]:
    return torch.tensor([[100.0, -100.0], [-100.0, 100.0]]), torch.tensor([1, 0])


def repaired(logits: Tensor, targets: Tensor) -> Tensor:
    return -logits.log_softmax(dim=-1).gather(1, targets[:, None]).mean()


def run() -> dict:
    from failure_lab.diagnostics import scalar, stats

    x, y = fixture()
    x.requires_grad_()
    bad = broken(x, y)
    (bad_grad,) = torch.autograd.grad(bad, x)
    good = repaired(x, y)
    (good_grad,) = torch.autograd.grad(good, x)
    reference = torch.nn.functional.cross_entropy(x, y)
    (reference_grad,) = torch.autograd.grad(reference, x)
    torch.testing.assert_close(good, reference)
    torch.testing.assert_close(good_grad, reference_grad)
    return {
        "symptom": "Infinite loss and NaN gradients for confident wrong predictions.",
        "hypothesis": "Float32 probability-space softmax underflows before log is applied.",
        "evidence": {
            "probabilities": stats(x.softmax(-1)),
            "zero_probabilities": int((x.softmax(-1) == 0).sum()),
            "float64_min_probability": float(x.detach().double().softmax(-1).min()),
        },
        "root_cause": "exp(-200) is below float32 range; log(0) is -inf.",
        "repair": "Compute log_softmax directly, avoiding probability materialization.",
        "broken": {"loss": scalar(bad.item()), "gradients": stats(bad_grad)},
        "repaired": {"loss": good.item(), "gradients": stats(good_grad)},
        "verification": {
            "expected_failure": not bool(torch.isfinite(bad)),
            "finite_repair": bool(torch.isfinite(good)) and bool(torch.isfinite(good_grad).all()),
            "reference_match": True,
        },
    }
