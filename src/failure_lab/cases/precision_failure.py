"""Explicit CPU float16 simulation of an auxiliary squared-residual loss."""

import torch
from torch import Tensor


def fixture() -> tuple[Tensor, Tensor]:
    return torch.tensor([300.0, -300.0, 400.0, -400.0], dtype=torch.float16), torch.zeros(
        4, dtype=torch.float16
    )


def broken(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def repaired(prediction: Tensor, target: Tensor) -> Tensor:
    # Promote before subtraction and squaring; promoting only the reduction is too late.
    return (prediction.float() - target.float()).square().mean()


def run() -> dict:
    from failure_lab.diagnostics import scalar, stats

    p, target = fixture()
    p.requires_grad_()
    bad, good = broken(p, target), repaired(p, target)
    (bad_grad,) = torch.autograd.grad(bad, p)
    (good_grad,) = torch.autograd.grad(good, p)
    reference_p = p.detach().double().requires_grad_()
    reference = torch.nn.functional.mse_loss(reference_p, target.double())
    (reference_grad,) = torch.autograd.grad(reference, reference_p)
    torch.testing.assert_close(good.double(), reference)
    torch.testing.assert_close(good_grad.double(), reference_grad)
    return {
        "symptom": "Finite float16 residuals produce infinite squares and loss.",
        "hypothesis": "Elementwise squaring exceeds float16's finite range before reduction.",
        "evidence": {
            "execution": "CPU explicit float16; not CUDA/MPS autocast",
            "float16_max": torch.finfo(torch.float16).max,
            "input": stats(p),
            "squared_residuals": stats((p - target).square()),
            "cast_after_square_loss": scalar((p - target).square().float().mean().item()),
            "float64_reference_loss": reference.item(),
        },
        "root_cause": "300^2 and 400^2 exceed 65504; float32 reduction cannot recover infinities.",
        "repair": "Promote loss operands to float32 before subtraction, square, and reduction.",
        "broken": {"loss": scalar(bad.item()), "gradients": stats(bad_grad)},
        "repaired": {"loss": good.item(), "gradients": stats(good_grad)},
        "verification": {
            "expected_failure": bool(torch.isinf(bad)),
            "finite_repair": bool(torch.isfinite(good)) and bool(torch.isfinite(good_grad).all()),
            "float64_reference_match": True,
        },
    }
