import pytest
import torch
import torch.nn.functional as F

from failure_lab.cases.precision_failure import broken, fixture, repaired


def test_overflow_occurs_before_reduction():
    p, y = fixture()
    p.requires_grad_()
    assert torch.isfinite(p).all()
    assert torch.isinf((p - y).square()).all()
    assert torch.isinf(broken(p, y))
    assert torch.isinf((p - y).square().float().mean())
    # A finite gradient does not imply a valid forward pass.
    assert torch.isfinite(torch.autograd.grad(broken(p, y), p)[0]).all()


@pytest.mark.parametrize("values", [[300.0, -400.0], [1.0, -2.0], [60000.0, -60000.0]])
def test_promoted_loss_matches_double_reference(values):
    p = torch.tensor(values, dtype=torch.float16, requires_grad=True)
    y = torch.zeros_like(p)
    actual = repaired(p, y)
    reference_p = p.detach().double().requires_grad_()
    reference = F.mse_loss(reference_p, y.double())
    (grad,) = torch.autograd.grad(actual, p)
    (reference_grad,) = torch.autograd.grad(reference, reference_p)
    assert torch.isfinite(actual)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(actual.double(), reference)
    torch.testing.assert_close(grad.double(), reference_grad)


def test_promote_before_subtraction_too():
    p = torch.tensor([60000.0, -60000.0], dtype=torch.float16)
    y = -p
    assert torch.isinf(p - y).all()
    assert torch.isfinite(repaired(p, y))
