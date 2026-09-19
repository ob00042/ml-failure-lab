import pytest
import torch
import torch.nn.functional as F

from failure_lab.cases.unstable_loss import broken, fixture, repaired


def test_expected_failure_and_underflow_evidence():
    x, y = fixture()
    x.requires_grad_()
    loss = broken(x, y)
    loss.backward()
    assert torch.isinf(loss)
    assert torch.isnan(x.grad).all()
    assert (x.softmax(-1) == 0).any()
    assert (x.detach().double().softmax(-1) > 0).all()


@pytest.mark.parametrize("scale", [1.0, 100.0, 1000.0])
def test_repair_matches_reference_outputs_and_gradients(scale):
    generator = torch.Generator().manual_seed(42)
    x = (torch.randn(8, 5, generator=generator) * scale).requires_grad_()
    y = torch.arange(8) % 5
    actual = repaired(x, y)
    expected = F.cross_entropy(x, y)
    (grad,) = torch.autograd.grad(actual, x)
    (reference_grad,) = torch.autograd.grad(expected, x)
    assert torch.isfinite(actual)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad, reference_grad)
