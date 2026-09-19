import pytest
import torch
import torch.nn.functional as F

from failure_lab.cases.broadcasting_bug import broken, fixture, repaired, train


def test_perfect_prediction_is_penalized_by_broadcasting():
    _, target = fixture()
    p = target.clone().requires_grad_()
    loss = broken(p, target[:, 0])
    assert (p - target[:, 0]).shape == (32, 32)
    assert loss > 1
    (grad,) = torch.autograd.grad(loss, p)
    assert grad.norm() > 0.1
    torch.testing.assert_close(loss, 2 * target.var(unbiased=False))
    torch.testing.assert_close(repaired(p, target), torch.tensor(0.0))


@pytest.mark.parametrize("shapes", [((4, 1), (4,)), ((1, 4), (4,)), ((4, 2), (4, 1))])
def test_shape_guard_rejects_broadcast_compatible_inputs(shapes):
    with pytest.raises(ValueError, match="Exact shapes required"):
        repaired(torch.ones(shapes[0]), torch.ones(shapes[1]))


def test_repair_matches_paired_reference_and_gradients():
    p = torch.tensor([[1.0], [3.0]], requires_grad=True)
    y = torch.tensor([[2.0], [0.0]])
    actual, expected = repaired(p, y), F.mse_loss(p, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0]
    )


def test_correct_pairing_learns_relationship():
    bad, good = train(False), train(True)
    assert bad["paired_mse"] > 1
    assert abs(bad["weight"]) < 1e-5
    assert good["paired_mse"] < 1e-6
    assert abs(good["weight"] - 2) < 0.001
