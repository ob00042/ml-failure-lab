import torch

from failure_lab.cases.exploding_gradients import dataset, train


def test_divergence_matches_curvature_and_retains_failure_step():
    x, _ = dataset()
    hessian = 2 * x.square().mean().item()
    bad = train(0.1)
    assert abs(1 - 0.1 * hessian) > 1
    assert not bad["finite_training"]
    assert bad["history"][-1]["loss"] == "inf"
    # Early loss growth follows the exact quadratic recurrence, not a guessed threshold.
    ratio = bad["history"][1]["loss"] / bad["history"][0]["loss"]
    torch.testing.assert_close(torch.tensor(ratio), torch.tensor((1 - 0.1 * hessian) ** 2))


def test_corrected_lr_converges_without_clipping():
    good = train(0.01)
    assert good["finite_training"]
    assert len(good["history"]) == 40
    assert max(r["grad_norm"] for r in good["history"]) < 207
    assert good["final_loss"] < 1e-8
    assert good["final_loss"] < good["history"][0]["loss"] * 1e-6
    torch.testing.assert_close(torch.tensor(good["final_weight"]), torch.tensor(3.0))
