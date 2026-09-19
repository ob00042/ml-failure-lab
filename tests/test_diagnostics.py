import math

import pytest
import torch

from failure_lab.diagnostics import norm, scalar, stats


def test_stats_separate_nonfinite_values():
    result = stats(torch.tensor([1.0, float("nan"), float("inf"), -float("inf")]))
    assert result["finite_count"] == 1
    assert result["nan_count"] == 1
    assert result["inf_count"] == 2
    assert not result["all_finite"]
    assert result["min_finite"] == result["max_finite"] == 1
    assert stats(torch.tensor([]))["min_finite"] is None


def test_norm_does_not_overflow_when_float32_square_would():
    value = norm([torch.tensor([1e20, 1e20])])
    assert math.isfinite(value)
    assert value == pytest.approx(math.sqrt(2) * 1e20)


@pytest.mark.parametrize(
    "value, expected",
    [(float("nan"), "nan"), (float("inf"), "inf"), (-float("inf"), "-inf"), (1.5, 1.5)],
)
def test_json_scalar_encoding(value, expected):
    assert scalar(value) == expected
