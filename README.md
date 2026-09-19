# ml-failure-lab

ML systems often fail without throwing useful exceptions. This repository contains reproducible examples of numerical and semantic failures, the instrumentation used to diagnose them, and regression tests that verify each repair.

Four small CPU experiments isolate realistic failure mechanisms. Each has a broken computation, a falsifiable diagnosis, a minimal repair, and measured evidence. Synthetic data keeps the causal argument inspectable; no datasets or accelerators are required.

## Failure cases

| Case | Symptom | Root cause | Repair | Verification |
| --- | --- | --- | --- | --- |
| `unstable-loss` | Infinite loss, NaN gradients | Float32 softmax underflows before taking log | Compute log-softmax directly | Finite loss/gradients; cross-entropy reference agreement |
| `exploding-gradients` | Growing norms, then infinite loss | Feature scale makes the learning rate exceed the quadratic stability interval | Correct the learning rate | Predicted early growth; bounded gradients; convergence |
| `broadcasting-bug` | Finite training converges to the wrong predictor | `[N, 1] - [N]` computes all pairs | Preserve target dimension; reject unequal shapes | Shape regression tests; paired MSE and learned slope |
| `precision-failure` | Infinite loss despite finite gradients | Float16 squared residuals exceed 65504 | Promote operands before loss arithmetic | Finite result/gradients; float64 reference agreement |

## Diagnostic methodology

Every report separates **symptom → hypothesis → evidence → root cause → repair → verification**. The experiments change one causal factor at a time. Loss, gradients, parameter norms, intermediate tensor statistics, and semantic references provide different kinds of evidence; none alone establishes correctness.

See [case notes](docs/case-notes.md) for equations, competing explanations, and repair limits. The implementation is in [`src/failure_lab/cases`](src/failure_lab/cases), with shared diagnostics in [`diagnostics.py`](src/failure_lab/diagnostics.py). Full training traces are included for the exploding-gradient case.

## Measured results

These are actual local CPU measurements from Python 3.13.7, PyTorch 2.14.0, NumPy 2.5.3 on Apple Silicon macOS. CUDA and MPS both reported unavailable in the measuring process (MPS support was built into PyTorch). Each [baseline JSON](results/baseline) records its environment, configuration evidence, and verification outcomes.

| Measurement | Broken | Repaired |
| --- | ---: | ---: |
| Unstable classification loss | `inf`; 4/4 gradients NaN | 200; 4/4 gradients finite |
| Regression with unstable updates | `inf` at step 23 (zero-based) | Final MSE 0 after 40 updates |
| Broadcasting: correctly paired evaluation MSE | 1.41935 | 3.01236e-8 after 120 updates |
| Float16 residual loss | `inf`; 4/4 gradients finite | 125000; matches float64 |

Zero MSE here is a measured float32 result for a tiny exactly representable linear problem, not a general training guarantee. Tests use tolerant convergence criteria instead of requiring exact training losses.

## Reproduce

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m failure_lab.runner list
python -m failure_lab.runner run unstable-loss
python -m failure_lab.runner run exploding-gradients
python -m failure_lab.runner run broadcasting-bug
python -m failure_lab.runner run precision-failure
python -m failure_lab.runner run all
```

All commands explicitly use CPU tensors. Runs seed Python, NumPy, and PyTorch with 7, enable deterministic algorithms, and use one PyTorch thread. The current fixtures are deterministic and do not require random sampling. Local measurements are not GPU benchmarks.

Each run prints broken behavior, diagnosis, repair, and verification, then writes `results/<case>.json`. Repeating a run overwrites that file. Use `--output-dir path/to/run` to keep separate runs. Baselines are committed; routine outputs are ignored. Non-finite numbers are encoded as strings (`"inf"`, `"-inf"`, `"nan"`) so the reports are strict JSON. A failed verification exits nonzero; expected broken behavior is a passing condition only when it matches the case's expectations.

The supported dependency ranges are in `pyproject.toml`. [`results/baseline/requirements-local.txt`](results/baseline/requirements-local.txt) records the exact local environment for provenance; it is not a portable lockfile or a claim that every supported version was tested.

## Testing

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
# Or all validation and CPU experiments:
make check
```

Tests prove the broken conditions, numerical/reference agreement, finite repaired gradients, analytic loss growth, silent shape rejection, training convergence, and CLI report/exit behavior. CLI tests name the expected cases independently of the runner registry. PyTorch reference comparisons use `torch.testing.assert_close`.

GitHub Actions installs CPU PyTorch and runs Ruff, pytest, and all four cases on Python 3.11 and 3.13. That workflow is supplied here; a hosted GitHub run has not been observed during local development.

## Docker

```bash
docker build -t ml-failure-lab .
docker run --rm ml-failure-lab python -m pytest -q
docker run --rm -v "$PWD/results:/app/results" ml-failure-lab
```

The image installs PyTorch from the CPU wheel index and defaults to all cases. It includes development tools to run tests. Docker validation status is recorded in [the workflow log](docs/agent-workflow.md).

## Engineering observations

- Stable softmax alone does not make `log(softmax(x))` stable: tiny probabilities can still become zero.
- Learning-rate stability depends on feature scale and curvature. Gradient clipping would cap updates but does not explain or remove the excessive learning rate in this example.
- Broadcasting is useful tensor behavior; shape compatibility is not semantic compatibility. Check the contract at the loss boundary.
- A finite backward pass cannot validate a non-finite forward pass. Promotion must happen before the overflowing operation.
- Keep diagnostics numerically safer than the workload: norms accumulate in float64.

## Limitations

These are bounded synthetic reproductions, not production incident reports, model quality benchmarks, or proofs about arbitrary networks. The gradient case is deliberately a convex linear model so the diagnosis has an analytic check. The precision case explicitly simulates float16 arithmetic on CPU; it does not claim to reproduce CUDA autocast policy or accelerator performance. Float32 loss computation cannot restore information already lost upstream, and gradients cast back to float16 can still overflow at larger scales. Other dtypes, stochastic training, and distributed execution need separate validation.

## Agent-assisted workflow

[Development log](docs/agent-workflow.md) records the tasks, hypotheses, executed experiments, and real integration/test mistakes encountered. Codex performed the implementation and inspections recorded here. No independent human code review is claimed.

MIT licensed. Contributions should include a distinct failure mechanism, diagnostic evidence, a minimal repair, and a regression test rather than merely increasing the case count.
