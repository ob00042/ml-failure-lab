# Agent-assisted development record

This record describes work actually performed in this Codex session. The user supplied the project brief and requested sequential case development. Codex implemented and inspected the code, executed experiments, and reviewed tool output.

## Environment and scope (Initial Check)

- **Task:** Inspect the workspace before implementation; build a compact CPU debugging lab.
- **Proposal:** Use an isolated virtual environment and four independent, analytically understandable cases.
- **Inspection:** Empty workspace, no Git repository; Python 3.13.7 on macOS 26.3.1 arm64. The default interpreter had none of torch, NumPy, pytest, or Ruff. After installation: PyTorch 2.14.0, CUDA unavailable, MPS built but unavailable in the process.
- **Outcome:** CPU is the measured execution target. Git and a local virtual environment were created. No fifth case was added.
- **Operational events:** Initial dependency download could not resolve hosts inside the sandbox, and Git writes required escalation. Authorized tool retries completed these operations; those restrictions were not ML failures.

## First case: probability-space loss

- **Task:** Finish one case before starting the others.
- **Proposal/hypothesis:** `log(softmax)` underflows on large logit gaps; log-space evaluation preserves the mathematical loss.
- **Inspection:** First implemented only the broken loss and fixture, then ran backward and inspected probabilities, loss, and gradients. Float32 gave zero probabilities, infinite loss, and four NaNs. Float64 probabilities were nonzero.
- **Diagnosis:** Supported. This isolates probability underflow rather than softmax exponential overflow.
- **Confirmation:** Added diagnostics, direct log-softmax repair, and tests against cross entropy for both outputs and gradients at several scales. Four tests passed, then a baseline JSON was saved before the second case began.

## Second case: gradient explosion

- **Task:** Demonstrate unstable optimization with meaningful gradient and parameter instrumentation.
- **Proposal/hypothesis:** A learning rate left unchanged after feature scaling exceeds the quadratic stability limit.
- **Inspection:** Ran the broken training loop before adding its repaired experiment. Inspected Hessian, stability bound, first trace row, and failure row. Loss became infinite at step 23, while gradients were still finite.
- **Diagnosis:** Supported by the measured curvature and analytic error recurrence. There was no need to introduce clipping as a generic repair.
- **Confirmation:** Learning rate 0.01 converged; early broken loss growth matched the recurrence. Six cumulative tests passed and a full before/after trace was saved before proceeding.

## Third case: semantic broadcasting

- **Task:** Reproduce a silent correctness failure.
- **Proposal/hypothesis:** A squeezed target creates an all-pairs objective with a constant-prediction optimum.
- **Inspection:** Ran the broken loss on perfect predictions; observed `[32, 32]` residuals, loss 2.83871, and nonzero gradient norm. Compared with correctly paired loss zero.
- **Diagnosis:** Supported by an explicit all-pairs reference and the learned near-zero slope under the broken objective.
- **Confirmation:** Added shape validation and correct target layout. Twelve cumulative tests passed. Repaired training reached paired MSE about `3.01e-8`.
- **Actual integration mistake:** A scripted registry edit assumed a trailing comma that was not present in the formatted one-line dictionary. Module tests passed, but the CLI rejected `broadcasting-bug`. Inspected the runner, corrected registration, and reran successfully. Later CLI tests explicitly enumerate all four expected names independently of the registry.

## Fourth case: reduced precision

- **Task:** Show a credible precision failure without inventing accelerator results.
- **Proposal/hypothesis:** Float16 squaring overflows before a loss reduction; promotion must occur earlier.
- **Inspection:** Ran explicit CPU float16 arithmetic on finite residuals. Squared values and loss were infinite. Gradients were unexpectedly informative: all remained finite. Casting after the square was still infinite; casting before it yielded 125000.
- **Diagnosis:** Supported. Finite derivatives are consistent with backward using the finite input of the square; they do not certify the forward pass.
- **Confirmation:** Repaired outputs and derivatives matched float64 MSE on the same quantized inputs. Tests also cover subtraction overflow. Seventeen cumulative tests passed and the precision baseline was saved.

## Integration and validation

- **Task:** Test CLI behavior, report validity, diagnostic robustness, and CPU packaging.
- **Proposal:** Strict JSON reports, explicit verification outcomes, CPU CI, and a simple Docker image.
- **Inspection:** Added CLI enumeration/run-all tests, a failed-verification test, strict JSON parsing, and a diagnostic norm-overflow test.
- **Actual test-isolation mistake:** The first failed-verification test monkeypatched `importlib.import_module` on the shared module. That intercepted PyTorch's lazy imports during deterministic setup and raised `TypeError` before reaching the intended assertion. The traceback exposed the mistake. Changed the runner to import the function locally and patched only that runner binding. The suite then passed all 27 tests.
- **Confirmation:** Local `make check` runs Ruff, format checks, all tests, and all CPU cases. Docker build and execution are verified separately below. 

## Final observed validation

- Local `make check PYTHON=.venv/bin/python`: Ruff and format checks passed; **27 tests passed**; all four CPU cases reported every verification true.
- `docker build -t ml-failure-lab .`: succeeded using `python:3.13-slim` on Linux arm64 and PyTorch `2.14.0+cpu`.
- `docker run --rm ml-failure-lab python -m pytest -q`: **27 tests passed**.
- `docker run --rm ml-failure-lab`: all four default-command experiments completed with every verification true.
- Minor platform-dependent rounding was observed: the first gradient norm was 206.349182 on macOS versus 206.349213 in Linux. Both satisfy the analytic/tolerance checks. Deterministic execution within an environment does not imply bitwise equality across platforms.
