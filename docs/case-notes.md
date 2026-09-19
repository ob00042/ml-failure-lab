# Diagnostic case notes

## 1. Unstable loss

**Symptom.** Confident wrong predictions with logits `[100, -100]` produce infinite negative log likelihood and NaN derivatives in float32.

**Hypothesis.** Probability materialization loses representable log-probabilities through underflow. This is not exponential overflow inside PyTorch softmax, which already stabilizes its exponentiation.

**Evidence.** The unlikely probability is zero in float32 but about `1.3838965e-87` in float64. `exp(-200)` is below the float32 subnormal range. Taking log after underflow yields negative infinity; the resulting chain rule contains undefined numerical operations.

**Root cause.** The intermediate representation, not the mathematical loss, is out of range.

**Repair.** Replace `softmax(...).log()` with `log_softmax(...)`. No epsilon or probability clamp is needed. Clamping would alter the loss and gradients for the hardest examples.

**Verification.** Saved loss is 200 with finite gradients. Tests compare outputs and gradients against `torch.nn.functional.cross_entropy` across multiple logit scales, and explicitly reproduce the original non-finite result. Finite input logits are assumed; log-softmax cannot repair NaNs produced upstream.

## 2. Exploding gradients

**Symptom.** Full-batch linear regression on `x ∈ [-10, 10]`, `y = 3x`, from weight zero diverges at learning rate 0.1. The first gradient norm is about 206.35; loss reaches infinity at step 23, with gradient norm about `1.0172e20`.

**Hypothesis.** An unchanged learning rate becomes unstable when features change units by a factor of ten.

**Evidence.** For `L(w) = mean((wx - 3x)^2)`, the Hessian is `H = 2 mean(x²) ≈ 68.7831`. The error recurrence is `e_next = (1 - lr H)e`. Stable constant-step gradient descent requires `0 < lr < 2/H ≈ 0.0290769`. At 0.1 the error multiplier is approximately -5.8783: it changes sign and grows. The test checks the early loss ratio against this squared multiplier. For features divided by ten, the stability ceiling is about 2.90769.

**Root cause.** Feature scaling changes curvature quadratically, placing the chosen learning rate outside the stability interval. No randomness, minibatch noise, or deep-network effects are needed to explain it.

**Repair.** Lower the learning rate to 0.01, giving an error multiplier about 0.31217. Keep data, initialization, objective, and update rule unchanged. Clipping is intentionally unnecessary here. It could bound updates but would change the dynamics without correcting the underlying step-size choice.

**Verification.** Forty repaired updates stay finite, with gradient norms bounded by the initial norm, and reach the exact target weight in local float32 arithmetic. Tests require MSE below `1e-8`, not an exact zero. Trace entries measure loss, gradient norm, and parameter norm before each update; the first non-finite step is retained and stops the broken run. Final loss is evaluated after the last applied update. Norms accumulate in float64 to avoid diagnostic overflow.

## 3. Silent broadcasting

**Symptom.** Perfect predictions receive loss about 2.83871 and nonzero gradients. Training remains finite but learns a near-zero slope and bias 0.5, instead of slope 2.

**Hypothesis.** A target squeeze has turned paired loss into all-pairs loss.

**Evidence.** `[32, 1] - [32]` broadcasts to `[32, 32]`, because the target aligns with the last axis. An explicit double-loop reference gives the same wrong loss. With perfect predictions this objective equals twice the population variance of the target, rather than zero.

**Root cause.** The objective is `sum_ij (p_i - y_j)^2 / N²`, minimized by making every prediction equal the target mean. PyTorch accepts the shapes because they satisfy its documented broadcasting semantics; a raw subtraction has no paired-regression contract to enforce.

**Repair.** Preserve the `[N, 1]` target at the data boundary and require exact shape equality before computing the loss. The loss does not silently reshape arbitrary inputs, since doing so can hide a separate data error.

**Verification.** Shape mismatch tests cover several broadcast-compatible pairs. Correct loss and gradients match PyTorch MSE. Training reaches paired MSE around `3.01e-8`, compared with 1.41935 for the broken model. A batch of one can conceal this bug, so the regression uses multiple distinct targets.

## 4. Reduced precision

**Symptom.** Float16 residuals 300 and 400 are finite, but squaring them gives infinity. The mean loss is infinity even though the four derivatives remain finite.

**Hypothesis.** Elementwise arithmetic overflows before the reduction.

**Evidence.** Float16's largest finite value is 65504. Both 90000 and 160000 exceed it. Casting the already-squared values to float32 still gives infinity; casting operands first gives 125000. Finite derivatives are expected here: the derivative of squaring is computed from its finite input, and the mean divides the gradient by four.

**Root cause.** A manually written auxiliary loss inherits a dtype with insufficient range for its intermediate values.

**Repair.** Promote prediction and target to float32 before subtraction, squaring, and reduction. Casting only the mean is too late. This also protects subtraction for oppositely signed large finite inputs.

**Verification.** Compare loss and gradients to float64 MSE on the same already-quantized inputs. Tests cover small residuals, overflowing squares, and overflowing subtraction. This is explicit CPU arithmetic, not an accelerator/autocast benchmark. In an actual mixed-precision pipeline, autocast policies vary by operation; inspect the executed dtype. Loss scaling is not a repair for overflow that already occurred in the forward pass. A wider loss also cannot guarantee that gradients remain in float16 range for every input.
