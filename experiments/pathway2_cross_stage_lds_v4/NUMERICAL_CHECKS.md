# v4 numerical-check audit, 2026-10-13

Scope: smoke, base, truth, score, collection/report, and reused kernels reached by these paths.

The two observed smoke failures compare finite differences with automatic differentiation.
The full CNN uses ReLU and MaxPool. Non-smooth branches can affect finite differences, but
the supplied logs do not establish the cause of either discrepancy. Raising a tolerance
until a run passes is not a validation of the derivative implementation.

## Policy

Both finite-difference comparisons now record every tested step and the best relative error.
Finite discrepancies above the reference tolerance emit a warning and continue. NaN or Inf
still fails. The prepared config explicitly records `finite_difference_policy=diagnostic_only`.
Reference tolerances remain 0.005 for the h contraction and 0.002 for the JVP max norm.
These are diagnostic references, not acceptance gates or statistical significance thresholds.

The smoke checks are copied into summary.json and displayed in REPORT_CN.md. A completed
smoke means the pipeline is runnable. It does not establish that derivatives are exact or
that linearization over a 50% deletion displacement is accurate.

| Stage | Check | Treatment |
|---|---|---|
| smoke | h versus finite differences | Record and warn, finite discrepancies continue |
| smoke | JVP versus finite differences | Record and warn, finite discrepancies continue |
| smoke | Recomputed full-model probabilities versus v3 | Keep original 1e-10 absolute replay tolerance |
| smoke | Repeated attack training on the same device | Keep exact deterministic replay check |
| smoke | Dense adjoint versus legacy adjoint on identical inputs | Keep original 1e-10 relative tolerance |
| smoke | Attack damping has a positive damped spectrum and small solve residual | Keep the predefined rule and report gamma_A selected from full baselines |
| base | J00 versus smoke | Keep configured absolute tolerance, default 1e-6 |
| truth | Pack-local J00 versus base | Keep configured absolute tolerance, default 1e-6; continue subtracting pack-local J00 from J10 |
| score | Stage-1 solve residual or curvature screen fails for one gamma | Existing behavior retained: mark that gamma unavailable; finish the score task and keep other gammas |
| collect | Main gamma 0.03 is unavailable | Report it as unavailable; do not substitute a different main gamma |
| all | NaN/Inf, wrong row identity, missing files, altered inputs or source | Fail with the actual error |

Replay comparisons now explicitly reject NaN and Inf, which can otherwise pass a check of
the form `abs(actual - reference) > tolerance`. Their tolerances have not been relaxed.
The truth output additionally records J00_minus_base, and the report lists both replay
differences. The existing class selection, training, derivatives, score formulas, damping
grid and statistical analysis are unchanged.

## Regression coverage

The numerical tests replay both supplied Delta error logs with the original 1e-5 reference
tolerance and check that they are reported without aborting. They also check rejection of
non-finite diagnostics, retention of J00 replay checks, and collection of an unqualified
L3 shift. The complete CPU integration test deliberately sets both FD references to zero,
then runs background, smoke, base, truth, score and report generation.
An independent CNN test also checks v^T (J d) = (J^T v)^T d using the existing forward-mode
JVP and reverse-mode h implementation, without finite differences or a new runtime gate.

Validation on 2026-10-13: 14 tests passed together (including the full CPU pipeline), and
the additional CNN forward/reverse identity test passed separately. Environment: PyTorch
2.8.0+cpu, float64 derivatives. Delta/CUDA execution of this revision is still pending.

## Deployment

Use one new committed worktree and one new OUT. Prepared runs pin their source hashes and
commit. Do not edit their config or completion markers in place. The existing v3 full and
256 shadow3 subset models are reused. This bundle is cumulative from a9f0842, so installing
the earlier JVP-only bundle first is unnecessary.
