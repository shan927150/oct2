#!/bin/bash
# Submit only; E0 and the two B0 tasks wait for a successful CUDA regression job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OCT_B_CODE_ROOT="$PWD" OCT_B_COMMIT="$(git rev-parse HEAD)"
export OCT_BASELINE_ROOT="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
export OCT_RUNS_ROOT="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$OCT_BASELINE_ROOT" --output-root "$OCT_RUNS_ROOT"
mkdir -p logs "$OCT_RUNS_ROOT"
TEST=$(sbatch --parsable scripts/cross_stage/13_e0_b0_tests.slurm)
TEST=${TEST%%;*}
E0=$(sbatch --parsable --dependency="afterok:$TEST" --kill-on-invalid-dep=yes scripts/cross_stage/13_e0_residual.slurm)
E0=${E0%%;*}
B0=$(sbatch --parsable --dependency="afterok:$TEST" --kill-on-invalid-dep=yes scripts/cross_stage/14_b0_dose.slurm)
B0=${B0%%;*}
printf 'GPU_TEST=%s\nE0=%s\nB0_ARRAY=%s\nCODE=%s\n' "$TEST" "$E0" "$B0" "$OCT_B_COMMIT" \
  | tee "$OCT_RUNS_ROOT/submission_${TEST}.txt"
