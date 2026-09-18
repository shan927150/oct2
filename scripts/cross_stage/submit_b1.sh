#!/bin/bash
# Submit only. The B1 array waits for a successful GPU regression job; if the tests fail
# the array is killed rather than left queued against untested code.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OCT_B_CODE_ROOT="$PWD" OCT_B_COMMIT="$(git rev-parse HEAD)"
export OCT_BASELINE_ROOT="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
export OCT_RUNS_ROOT="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$OCT_BASELINE_ROOT" --output-root "$OCT_RUNS_ROOT"
mkdir -p logs "$OCT_RUNS_ROOT"
TEST=$(sbatch --parsable scripts/cross_stage/16_b1_tests.slurm)
TEST=${TEST%%;*}
B1=$(sbatch --parsable --dependency="afterok:$TEST" --kill-on-invalid-dep=yes \
     scripts/cross_stage/16_b1_smoke.slurm)
B1=${B1%%;*}
printf 'GPU_TEST=%s\nB1_ARRAY=%s\nCODE=%s\n' "$TEST" "$B1" "$OCT_B_COMMIT" \
  | tee "$OCT_RUNS_ROOT/submission_b1_${TEST}.txt"
