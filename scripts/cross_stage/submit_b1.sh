#!/bin/bash
# Submit only. The B1 array waits for a successful GPU regression job; if the tests fail
# the array is killed rather than left queued against untested code.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "B code has tracked edits; commit/install the reviewed version before submission" >&2
  exit 2
}
export OCT_B_CODE_ROOT="$PWD" OCT_B_COMMIT="$(git rev-parse HEAD)"
export OCT_BASELINE_ROOT="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
export OCT_RUNS_ROOT="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$OCT_BASELINE_ROOT" --output-root "$OCT_RUNS_ROOT"
mkdir -p logs "$OCT_RUNS_ROOT"
TEST=$(sbatch --parsable scripts/cross_stage/16_b1_tests.slurm)
TEST=${TEST%%;*}
[[ "$TEST" =~ ^[0-9]+$ ]] || { echo "Unexpected test job id: $TEST" >&2; exit 2; }
printf 'GPU_TEST=%s\nCODE=%s\n' "$TEST" "$OCT_B_COMMIT" \
  > "$OCT_RUNS_ROOT/submission_b1_${TEST}.txt"
echo "GPU_TEST_SUBMITTED=$TEST"
B1=$(sbatch --parsable --dependency="afterok:$TEST" --kill-on-invalid-dep=yes \
     scripts/cross_stage/16_b1_smoke.slurm)
B1=${B1%%;*}
[[ "$B1" =~ ^[0-9]+$ ]] || { echo "Unexpected array id: $B1" >&2; exit 2; }
printf 'GPU_TEST=%s\nB1_ARRAY=%s\nCODE=%s\n' "$TEST" "$B1" "$OCT_B_COMMIT" \
  | tee "$OCT_RUNS_ROOT/submission_b1_${TEST}.txt"
