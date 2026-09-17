#!/bin/bash
# Sourced by E0/B0/test jobs after changing to the B worktree.
module reset
module load pytorch-conda/2.8
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
BASE="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
RUNS="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
DATA="${OCT_DATA_DIR:-/u/yli103/oct2/data}"
[[ -n "${OCT_B_COMMIT:-}" ]] || { echo "Use submit_e0_b0.sh to pin the code commit" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$OCT_B_COMMIT" ]] || { echo "Code changed since submission" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked code is dirty" >&2; exit 2; }
python3 -B experiments/pathway2/preflight.py verify --baseline-root "$BASE" --output-root "$RUNS"
mkdir -p "$RUNS"
pathway2_finish() {
  task_rc=$?
  trap - EXIT
  set +e
  python3 -B experiments/pathway2/preflight.py verify \
    --baseline-root "$BASE" --output-root "$RUNS" \
    > "$RUNS/postcheck_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}.json"
  audit_rc=$?
  [[ "$audit_rc" == 0 ]] || exit 91
  exit "$task_rc"
}
trap pathway2_finish EXIT
python3 -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'
