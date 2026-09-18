#!/bin/bash
# Route A step 1 (meeting 2026-09-18): extend the five affected-shadow
# baselines to E_max and log each epoch.  No automatic E* or spectrum scan.
#
# Run from the Route A worktree root (e.g. ~/oct2-exp-a) on a Delta login node:
#   export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4      # frozen original tree (read only)
#   export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A          # must end in /A
#   export OCT_DATA_DIR=/u/yli103/oct2/data
#   bash scripts/cross_stage/submit_A_step1.sh               # optional: OCT_ACCOUNT, OCT_A_EPOCHS (default 100)
#
# Submits a GPU test gate, then an array (afterok) with one task per affected-shadow seed. Nothing is overwritten.
set -euo pipefail
: "${OCT_BASELINE_ROOT:?}"; : "${OCT_RUNS_ROOT:?}"; : "${OCT_DATA_DIR:?}"
ACCOUNT="${OCT_ACCOUNT:-bgjy-delta-gpu}"
export OCT_A_EPOCHS="${OCT_A_EPOCHS:-100}"
case "$OCT_A_EPOCHS" in ''|*[!0-9]*) echo "OCT_A_EPOCHS must be an integer" >&2; exit 2 ;; esac
if [ "$OCT_A_EPOCHS" -le 50 ]; then
  echo "OCT_A_EPOCHS must extend beyond the frozen 50-epoch endpoint" >&2; exit 2
fi
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Tracked files are modified; commit or restore them before submitting." >&2; exit 2
fi
CODE="$(git rev-parse HEAD)"
export OCT_A_CODE="$CODE"
mkdir -p logs "$OCT_RUNS_ROOT"
PREFLIGHT="$(mktemp)"
if ! python3 experiments/pathway2/preflight.py verify --baseline-root "$OCT_BASELINE_ROOT" \
     --output-root "$OCT_RUNS_ROOT" > "$PREFLIGHT"; then
  cat "$PREFLIGHT" >&2; echo "Route A preflight failed; nothing submitted." >&2; exit 2
fi
FULL="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/shadow3_full"
N_RUNS="$(python3 - "$FULL/experiment_config.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["args"]
if a["seeds"] != [42, 43, 44, 45, 46]:
    raise SystemExit(f"Unexpected frozen Stage-1 seed panel: {a['seeds']}")
print(len(a["seeds"]))
PY
)"
STAMP="$(date +%Y%m%dT%H%M%S)"
export OCT_A_OUT="$OCT_RUNS_ROOT/convergence_E${OCT_A_EPOCHS}_${STAMP}"
if [ -e "$OCT_A_OUT" ]; then echo "$OCT_A_OUT already exists" >&2; exit 2; fi
mkdir -p "$OCT_A_OUT"
mv "$PREFLIGHT" "$OCT_A_OUT/preflight.json"
SUB="$OCT_A_OUT/submission.txt"
{ echo "CODE=$CODE"; echo "OUT=$OCT_A_OUT"; echo "EPOCHS=$OCT_A_EPOCHS"; echo "N_RUNS=$N_RUNS"; } > "$SUB"
TEST="$(sbatch --parsable --account="$ACCOUNT" scripts/cross_stage/20_A_tests.slurm)"
TEST="${TEST%%;*}"
echo "GPU_TEST=$TEST" >> "$SUB"
ARRAY="$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$TEST" --kill-on-invalid-dep=yes \
  --array="0-$((N_RUNS - 1))%4" scripts/cross_stage/20_A_convergence.slurm)"
ARRAY="${ARRAY%%;*}"
echo "CONV_ARRAY=$ARRAY" >> "$SUB"
cat "$SUB"
cat <<MSG

Monitor:
  sacct -X -j $TEST,$ARRAY --format=JobID%18,JobName%14,State,ExitCode,Elapsed,NodeList
After every task is COMPLETED 0:0:
  module load pytorch-conda/2.8
  python3 scripts/cross_stage/20_A_convergence_report.py --root "$OCT_A_OUT"
  # figures and descriptive REPORT.md land in $OCT_A_OUT/report
  # The report never selects E*. Review all five curves first.
MSG
