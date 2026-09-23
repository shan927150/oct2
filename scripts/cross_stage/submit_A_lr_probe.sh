#!/bin/bash
# Route A convergence-control probe (baselines only): 5 seeds x {control, decay},
# each continued from its Step-1 E100 checkpoint with paired state, orders and RNG.
#
#   export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4
#   export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A
#   export OCT_DATA_DIR=/u/yli103/oct2/data
#   export OCT_A_STEP1_ROOT=$HOME/oct2-pathway2-runs/A/convergence_E100_20260918T111502
#   bash scripts/cross_stage/submit_A_lr_probe.sh "advisor approved the decay probe, <date/channel>"
#
# The decay arm changes the training recipe, so a non-empty human approval note
# is mandatory and is stored in submission.txt. No LOO, attack, truth or damping
# job is submitted here. Optional: OCT_ACCOUNT, OCT_A_RESUME (100),
# OCT_A_EXTRA (40), OCT_A_DECAY (20@1e-4,20@1e-5).
set -euo pipefail
NOTE="${1:-}"
if [ -z "$NOTE" ]; then
  echo "Usage: submit_A_lr_probe.sh \"who approved the decay-arm recipe and where\"" >&2
  exit 2
fi
: "${OCT_BASELINE_ROOT:?}"; : "${OCT_RUNS_ROOT:?}"; : "${OCT_DATA_DIR:?}"; : "${OCT_A_STEP1_ROOT:?}"
ACCOUNT="${OCT_ACCOUNT:-bgjy-delta-gpu}"
export OCT_A_RESUME="${OCT_A_RESUME:-100}"
export OCT_A_EXTRA="${OCT_A_EXTRA:-40}"
export OCT_A_DECAY="${OCT_A_DECAY:-20@1e-4,20@1e-5}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Tracked files are modified; commit or restore them before submitting." >&2; exit 2
fi
export OCT_A_CODE="$(git rev-parse HEAD)"
for seed in 42 43 44 45 46; do
  CK="$OCT_A_STEP1_ROOT/shadow3_seed${seed}/checkpoints/epoch$(printf '%03d' "$OCT_A_RESUME").pt"
  test -f "$CK" || { echo "Missing Step-1 checkpoint: $CK" >&2; exit 2; }
done
mkdir -p logs "$OCT_RUNS_ROOT"
PREFLIGHT="$(mktemp)"
if ! python3 experiments/pathway2/preflight.py verify --baseline-root "$OCT_BASELINE_ROOT" \
     --output-root "$OCT_RUNS_ROOT" > "$PREFLIGHT"; then
  cat "$PREFLIGHT" >&2; echo "Route A preflight failed; nothing submitted." >&2; exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export OCT_A_LR_OUT="$OCT_RUNS_ROOT/lr_probe_R${OCT_A_RESUME}_X${OCT_A_EXTRA}_${STAMP}"
if [ -e "$OCT_A_LR_OUT" ]; then echo "$OCT_A_LR_OUT already exists" >&2; exit 2; fi
mkdir -p "$OCT_A_LR_OUT"
mv "$PREFLIGHT" "$OCT_A_LR_OUT/preflight.json"
SUB="$OCT_A_LR_OUT/submission.txt"
{ echo "CODE=$OCT_A_CODE"; echo "OUT=$OCT_A_LR_OUT"; echo "STEP1_ROOT=$OCT_A_STEP1_ROOT";
  echo "RESUME=$OCT_A_RESUME"; echo "EXTRA=$OCT_A_EXTRA"; echo "DECAY=$OCT_A_DECAY";
  echo "APPROVAL_NOTE=$NOTE"; } > "$SUB"
TEST="$(sbatch --parsable --account="$ACCOUNT" scripts/cross_stage/23_A_lr_tests.slurm)"
TEST="${TEST%%;*}"
echo "GPU_TEST=$TEST" >> "$SUB"
ARRAY="$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$TEST" --kill-on-invalid-dep=yes \
  --array="0-9%4" scripts/cross_stage/23_A_lr_probe.slurm)"
ARRAY="${ARRAY%%;*}"
echo "PROBE_ARRAY=$ARRAY" >> "$SUB"
cat "$SUB"
cat <<MSG

Monitor:
  sacct -X -j $TEST,$ARRAY --format=JobID%18,JobName%16,State,ExitCode,Elapsed,NodeList
After all ten tasks are COMPLETED 0:0:
  module load pytorch-conda/2.8
  python3 scripts/cross_stage/23_A_lr_report.py --root "$OCT_A_LR_OUT"
The report is descriptive; whether to build the small-panel LOO on the decay
recipe is a separate, explicitly approved step.
MSG
