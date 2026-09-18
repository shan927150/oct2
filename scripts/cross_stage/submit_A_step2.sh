#!/bin/bash
# Route A step 2 (meeting 2026-09-18): rerun the formal truth + Level 3 score at ONE plateau epoch E*,
# then the v1.1 damping sweep with smaller gamma. Existing, already validated entry points only
# (05 via 10_run_calibration, 07, 12); every Stage 1 model (target, shadows, LOO) is trained for E* epochs.
#
#   export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4
#   export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A
#   export OCT_DATA_DIR=/u/yli103/oct2/data
#   bash scripts/cross_stage/submit_A_step2.sh 70        # 70 = E* chosen from the step 1 report
set -euo pipefail
E="${1:?usage: submit_A_step2.sh E_STAR}"
: "${OCT_BASELINE_ROOT:?}"; : "${OCT_RUNS_ROOT:?}"; : "${OCT_DATA_DIR:?}"
case "$E" in ''|*[!0-9]*) echo "E_STAR must be an integer" >&2; exit 2 ;; esac
if [ "$E" -le 50 ]; then echo "E_STAR=$E does not extend the original 50 epochs" >&2; exit 2; fi
ACCOUNT="${OCT_ACCOUNT:-bgjy-delta-gpu}"
ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Tracked files are modified; commit or restore them before submitting." >&2; exit 2
fi
CODE="$(git rev-parse HEAD)"
mkdir -p logs "$OCT_RUNS_ROOT"
PREFLIGHT="$(mktemp)"
if ! python3 experiments/pathway2/preflight.py verify --baseline-root "$OCT_BASELINE_ROOT" \
     --output-root "$OCT_RUNS_ROOT" > "$PREFLIGHT"; then
  cat "$PREFLIGHT" >&2; echo "Route A preflight failed; nothing submitted." >&2; exit 2
fi
export OCT_A_STEP2_ROOT="$OCT_RUNS_ROOT/l3_at_E${E}"
if [ -e "$OCT_A_STEP2_ROOT" ]; then echo "$OCT_A_STEP2_ROOT already exists; results are never overwritten" >&2; exit 2; fi
PANEL_SRC="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json"
mkdir -p "$OCT_A_STEP2_ROOT/panel"
cp -p "$PANEL_SRC" "$OCT_A_STEP2_ROOT/panel/eligibility_preflight.json"
cmp "$PANEL_SRC" "$OCT_A_STEP2_ROOT/panel/eligibility_preflight.json"
mv "$PREFLIGHT" "$OCT_A_STEP2_ROOT/route_a_preflight.json"
SUB="$OCT_A_STEP2_ROOT/submission.txt"
{ echo "CODE=$CODE"; echo "E_STAR=$E"; echo "ROOT=$OCT_A_STEP2_ROOT";
  echo "PANEL_SHA256=$(sha256sum "$PANEL_SRC" | cut -d' ' -f1)"; } > "$SUB"
COMMON=(--root "$OCT_A_STEP2_ROOT" --data_dir "$OCT_DATA_DIR" --shadow_epochs "$E")
FULL="$(sbatch --parsable --account="$ACCOUNT" --job-name=oct_A_truth_full \
  scripts/cross_stage/10_calibration.slurm truth "${COMMON[@]}" --condition full)"
echo "TRUTH_FULL=$FULL" >> "$SUB"
DOSE="$(sbatch --parsable --account="$ACCOUNT" --job-name=oct_A_truth_dose \
  scripts/cross_stage/10_calibration.slurm truth "${COMMON[@]}" --condition dose01)"
echo "TRUTH_DOSE01=$DOSE" >> "$SUB"
SCORE="$(sbatch --parsable --account="$ACCOUNT" --job-name=oct_A_score --dependency="afterok:$FULL" \
  --kill-on-invalid-dep=yes scripts/cross_stage/10_calibration.slurm score "${COMMON[@]}" --condition full \
  --damping_attack 0.2 --damping_shadow 1 --damping_shadow_grid 0.8 1.2)"
echo "SCORE=$SCORE" >> "$SUB"
DIAG="$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$FULL:$DOSE:$SCORE" --kill-on-invalid-dep=yes \
  --export=ALL,OCT_A_STEP2_ROOT="$OCT_A_STEP2_ROOT" scripts/cross_stage/21_A_diag_at_E.slurm)"
echo "DIAG=$DIAG" >> "$SUB"
cat "$SUB"
cat <<MSG

Monitor:
  sacct -X -j $FULL,$DOSE,$SCORE,$DIAG --format=JobID%12,JobName%18,State,ExitCode,Elapsed,NodeList
After all four are COMPLETED 0:0:
  module load pytorch-conda/2.8
  python3 scripts/cross_stage/22_A_compare_E.py --new_root "$OCT_A_STEP2_ROOT" \\
      --baseline_root "$OCT_BASELINE_ROOT"
MSG
