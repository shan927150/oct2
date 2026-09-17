#!/bin/bash
#SBATCH --job-name=oct_b0
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/oct_b0-%A_%a.out
#SBATCH --error=logs/oct_b0-%A_%a.err
# B0: small-dose retraining of the ORIGINAL 50-epoch algorithm.
# Array task 0 -> seed 42, task 1 -> seed 43. Each task covers 4 doses x 2 patients
# = 8 perturbed Stage-1 runs, all under the B output root. The original results
# directory is never written to.
#
# Each --condition gets its own 05 output directory and therefore trains its own
# baseline, so a task performs 4 baseline + 8 perturbed trainings. The redundancy is
# deliberate: 14_b0_dose_derivative.py refuses to compare doses whose baselines are
# not bitwise identical, so the repeated baselines are a determinism check, not waste.
#
# Walltime: one alpha=0.1 truth over 8 patients x 5 seeds took 51 min, i.e. ~1.3 min
# per Stage-1 training. 12 trainings plus attack/no-op overhead should be well under
# an hour; 8h is a deliberately loose first-run bound. Shrink it after one run.
set -euo pipefail

module reset
module load pytorch-conda/2.8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src"

SEEDS=(42 43)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
BASE="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
RUNS="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
ROOT="$RUNS/b0_seed${SEED}"
DATA="${OCT_DATA_DIR:-/u/yli103/oct2/data}"
PANEL_SRC="$BASE/results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json"
mkdir -p logs "$ROOT/panel"

# Reuse the frozen panel verbatim; 08 is never re-run, so the patient selection
# cannot drift toward whichever patients look convenient.
cp -n "$PANEL_SRC" "$ROOT/panel/eligibility_preflight.json"

python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$BASE" --output-root "$RUNS"

COMMON=(--root "$ROOT" --data_dir "$DATA" --stage1_seeds "$SEED"
        --patients 807 2085 --require_cuda)

# Baseline first: it must reproduce the original bitwise before any dose is trusted.
python3 -B scripts/cross_stage/10_run_calibration.py --phase baseline \
  --condition dose003 "${COMMON[@]}"

for CONDITION in dose003 dose001 dose0003 dose0001; do
  echo "=== $CONDITION seed=$SEED ==="
  python3 -B scripts/cross_stage/10_run_calibration.py --phase truth \
    --condition "$CONDITION" "${COMMON[@]}"
done

python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$BASE" --output-root "$RUNS"
echo "B0_OUTPUT: $ROOT"
echo "Analyse both seeds with 14_b0_dose_derivative.py once the array finishes."
