#!/bin/bash
#SBATCH --job-name=oct_A1_truth
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
CONDITION="${1:?expected full or dose01}"
case "$CONDITION" in full|dose01) ;; *) echo "condition must be full or dose01" >&2; exit 2 ;; esac
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_DATA_DIR:?}"; : "${OCT_BASELINE_ROOT:?}"
: "${OCT_A_E_STAR:?}"; : "${OCT_A_CODE:?}"; : "${OCT_A_SELECTION:?}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
ACTION=truth-full
if [ "$CONDITION" = dose01 ]; then ACTION=truth-dose; fi
python3 scripts/cross_stage/21_A_step2_control.py job-check --root "$OCT_A_STEP2_ROOT" --action "$ACTION"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
REFERENCE="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/shadow3_${CONDITION}"
srun --unbuffered python scripts/cross_stage/21_A_truth_at_E.py \
  --require_cuda \
  --root "$OCT_A_STEP2_ROOT" --data_dir "$OCT_DATA_DIR" --reference_dir "$REFERENCE" \
  --selection "$OCT_A_SELECTION" \
  --condition "$CONDITION" --shadow_epochs "$OCT_A_E_STAR" --original_epochs 50
echo "A1_TRUTH_${CONDITION}_DONE"
