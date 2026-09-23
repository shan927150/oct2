#!/bin/bash
#SBATCH --job-name=oct_A_lrprobe
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Route A convergence-control probe: array 0-9 = seed(42..46) x arm(control,decay),
# continuing each affected baseline from its Step-1 E100 checkpoint. Baselines only.
# Submit through submit_A_lr_probe.sh run after the separately reviewed GPU test.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_BASELINE_ROOT:?}"
: "${OCT_DATA_DIR:?}"
: "${OCT_A_STEP1_ROOT:?path of the completed convergence_E100 output root}"
: "${OCT_A_LR_OUT:?created by submit_A_lr_probe.sh}"
: "${OCT_A_CODE:?submit through submit_A_lr_probe.sh}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/cross_stage/23_A_lr_control.py check "$OCT_A_LR_OUT"
FULL="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/shadow3_full"
srun --unbuffered python scripts/cross_stage/23_A_lr_continuation.py \
  --require_cuda --full_dir "$FULL" --data_dir "$OCT_DATA_DIR" \
  --step1_root "$OCT_A_STEP1_ROOT" --out_root "$OCT_A_LR_OUT" \
  --array_index "${SLURM_ARRAY_TASK_ID:?}" \
  --resume_epoch "${OCT_A_RESUME:-100}" --extra_epochs "${OCT_A_EXTRA:-40}" \
  --decay_schedule "${OCT_A_DECAY:-20@1e-4,20@1e-5}" "$@"
