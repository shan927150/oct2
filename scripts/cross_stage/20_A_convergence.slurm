#!/bin/bash
#SBATCH --job-name=oct_A_conv
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Route A step 1 (meeting 2026-09-18). One array task per Stage 1 model:
#   0-4 = affected shadow seeds 42..46 (with Hessian Ritz endpoints), 5 = target, 6-9 = fixed shadows.
# Submit through scripts/cross_stage/submit_A_step1.sh, which runs the read-only preflight first.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_BASELINE_ROOT:?set OCT_BASELINE_ROOT to the frozen oct2-calibration-v4 tree}"
: "${OCT_DATA_DIR:?set OCT_DATA_DIR}"
: "${OCT_A_OUT:?set OCT_A_OUT (created by submit_A_step1.sh)}"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
FULL="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/shadow3_full"
V11="${OCT_V11_DAMPING_DIR:-$OCT_BASELINE_ROOT/results/stage1_diagnostics_v11/damping_22122145}"
srun --unbuffered python scripts/cross_stage/20_A_convergence_curve.py \
  --require_cuda --full_dir "$FULL" --data_dir "$OCT_DATA_DIR" --out_root "$OCT_A_OUT" \
  --array_index "${SLURM_ARRAY_TASK_ID:?array task id missing}" \
  --epochs "${OCT_A_EPOCHS:-100}" --v11_damping_dir "$V11" "$@"
