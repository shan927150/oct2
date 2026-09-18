#!/bin/bash
#SBATCH --job-name=oct_A_conv
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Route A step 1 (meeting 2026-09-18). Primary array tasks 0-4 are the
# affected-shadow seeds 42..46.  No Hessian spectrum or secondary models run
# unless explicitly requested outside the primary submission.
# Submit through scripts/cross_stage/submit_A_step1.sh, which runs the read-only preflight first.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_BASELINE_ROOT:?set OCT_BASELINE_ROOT to the frozen oct2-calibration-v4 tree}"
: "${OCT_DATA_DIR:?set OCT_DATA_DIR}"
: "${OCT_A_OUT:?set OCT_A_OUT (created by submit_A_step1.sh)}"
: "${OCT_A_CODE:?submit through submit_A_step1.sh}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
FULL="$OCT_BASELINE_ROOT/results/cross_stage_calibration_v4_1/shadow3_full"
srun --unbuffered python scripts/cross_stage/20_A_convergence_curve.py \
  --require_cuda --full_dir "$FULL" --data_dir "$OCT_DATA_DIR" --out_root "$OCT_A_OUT" \
  --array_index "${SLURM_ARRAY_TASK_ID:?array task id missing}" \
  --epochs "${OCT_A_EPOCHS:-100}" --gradient_every 5 \
  --dropout_mc_every 25 --dropout_mc_reps 16 "$@"
