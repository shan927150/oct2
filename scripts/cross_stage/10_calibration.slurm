#!/bin/bash
#SBATCH --job-name=oct_cal_v4
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from the repository root; supply your account using sbatch --account.
# Example: sbatch --account=bgjy-delta-gpu scripts/cross_stage/10_calibration.slurm preflight
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun --unbuffered python scripts/cross_stage/10_run_calibration.py --require_cuda --phase "${1:-preflight}" "${@:2}"
