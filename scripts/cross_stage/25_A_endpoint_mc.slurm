#!/bin/bash
#SBATCH --job-name=oct_A_mc128
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_DIR:?}"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun python3 scripts/cross_stage/24_A_smallpanel_control.py mc-exec "${OCT_A_SP_OUT:?}"
