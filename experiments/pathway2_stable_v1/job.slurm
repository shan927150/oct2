#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
pkg="$1"
out="$2"
mode="$3"
if [[ "$mode" == smoke ]]; then
  srun python3 "$pkg/test_experiment.py" --require-torch
  srun python3 "$pkg/run.py" gpu --out "$out" --index 0 --smoke
elif [[ "$mode" == run ]]; then
  srun python3 "$pkg/run.py" gpu --out "$out" --index "${SLURM_ARRAY_TASK_ID:?}"
else
  exit 2
fi
