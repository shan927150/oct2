#!/bin/bash
#SBATCH --job-name=oct_c_query
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# Same GPU type as v6 (gpuA100x4): the attack replay must be exact.
# Arguments: package dir, frozen checkout, interface OUT, destination, [classes, default 0,1,2,3]
set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
pkg="$1"; frozen="$2"; out="$3"; dest="$4"; classes="${5:-0,1,2,3}"
srun python3 "$pkg/method_c_query.py" worker --repo "$frozen" --out "$out" --dest "$dest" \
  --task "$SLURM_ARRAY_TASK_ID" --classes "$classes"
