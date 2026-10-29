#!/bin/bash
#SBATCH --job-name=oct_grad_norm
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Gradient norms and feature plots. Arguments: package, frozen checkout, interface OUT, optional --seeds.
pkg="$1"; frozen="$2"; out="$3"; shift 3
srun python3 "$pkg/self_influence.py" --repo "$frozen" --out "$out" "$@"
srun python3 "$pkg/case_features.py" --dir "$out/case_study/gamma0.03_k100" "$@"
