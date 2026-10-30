#!/bin/bash
#SBATCH --job-name=oct_attack_prune
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-14%4

set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
pkg="$1"; frozen="$2"; out="$3"; dest="$4"; n="${5:-50}"
srun python3 "$pkg/class_image_ablation.py" --repo "$frozen" --out "$out" --dest "$dest" --task "$SLURM_ARRAY_TASK_ID" --n-selected "$n"
