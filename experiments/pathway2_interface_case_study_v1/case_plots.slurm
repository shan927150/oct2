#!/bin/bash
#SBATCH --job-name=oct_case_plots
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

pkg="$1"
frozen="$2"
out="$3"
case_dir="$out/case_study/gamma0.03_k100"

srun python3 "$pkg/case_figures.py" --dir "$case_dir" --examples 8
srun python3 "$pkg/case_montage.py" --repo "$frozen" --out "$out" --dir "$case_dir"
