#!/bin/bash
#SBATCH --job-name=oct_class_image_ablation
#SBATCH --account=bgjy-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Arguments: package, frozen checkout, interface OUT, then Python options.
pkg="$1"
frozen="$2"
out="$3"
shift 3
srun python3 "$pkg/class_image_ablation.py" --repo "$frozen" --out "$out" "$@"
