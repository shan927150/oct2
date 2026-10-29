#!/bin/bash
#SBATCH --job-name=oct_logit_diag
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

# Read existing probabilities and logit tangents. Arguments: package, frozen checkout, interface OUT.
srun python3 "$1/logit_diag.py" --repo "$2" --out "$3"
