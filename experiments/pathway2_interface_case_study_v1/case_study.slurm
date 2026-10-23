#!/bin/bash
#SBATCH --job-name=oct_if_case
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Arguments: diagnostic package, original frozen checkout, completed interface OUT.
for k in 20 100; do
    srun python3 "$1/case_study.py" --repo "$2" --out "$3" --gamma 0.03 --top-k "$k"
done
