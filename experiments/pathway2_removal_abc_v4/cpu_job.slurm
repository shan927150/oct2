#!/bin/bash
#SBATCH --job-name=oct_removal_cpu
#SBATCH --account=bgjy-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Arguments: package dir, script name, then the script's own options.
# e.g. sbatch cpu_job.slurm "$pkg" method_a_score_drop.py --repo ... --out ... --dest ...
set -euo pipefail
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
pkg="$1"; script="$2"; shift 2
srun python3 "$pkg/$script" "$@"
