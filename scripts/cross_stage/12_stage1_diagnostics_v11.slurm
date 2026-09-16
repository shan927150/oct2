#!/bin/bash
#SBATCH --job-name=oct_s1v11
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:-}"
case "$mode" in
  tests)
    export STAGE1_DIAG_REQUIRE_CUDA=1
    srun --unbuffered python scripts/cross_stage/tests/test_stage1_diagnostics.py
    srun --unbuffered python scripts/cross_stage/tests/test_stage1_v11.py
    ;;
  damping|convergence)
    srun --unbuffered python scripts/cross_stage/12_stage1_diagnostics_v11.py \
      --mode "$mode" --require_cuda \
      --out_dir "results/stage1_diagnostics_v11/${mode}_${SLURM_JOB_ID}" "${@:2}"
    ;;
  *) echo "Expected tests, damping or convergence" >&2; exit 2 ;;
esac
