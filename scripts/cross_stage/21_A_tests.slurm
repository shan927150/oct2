#!/bin/bash
#SBATCH --job-name=oct_A2_test
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_A_CODE:?}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
python3 scripts/cross_stage/21_A_step2_control.py job-check --root "$OCT_A_STEP2_ROOT" --action test
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun --unbuffered python scripts/cross_stage/tests/test_A_control_plane.py
srun --unbuffered python scripts/cross_stage/tests/run_A_step2_chain.py \
  --output_dir "$OCT_A_STEP2_ROOT/step2_gpu_smoke"
echo A_STEP2_GPU_TESTS_PASSED
