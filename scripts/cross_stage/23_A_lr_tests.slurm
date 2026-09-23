#!/bin/bash
#SBATCH --job-name=oct_A_lrtest
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_A_CODE:?submit through submit_A_lr_probe.sh}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export STAGE1_DIAG_REQUIRE_CUDA=1
srun --unbuffered python scripts/cross_stage/tests/test_A_lr_continuation.py
srun --unbuffered python scripts/cross_stage/tests/test_A_convergence.py
echo A_LR_PROBE_TESTS_PASSED
