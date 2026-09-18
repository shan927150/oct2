#!/bin/bash
#SBATCH --job-name=oct_A1_score
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
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_DATA_DIR:?}"; : "${OCT_A_E_STAR:?}"; : "${OCT_A_CODE:?}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
python3 scripts/cross_stage/21_A_step2_control.py job-check --root "$OCT_A_STEP2_ROOT" --action score
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun --unbuffered python scripts/cross_stage/10_run_calibration.py --require_cuda --phase score \
  --root "$OCT_A_STEP2_ROOT" --data_dir "$OCT_DATA_DIR" --shadow_epochs "$OCT_A_E_STAR" \
  --condition full --damping_attack 0.2 --damping_shadow 1 --damping_shadow_grid
echo A1_SCORE_GAMMA1_DONE
