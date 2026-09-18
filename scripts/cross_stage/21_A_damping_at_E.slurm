#!/bin/bash
#SBATCH --job-name=oct_A2_damp
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
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_DATA_DIR:?}"; : "${OCT_A_CODE:?}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
python3 scripts/cross_stage/21_A_step2_control.py job-check --root "$OCT_A_STEP2_ROOT" --action damping
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
FULL="$OCT_A_STEP2_ROOT/shadow3_full"
DOSE="$OCT_A_STEP2_ROOT/shadow3_dose01"
srun --unbuffered python scripts/cross_stage/12_stage1_diagnostics_v11.py --mode damping --require_cuda \
  --full_dir "$FULL" --dose_dir "$DOSE" --data_dir "$OCT_DATA_DIR" \
  --out_dir "$OCT_A_STEP2_ROOT/diag_damping" --damping_grid 0.01 0.03 0.1 0.3 1 2
echo A2_DAMPING_DONE
