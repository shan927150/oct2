#!/bin/bash
#SBATCH --job-name=oct_A1_pre
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# This is the checkpoint/dose consistency stage.  It performs no HVP and no
# training, but uses the same pinned PyTorch environment as the GPU stages.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_DATA_DIR:?}"; : "${OCT_A_CODE:?}"
test "$(git rev-parse HEAD)" = "$OCT_A_CODE"
test -z "$(git status --porcelain --untracked-files=no)"
python3 scripts/cross_stage/21_A_step2_control.py job-check --root "$OCT_A_STEP2_ROOT" --action preflight
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun --unbuffered python scripts/cross_stage/12_stage1_diagnostics_v11.py --mode preflight \
  --full_dir "$OCT_A_STEP2_ROOT/shadow3_full" \
  --dose_dir "$OCT_A_STEP2_ROOT/shadow3_dose01" \
  --data_dir "$OCT_DATA_DIR" --out_dir "$OCT_A_STEP2_ROOT/diag_preflight"
echo A1_TRUTH_PREFLIGHT_DONE
