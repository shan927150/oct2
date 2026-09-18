#!/bin/bash
#SBATCH --job-name=oct_A_diagE
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Route A step 2: unchanged v1.1 diagnostics (12_stage1_diagnostics_v11.py) on the E*-epoch truth.
# preflight = scale/linearity ratios (no GPU work); damping = Lanczos-MINRES solves on a smaller-gamma grid.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f src/config.py
: "${OCT_A_STEP2_ROOT:?}"; : "${OCT_DATA_DIR:?}"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
FULL="$(echo "$OCT_A_STEP2_ROOT"/shadow*_full)"
DOSE="$(echo "$OCT_A_STEP2_ROOT"/shadow*_dose01)"
test -d "$FULL" && test -d "$DOSE" || { echo "Expected one shadow*_full and one shadow*_dose01 under $OCT_A_STEP2_ROOT" >&2; exit 2; }
GRID="${OCT_A_DAMPING_GRID:-0.01 0.03 0.1 0.3 1 2}"
srun --unbuffered python scripts/cross_stage/12_stage1_diagnostics_v11.py --mode preflight \
  --full_dir "$FULL" --dose_dir "$DOSE" --data_dir "$OCT_DATA_DIR" --out_dir "$OCT_A_STEP2_ROOT/diag_preflight"
# shellcheck disable=SC2086
srun --unbuffered python scripts/cross_stage/12_stage1_diagnostics_v11.py --mode damping --require_cuda \
  --full_dir "$FULL" --dose_dir "$DOSE" --data_dir "$OCT_DATA_DIR" --out_dir "$OCT_A_STEP2_ROOT/diag_damping" \
  --damping_grid $GRID
echo A_STEP2_DIAG_DONE
