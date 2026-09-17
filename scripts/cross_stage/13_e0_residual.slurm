#!/bin/bash
#SBATCH --job-name=oct_e0
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=logs/oct_e0-%j.out
#SBATCH --error=logs/oct_e0-%j.err
# E0: residual decomposition on existing checkpoints. No training, reads the original
# truth directories read-only and writes only under the B output root.
# Walltime is a first-run guess: record the real elapsed_seconds from manifest.json
# and shrink it before queueing the full panel.
set -euo pipefail

module reset
module load pytorch-conda/2.8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src"

BASE="${OCT_BASELINE_ROOT:-$HOME/oct2-calibration-v4}"
RUNS="${OCT_RUNS_ROOT:-$HOME/oct2-pathway2-runs/B}"
OUT="$RUNS/e0_${SLURM_JOB_ID:-manual}"
mkdir -p logs "$RUNS"

python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$BASE" --output-root "$RUNS"

python3 -B scripts/cross_stage/13_e0_residual_decomposition.py \
  --full_dir "$BASE/results/cross_stage_calibration_v4_1/shadow3_full" \
  --dose_dir "$BASE/results/cross_stage_calibration_v4_1/shadow3_dose01" \
  --data_dir "${OCT_DATA_DIR:-/u/yli103/oct2/data}" \
  --out_dir "$OUT" \
  --seeds 42 43 --core_patients 807 2085 \
  --gammas 0 0.03 0.3 1.0 --hvp_batch 64 --dropout_mc_reps 16 \
  --require_cuda

python3 -B experiments/pathway2/preflight.py verify \
  --baseline-root "$BASE" --output-root "$RUNS"
echo "E0_OUTPUT: $OUT"
