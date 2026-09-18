#!/bin/bash
#SBATCH --job-name=oct_b1
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/oct_b1-%A_%a.out
#SBATCH --error=logs/oct_b1-%A_%a.err
# One task per seed, as asked: the two seeds are separate jobs, and each task carries
# the whole patient list so a seed's cells are never split across jobs.
set -euo pipefail
cd "${OCT_B_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/cross_stage/pathway2_job_env.sh
SEEDS=(42 43)
TASK="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
[[ "$TASK" == 0 || "$TASK" == 1 ]] || exit 2
SEED="${SEEDS[$TASK]}"
OUT="$RUNS/b1_${SLURM_ARRAY_JOB_ID}_seed${SEED}"
FULL="$BASE/results/cross_stage_calibration_v4_1/shadow3_full"
DOSE="$BASE/results/cross_stage_calibration_v4_1/shadow3_dose01"
# Prefixes: one optimizer step, one epoch, five epochs. Nothing longer until the short
# chains pass -- a 50-epoch endpoint is not a reference for a short-chain derivative.
python3 -B scripts/cross_stage/16_b1_smoke.py \
  --full_dir "$FULL" --dose_dir "$DOSE" --data_dir "$DATA" \
  --out_dir "$OUT" --seed "$SEED" \
  --patients 807 2085 1369 \
  --prefix_epochs 0 1 5 \
  --exponents 10 14 18 22 24 \
  --require_cuda
echo "B1_OUTPUT: $OUT"
