#!/bin/bash
#SBATCH --job-name=oct_b0
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/oct_b0-%A_%a.out
#SBATCH --error=logs/oct_b0-%A_%a.err
set -euo pipefail
cd "${OCT_B_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/cross_stage/pathway2_job_env.sh
SEEDS=(42 43)
TASK="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
[[ "$TASK" == 0 || "$TASK" == 1 ]] || exit 2
SEED="${SEEDS[$TASK]}"
ROOT="$RUNS/b0_${SLURM_ARRAY_JOB_ID}_seed${SEED}"
FULL="$BASE/results/cross_stage_calibration_v4_1/shadow3_full"
DOSE="$BASE/results/cross_stage_calibration_v4_1/shadow3_dose01"
HPROBE="${OCT_H_DIR:-$BASE/results/stage1_diagnostics_v11/damping_22122145}/h_seed${SEED}.pt"
[[ -f "$HPROBE" ]] || { echo "Missing original h probe: $HPROBE" >&2; exit 2; }
python3 -B scripts/cross_stage/14_b0_train.py \
  --full_dir "$FULL" --dose_dir "$DOSE" --data_dir "$DATA" \
  --out_dir "$ROOT" --seed "$SEED" --patients 807 2085 --require_cuda
python3 -B scripts/cross_stage/14_b0_dose_derivative.py \
  --dose_dirs "$FULL" "$DOSE" "$ROOT/shadow3_dose003" "$ROOT/shadow3_dose001" \
    "$ROOT/shadow3_dose0003" "$ROOT/shadow3_dose0001" \
  --out_dir "$ROOT/analysis" --seeds "$SEED" --patients 807 2085 --h_probe "$HPROBE"
echo "B0_OUTPUT: $ROOT"
