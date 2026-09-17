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
set -euo pipefail
cd "${OCT_B_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/cross_stage/pathway2_job_env.sh
OUT="$RUNS/e0_${SLURM_JOB_ID}"
python3 -B scripts/cross_stage/13_e0_residual_decomposition.py \
  --full_dir "$BASE/results/cross_stage_calibration_v4_1/shadow3_full" \
  --dose_dir "$BASE/results/cross_stage_calibration_v4_1/shadow3_dose01" \
  --data_dir "$DATA" --out_dir "$OUT" \
  --seeds 42 43 --core_patients 807 2085 \
  --gammas 0 0.03 0.3 1.0 --hvp_batch 64 --dropout_mc_reps 16 --require_cuda
echo "E0_OUTPUT: $OUT"
