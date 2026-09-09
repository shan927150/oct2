#!/bin/bash
#SBATCH --job-name=oct_cs_ladder07
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgjy-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Offline score ladder + per-patient seed stats on a finished 05 directory.
#   sbatch --export=ALL,PILOT_DIR=results/cross_stage_patient_loo_05_pilot_v3 \
#          scripts/cross_stage/07_score_ladder.slurm
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/oct2}"
PILOT_DIR="${PILOT_DIR:-results/cross_stage_patient_loo_05_pilot_v3}"
DATA_DIR="${DATA_DIR:-./data}"
ATTACK_SEED_REPS="${ATTACK_SEED_REPS:-10}"
cd "$PROJECT_DIR"
mkdir -p logs
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1
if [ -f "$PROJECT_DIR/src/config.py" ]; then
  export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "=== 06 per-patient seed stats ==="
python scripts/cross_stage/06_pilot_seed_stats.py --pilot_dir "$PILOT_DIR"

echo "=== 07 score ladder ==="
python scripts/cross_stage/07_cross_stage_score_ladder.py \
  --pilot_dir "$PILOT_DIR" --data_dir "$DATA_DIR" \
  --attack_seed_reps "$ATTACK_SEED_REPS" --cg_iters 100 --cg_tol 1e-4 \
  --damping_attack 1e-3 --damping_shadow 1e-2 --damping_shadow_grid 3e-3 3e-2 1e-1 \
  --lanczos_iters 30 --cg_fail_tol 1e-3 --hvp_batch 128
echo "=== Finished: $(date) ==="
