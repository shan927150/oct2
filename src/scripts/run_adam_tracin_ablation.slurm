#!/bin/bash
#SBATCH --job-name=oct_adam_tracin
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --account=bgjy-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/oct_adam_tracin-%j.out
#SBATCH --error=logs/oct_adam_tracin-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/oct2}"
cd "$PROJECT_DIR"
mkdir -p logs results

echo "=== Adam-aware TracIn ablation: + same-ckpt raw control, + cluster diagnostics (replaces outlier) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"

module reset
module load pytorch-conda/2.8

ATTACK_DATA="./results/oct_mia_tracin/tracin/attack_data.npz"
[ -f "$ATTACK_DATA" ] || { echo "ERROR: $ATTACK_DATA not found"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Adam-aware scores. Writes 'scores'=adam_dot AND 'scores_raw' (same
#         checkpoints) into each scores_class*.npz. Skipped if already present.
# ---------------------------------------------------------------------------
if [ ! -f ./results/oct_mia_tracin_adam/tracin/scores_class0.npz ]; then
  python run_adam_tracin.py \
    --attack_data_path "$ATTACK_DATA" \
    --output_dir ./results/oct_mia_tracin_adam \
    --variant adam_dot \
    --checkpoint_every 5 \
    --classes 0 1 2 3
else
  echo "[skip] Adam scores already exist."
fi

# ---------------------------------------------------------------------------
# Step 1b: repackage same-checkpoint raw scores into a parallel dir (CPU only,
#          no gradient recompute) so raw vs adam is apples-to-apples.
# ---------------------------------------------------------------------------
python - <<'PY'
import numpy as np
from pathlib import Path
src = Path("results/oct_mia_tracin_adam/tracin")
dst = Path("results/oct_mia_tracin_rawckpt/tracin")
dst.mkdir(parents=True, exist_ok=True)
for c in range(4):
    d = dict(np.load(src / f"scores_class{c}.npz"))
    d["scores"] = d["scores_raw"]                 # same-ckpt raw TracIn
    d["self_influence"] = d["self_influence_raw"]
    np.savez_compressed(dst / f"scores_class{c}.npz", **d)
    print("wrote", dst / f"scores_class{c}.npz")
PY

# ---------------------------------------------------------------------------
# Step 2: LOO / Leave-K-Out for both score variants (GPU; retrains attack model).
#         Adam side skipped if its summary already exists.
# ---------------------------------------------------------------------------
run_loo () {  # $1 scores_dir  $2 score_label  $3 out_dir
  if [ -f "$3/loo_summary.json" ]; then echo "[skip] LOO exists: $3"; return; fi
  python run_loo_comprehensive_from_scores.py \
    --attack_data_path "$ATTACK_DATA" \
    --scores_dir "$1" --score_label "$2" --output_dir "$3" \
    --classes 0 1 2 3 --n_loo_points 50 --n_seeds 5 --n_test_points 50
}
run_loo ./results/oct_mia_tracin_adam/tracin    "Adam-aware Score"        ./results/oct_loo_full_adam
run_loo ./results/oct_mia_tracin_rawckpt/tracin "raw TracIn (same ckpt)"  ./results/oct_loo_full_raw_same_ckpt

# ---------------------------------------------------------------------------
# Step 3: cluster diagnostics (REPLACES plot_loo_outliers.py). Pure CPU.
#         between-cluster trend + within-cluster correlation, on both variants.
# ---------------------------------------------------------------------------
for suf in adam raw_same_ckpt; do
  loo=./results/oct_loo_full_${suf}
  python loo_cluster_trend_v2.py \
    --loo_dir "$loo" \
    --output_dir ./results/oct_cluster_trend_${suf} \
    --classes 0 1 2 3 --n_clusters 6
  python loo_within_cluster_corr.py \
    --loo_dir "$loo" \
    --output_dir ./results/oct_within_cluster_${suf} \
    --classes 0 1 2 3 --n_clusters 6
done

echo "=== Done ==="
echo "LOO:    ./results/oct_loo_full_adam , ./results/oct_loo_full_raw_same_ckpt"
echo "trend:  ./results/oct_cluster_trend_adam , ./results/oct_cluster_trend_raw_same_ckpt"
echo "within: ./results/oct_within_cluster_adam , ./results/oct_within_cluster_raw_same_ckpt"
echo "Finished at $(date)"
