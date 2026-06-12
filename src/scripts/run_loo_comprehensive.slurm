#!/bin/bash
#SBATCH --job-name=oct_loo_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgjy-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
# 部署到 Delta 后的项目目录 (默认 ~/oct2; 可 PROJECT_DIR=... sbatch 覆盖)
PROJECT_DIR="${PROJECT_DIR:-$HOME/oct2}"
cd "$PROJECT_DIR"
module reset
module load pytorch-conda/2.8
mkdir -p logs

# 见 run_tracin_experiments.slurm 说明。若先提交了那个 job 并已重生成 scores，
# 这里可 `REGEN_SCORES=0 sbatch run_loo_comprehensive.slurm` 避免重复生成。
REGEN_SCORES="${REGEN_SCORES:-1}"
TRACIN_DIR=./results/oct_mia_tracin

echo "=== Comprehensive LOO Validation ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date) | REGEN_SCORES=$REGEN_SCORES"

# ---- Step 0: 用 attack_l2=0 重生成 scores (复用 attack_data.npz) ----
if [ "$REGEN_SCORES" = "1" ]; then
  echo ""
  echo ">>> Step 0: Regenerating checkpoints + scores (attack_l2=0)"
  python run_tracin.py \
    --preset oct \
    --output_dir "$TRACIN_DIR" \
    --checkpoint_every 5 \
    --classes 0 1 2 3
else
  echo ">>> Step 0 skipped (REGEN_SCORES=0); using existing scores in $TRACIN_DIR/tracin/"
fi

# ---- 实验: Comprehensive LOO (50 pts × 5 seeds × 4 classes) ----
echo ""
echo ">>> Comprehensive Leave-One-Out / Leave-K-Out"
python run_loo_comprehensive.py \
  --attack_data_path "$TRACIN_DIR/tracin/attack_data.npz" \
  --output_dir ./results/oct_loo_full \
  --classes 0 1 2 3 \
  --n_loo_points 50 \
  --n_seeds 5 \
  --n_test_points 50

# ---- Outlier 移除 + 相关性重画 (两种方法) ----
# 读上一步产出的 loo_comprehensive_class*.json，纯 CPU，几秒钟。
echo ""
echo ">>> Outlier removal plots (IQR + extreme TracIn cluster)"
python plot_loo_outliers.py \
  --loo_dir ./results/oct_loo_full \
  --output_dir ./results/oct_loo_outlier \
  --classes 0 1 2 3 \
  --iqr_k 1.5 \
  --n_clusters 4 \
  --drop_cluster extreme_low

echo ""
echo "=== Done. $(date) ==="
echo "Outputs:"
echo "  LOO full:     ./results/oct_loo_full/"
echo "  Outlier plots:./results/oct_loo_outlier/"
