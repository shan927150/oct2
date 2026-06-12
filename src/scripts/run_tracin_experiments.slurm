#!/bin/bash
#SBATCH --job-name=oct_tracin_exp
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

# 是否用 l2=0 重生成 checkpoints+scores (复用 attack_data.npz)。
# 1 = 重生成 (默认，独立提交时必须); 0 = 沿用已有 scores (若同一轮已先跑过 regen)。
REGEN_SCORES="${REGEN_SCORES:-1}"
TRACIN_DIR=./results/oct_mia_tracin

echo "=== TracIn Validation Experiments ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date) | REGEN_SCORES=$REGEN_SCORES"
echo "======================================"

# ---- Step 0: 用 attack_l2=0 重生成 checkpoints + scores (复用 attack_data.npz) ----
# l2 改 0 后 attack model 梯度变了，旧 scores_class*.npz 失效，必须先重生成。
# run_tracin.py 检测到 attack_data.npz 存在会直接复用 (target/shadow 不重训)。
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

# ---- 实验 1: Mislabel Detection (不读 scores，自身用 l2=0 重训) ----
echo ""
echo ">>> Experiment 1: Mislabel Detection (Recovery@10%)"
python run_mislabel_detection.py \
  --attack_data_path "$TRACIN_DIR/tracin/attack_data.npz" \
  --output_dir ./results/oct_mislabel \
  --flip_ratio 0.10 \
  --classes 0 1 2 3

# ---- 实验 2: Leave-One-Out (DME, 用重生成的 scores_class1) ----
echo ""
echo ">>> Experiment 2: Leave-One-Out Validation (Class 1 = DME)"
python run_loo_validation.py \
  --attack_data_path "$TRACIN_DIR/tracin/attack_data.npz" \
  --scores_path "$TRACIN_DIR/tracin/scores_class1.npz" \
  --class_id 1 \
  --output_dir ./results/oct_loo \
  --n_proponents 5 \
  --n_opponents 5

echo ""
echo "=== All experiments done. $(date) ==="
echo "Outputs:"
echo "  Mislabel: ./results/oct_mislabel/"
echo "  LOO:      ./results/oct_loo/"
