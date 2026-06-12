#!/usr/bin/env python3
"""
实验: Pilot Leave-One-Out 验证 (10 点) [ours]。
已被 run_loo_comprehensive.py 替代，保留作快速单类验证。

选 5 top proponents + 5 top opponents，逐个删除重训，测特定 member 测试点 Δloss，
算 Spearman/Pearson(TracIn, Δloss)。预期: 删 proponent→loss↑, 删 opponent→loss↓。

用法:
  python run_loo_validation.py --scores_path .../scores_class1.npz --class_id 1
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_test_loss(model, test_x, test_y):
    """逐样本 CE loss。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        x_t = torch.tensor(test_x, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(test_y, dtype=torch.long, device=DEVICE)
        return criterion(model(x_t), y_t).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str,
                        default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--scores_path", type=str, default=None)
    parser.add_argument("--class_id", type=int, default=1)   # 默认 DME (强信号)
    parser.add_argument("--output_dir", type=str, default="./results/oct_loo")
    parser.add_argument("--n_proponents", type=int, default=5)
    parser.add_argument("--n_opponents", type=int, default=5)
    parser.add_argument("--n_test_points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    c = args.class_id
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scores_path is None:
        args.scores_path = f"./results/oct_mia_tracin/tracin/scores_class{c}.npz"

    ad = np.load(args.attack_data_path)
    tr_mask = ad["train_classes"] == c
    te_mask = ad["test_classes"] == c
    c_tr_x, c_tr_y = ad["attack_train_x"][tr_mask], ad["attack_train_y"][tr_mask]
    c_te_x, c_te_y = ad["attack_test_x"][te_mask], ad["attack_test_y"][te_mask]
    scores = np.load(args.scores_path)["scores"]
    logger.info(f"Class {c} ({CLASS_NAMES.get(c, c)}): train={len(c_tr_x)}, test={len(c_te_x)}, scores={scores.shape}")

    # 选 member 测试点，按对其平均 TracIn 排序选 proponents/opponents
    member_test_idx = np.where(c_te_y == 1)[0][:args.n_test_points]
    mean_scores = scores[member_test_idx].mean(axis=0)
    top_pro = np.argsort(mean_scores)[::-1][:args.n_proponents]
    top_opp = np.argsort(mean_scores)[:args.n_opponents]
    selected_idx = np.concatenate([top_pro, top_opp])
    selected_tracin = mean_scores[selected_idx]
    selected_labels = np.array(["proponent"] * args.n_proponents + ["opponent"] * args.n_opponents)

    from models import build_attack_model
    from attribution import train_attack_with_checkpoints
    from config import preset_oct
    cfg = preset_oct()
    n_in = c_tr_x.shape[1]

    def train_and_eval(train_x, train_y, label=""):
        """固定 seed 训练，返回选定测试点平均 loss。"""
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build_attack_model("nn", n_in, cfg.attack_n_hidden)
        model, _ = train_attack_with_checkpoints(
            model, train_x, train_y, epochs=cfg.attack_epochs, lr=cfg.attack_lr,
            batch_size=cfg.attack_batch_size, l2_ratio=cfg.attack_l2,
            checkpoint_every=cfg.attack_epochs, label=label)
        return float(compute_test_loss(model, c_te_x[member_test_idx], c_te_y[member_test_idx]).mean())

    baseline_loss = train_and_eval(c_tr_x, c_tr_y, "Baseline")
    logger.info(f"Baseline loss: {baseline_loss:.6f}")

    delta_losses = []
    for i, idx in enumerate(selected_idx):
        loo_mask = np.ones(len(c_tr_x), dtype=bool)
        loo_mask[idx] = False
        delta = train_and_eval(c_tr_x[loo_mask], c_tr_y[loo_mask], f"LOO-{i}") - baseline_loss
        delta_losses.append(delta)
        logger.info(f"  remove idx={idx} ({selected_labels[i]}): Δloss={delta:+.6f}, TracIn={selected_tracin[i]:+.4f}")
    delta_losses = np.array(delta_losses)

    spearman_r, spearman_p = spearmanr(selected_tracin, delta_losses)
    pearson_r, pearson_p = pearsonr(selected_tracin, delta_losses)
    logger.info(f"Spearman r={spearman_r:.4f} (p={spearman_p:.4f}), Pearson r={pearson_r:.4f} (p={pearson_p:.4f})")

    # 画图: TracIn vs Δloss + per-sample bar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Leave-One-Out — Class {c} ({CLASS_NAMES.get(c, c)})\n"
                 f"Spearman r={spearman_r:.3f} (p={spearman_p:.3f}), Pearson r={pearson_r:.3f} (p={pearson_p:.3f})",
                 fontsize=13, fontweight="bold")
    pro_mask = selected_labels == "proponent"
    ax = axes[0]
    ax.scatter(selected_tracin[pro_mask], delta_losses[pro_mask], c="#27ae60", s=80, label="Proponent",
               zorder=3, edgecolors="white")
    ax.scatter(selected_tracin[~pro_mask], delta_losses[~pro_mask], c="#8e44ad", s=80, label="Opponent",
               zorder=3, edgecolors="white")
    ax.axhline(0, color="gray", ls="--", alpha=0.4); ax.axvline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("TracIn Score"); ax.set_ylabel("Actual Δloss")
    ax.set_title("Predicted vs Actual Influence"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ["#27ae60" if l == "proponent" else "#8e44ad" for l in selected_labels]
    ax.bar(np.arange(len(selected_idx)), delta_losses, color=colors, edgecolor="white", alpha=0.8)
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Training Sample"); ax.set_ylabel("Δloss"); ax.set_title("Per-Sample LOO Effect")
    ax.set_xticks(np.arange(len(selected_idx)))
    ax.set_xticklabels([f"{'P' if l == 'proponent' else 'O'}{i}" for i, l in enumerate(selected_labels)], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(str(out_dir / f"loo_class{c}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"loo_class{c}.json", "w") as f:
        json.dump({"class": int(c), "class_name": CLASS_NAMES.get(c, str(c)),
                   "n_train": int(len(c_tr_x)), "n_test_points": int(len(member_test_idx)),
                   "baseline_loss": float(baseline_loss), "spearman_r": float(spearman_r),
                   "spearman_p": float(spearman_p), "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
                   "samples": [{"train_idx": int(idx), "type": selected_labels[i],
                                "membership": int(c_tr_y[idx]), "tracin_score": float(selected_tracin[i]),
                                "delta_loss": float(delta_losses[i])}
                               for i, idx in enumerate(selected_idx)]}, f, indent=2)


if __name__ == "__main__":
    main()
