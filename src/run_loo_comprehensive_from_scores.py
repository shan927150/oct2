#!/usr/bin/env python3
"""
实验: Comprehensive Leave-One-Out & Leave-K-Out 验证 [ours, 可验证 raw TracIn 或 Adam-aware scores]。

  Part A — Single-point LOO: 按 TracIn score 分层选 50 点，逐个删除重训 (×5 seeds)，
           测 Δloss=loo_loss-baseline，算 Spearman/Pearson(TracIn, Δloss)。
  Part B — Leave-K-Out: 批量删 top/bottom k proponents/opponents，放大单点微弱信号。

预期 (若 TracIn 准确): 删 proponent → loss↑ (Δ>0); 删 opponent → loss↓ (Δ<0)。
loss=CrossEntropyLoss(reduction='none') 与 attack 训练一致。

用法:
  python run_loo_comprehensive.py --attack_data_path .../attack_data.npz \
      --output_dir ./results/oct_loo_full --classes 0 1 2 3
"""
import argparse
import json
import logging
import time
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
    """attack model 在测试点上的逐样本 CE loss。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        x_t = torch.tensor(test_x, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(test_y, dtype=torch.long, device=DEVICE)
        return criterion(model(x_t), y_t).cpu().numpy()


def train_and_eval(train_x, train_y, test_x, test_y, n_in, cfg, seed, label=""):
    """固定 seed 训练 attack model，返回测试集平均 loss。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    from models import build_attack_model
    from attribution import train_attack_with_checkpoints
    model = build_attack_model("nn", n_in, cfg.attack_n_hidden)
    model, _ = train_attack_with_checkpoints(
        model, train_x, train_y, epochs=cfg.attack_epochs, lr=cfg.attack_lr,
        batch_size=cfg.attack_batch_size, l2_ratio=cfg.attack_l2,
        checkpoint_every=cfg.attack_epochs, label=label)
    return float(compute_test_loss(model, test_x, test_y).mean())


def select_stratified_points(mean_scores, n_points=50):
    """按 TracIn score 分层选点: top extremes + 中段均匀 + bottom extremes，覆盖整个范围。"""
    ranked = np.argsort(mean_scores)[::-1]
    n_extreme = min(10, n_points // 5)
    n_mid = n_points - 2 * n_extreme
    top_idx, bottom_idx = ranked[:n_extreme], ranked[-n_extreme:]
    mid_range = ranked[n_extreme:-n_extreme]
    if len(mid_range) > n_mid:
        step = len(mid_range) / n_mid
        mid_idx = mid_range[[int(i * step) for i in range(n_mid)]]
    else:
        mid_idx = mid_range
    return np.unique(np.concatenate([top_idx, mid_idx, bottom_idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str,
                        default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--scores_dir", type=str, default="./results/oct_mia_tracin/tracin",
                        help="Directory containing scores_class{c}.npz. Default is raw TracIn; set to Adam-aware scores dir for ablation.")
    parser.add_argument("--score_label", type=str, default="TracIn Score",
                        help="Axis label for the score being validated, e.g. Adam-aware Score")
    parser.add_argument("--output_dir", type=str, default="./results/oct_loo_full")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--n_loo_points", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_test_points", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.attack_data_path)
    attack_train_x, attack_train_y, train_classes = ad["attack_train_x"], ad["attack_train_y"], ad["train_classes"]
    attack_test_x, attack_test_y, test_classes = ad["attack_test_x"], ad["attack_test_y"], ad["test_classes"]

    from config import preset_oct
    cfg = preset_oct()
    seeds = [args.base_seed + s for s in range(args.n_seeds)]
    all_class_results = {}

    for c in args.classes:
        t0 = time.time()
        logger.info(f"\n{'='*70}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*70}")
        tr_mask, te_mask = train_classes == c, test_classes == c
        c_tr_x, c_tr_y = attack_train_x[tr_mask], attack_train_y[tr_mask]
        c_te_x, c_te_y = attack_test_x[te_mask], attack_test_y[te_mask]
        n_in, n_train = c_tr_x.shape[1], len(c_tr_x)

        member_test_idx = np.where(c_te_y == 1)[0][:args.n_test_points]
        test_x_eval, test_y_eval = c_te_x[member_test_idx], c_te_y[member_test_idx]

        scores = np.load(str(Path(args.scores_dir) / f"scores_class{c}.npz"))["scores"]
        mean_scores = scores[member_test_idx].mean(axis=0)

        # ---- Part A: Single-point LOO ----
        logger.info(f"  --- Part A: Single-point LOO ({args.n_loo_points} pts × {args.n_seeds} seeds) ---")
        selected = select_stratified_points(mean_scores, args.n_loo_points)
        baseline_losses = [train_and_eval(c_tr_x, c_tr_y, test_x_eval, test_y_eval, n_in, cfg, s,
                                          f"BL-c{c}-s{s}") for s in seeds]
        baseline_mean, baseline_std = float(np.mean(baseline_losses)), float(np.std(baseline_losses))
        logger.info(f"  Baseline loss: {baseline_mean:.6f} ± {baseline_std:.6f}")

        loo_results = []
        for i, idx in enumerate(selected):
            loo_mask = np.ones(n_train, dtype=bool)
            loo_mask[idx] = False
            loo_losses = [train_and_eval(c_tr_x[loo_mask], c_tr_y[loo_mask], test_x_eval, test_y_eval,
                                        n_in, cfg, s, f"LOO-c{c}-{i}") for s in seeds]
            loo_results.append({
                "train_idx": int(idx), "tracin_score": float(mean_scores[idx]),
                "membership": int(c_tr_y[idx]),
                "delta_loss_mean": float(np.mean(loo_losses) - baseline_mean),
                "delta_loss_std": float(np.std(loo_losses)),
            })
            if (i + 1) % 10 == 0:
                logger.info(f"    LOO {i+1}/{len(selected)}: Δloss={loo_results[-1]['delta_loss_mean']:+.6f}")

        tracin_arr = np.array([r["tracin_score"] for r in loo_results])
        delta_arr = np.array([r["delta_loss_mean"] for r in loo_results])
        sp_r, sp_p = spearmanr(tracin_arr, delta_arr)
        pe_r, pe_p = pearsonr(tracin_arr, delta_arr)
        logger.info(f"  Spearman r={sp_r:.4f} (p={sp_p:.4f}), Pearson r={pe_r:.4f} (p={pe_p:.4f})")

        # ---- Part B: Leave-K-Out ----
        logger.info("  --- Part B: Leave-K-Out ---")
        ranked_desc, ranked_asc = np.argsort(mean_scores)[::-1], np.argsort(mean_scores)
        lko_results = []
        for k in [10, 20, 50]:
            for direction, name, indices in [("proponent", f"top-{k} proponents", ranked_desc[:k]),
                                             ("opponent", f"bottom-{k} opponents", ranked_asc[:k])]:
                if k > n_train // 2:
                    continue
                lko_mask = np.ones(n_train, dtype=bool)
                lko_mask[indices] = False
                lko_losses = [train_and_eval(c_tr_x[lko_mask], c_tr_y[lko_mask], test_x_eval, test_y_eval,
                                            n_in, cfg, s, f"LKO-c{c}-{name}") for s in seeds]
                delta = float(np.mean(lko_losses) - baseline_mean)
                lko_results.append({"condition": name, "direction": direction, "k": int(k),
                                    "delta_loss_mean": delta, "delta_loss_std": float(np.std(lko_losses)),
                                    "removed_member_ratio": float(c_tr_y[indices].mean())})
                logger.info(f"    {name}: Δloss={delta:+.6f} {'↑' if delta > 0 else '↓'}")

        elapsed = time.time() - t0
        logger.info(f"  Class {c} done in {elapsed:.1f}s")

        # ---- 可视化 ----
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"LOO Validation — Class {c} ({CLASS_NAMES.get(c, c)}) | n_train={n_train}\n"
                     f"Single-point: Spearman r={sp_r:.3f} (p={sp_p:.3f}), Pearson r={pe_r:.3f} (p={pe_p:.3f})",
                     fontsize=13, fontweight="bold")
        from matplotlib.lines import Line2D
        ax = axes[0]
        colors = ["#e74c3c" if r["membership"] == 1 else "#3498db" for r in loo_results]
        ax.scatter(tracin_arr, delta_arr, c=colors, alpha=0.5, s=30, edgecolors="white", lw=0.5)
        ax.axhline(0, color="gray", ls="--", alpha=0.4); ax.axvline(0, color="gray", ls="--", alpha=0.4)
        if len(tracin_arr) > 2:
            z = np.polyfit(tracin_arr, delta_arr, 1)
            xl = np.linspace(tracin_arr.min(), tracin_arr.max(), 100)
            ax.plot(xl, np.polyval(z, xl), "k--", alpha=0.5, lw=1)
        ax.set_xlabel(args.score_label); ax.set_ylabel("Δloss (mean over seeds)")
        ax.set_title(f"Single-point LOO ({len(selected)} pts)"); ax.grid(True, alpha=0.3)
        ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Member'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Non-member')],
                  fontsize=8)

        ax = axes[1]
        pro = [r for r in lko_results if r["direction"] == "proponent"]
        opp = [r for r in lko_results if r["direction"] == "opponent"]
        x_pos = np.arange(len(pro))
        ax.bar(x_pos - 0.175, [r["delta_loss_mean"] for r in pro], 0.35, color="#27ae60",
               label="Remove proponents", edgecolor="white")
        ax.bar(x_pos + 0.175, [r["delta_loss_mean"] for r in opp], 0.35, color="#8e44ad",
               label="Remove opponents", edgecolor="white")
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.set_xticks(x_pos); ax.set_xticklabels([f"k={r['k']}" for r in pro])
        ax.set_xlabel("Samples removed"); ax.set_ylabel("Δloss"); ax.set_title("Leave-K-Out")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.hist(mean_scores, bins=50, alpha=0.4, color="gray", density=True, label="All training")
        ax.hist(mean_scores[selected], bins=30, alpha=0.6, color="#e74c3c", density=True, label="Selected")
        ax.set_xlabel(f"Mean {args.score_label}"); ax.set_ylabel("Density")
        ax.set_title("Selected LOO Points"); ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(str(out_dir / f"loo_comprehensive_class{c}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(out_dir / f"loo_comprehensive_class{c}.json", "w") as f:
            json.dump({"class": int(c), "class_name": CLASS_NAMES.get(c, str(c)), "n_train": int(n_train),
                       "n_test_eval": int(len(member_test_idx)), "n_loo_points": int(len(selected)),
                       "n_seeds": int(args.n_seeds), "baseline_loss_mean": baseline_mean,
                       "baseline_loss_std": baseline_std, "spearman_r": float(sp_r), "spearman_p": float(sp_p),
                       "pearson_r": float(pe_r), "pearson_p": float(pe_p), "elapsed_sec": float(elapsed),
                       "single_point_loo": loo_results, "leave_k_out": lko_results}, f, indent=2)
        all_class_results[c] = {"spearman_r": float(sp_r), "spearman_p": float(sp_p),
                                "pearson_r": float(pe_r), "pearson_p": float(pe_p)}

    print(f"\n{'='*70}\n  COMPREHENSIVE LOO — ALL CLASSES\n{'='*70}")
    print(f"  {'Class':<10} {'Spearman r':>12} {'p':>8} {'Pearson r':>12} {'p':>8}")
    for c in args.classes:
        r = all_class_results.get(c)
        if r is None:
            continue
        print(f"  {CLASS_NAMES.get(c, str(c)):<10} {r['spearman_r']:>+12.4f} {r['spearman_p']:>8.4f} "
              f"{r['pearson_r']:>+12.4f} {r['pearson_p']:>8.4f}")
    with open(out_dir / "loo_summary.json", "w") as f:
        json.dump(all_class_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
