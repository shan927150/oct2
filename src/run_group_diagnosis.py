#!/usr/bin/env python3
"""
实验: LOO 结果的 per-group 相关性诊断 [ours, 导师要求]。

导师在 DME 散点图观察到 4 个簇，想看组内相关性以解释整体弱相关。
用 KMeans 按 TracIn score 把样本聚成 n_groups 组，分别算 Spearman/Pearson(TracIn, Δloss)。

用法:
  python run_group_diagnosis.py --loo_dir ./results/oct_loo_full \
      --output_dir ./results/oct_group_diagnosis --classes 0 1 2 3
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
COLORS = ["#e74c3c", "#3498db", "#27ae60", "#e67e22"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo_dir", type=str, default="./results/oct_loo_full")
    parser.add_argument("--output_dir", type=str, default="./results/oct_group_diagnosis")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--n_groups", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in args.classes:
        loo_path = Path(args.loo_dir) / f"loo_comprehensive_class{c}.json"
        if not loo_path.exists():
            logger.warning(f"No LOO data for class {c}")
            continue
        with open(loo_path) as f:
            points = json.load(f)["single_point_loo"]
        tracin = np.array([p["tracin_score"] for p in points])
        delta = np.array([p["delta_loss_mean"] for p in points])
        membership = np.array([p["membership"] for p in points])
        logger.info(f"\n{'='*60}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*60}")

        # KMeans 按 TracIn score 聚类，并按簇中心排序重映射 group 标签
        km = KMeans(n_clusters=args.n_groups, random_state=42, n_init=10)
        groups = km.fit_predict(tracin.reshape(-1, 1))
        order = np.argsort(km.cluster_centers_.flatten())
        label_map = {old: new for new, old in enumerate(order)}
        groups = np.array([label_map[g] for g in groups])

        sp_all, sp_p_all = spearmanr(tracin, delta)
        pe_all, pe_p_all = pearsonr(tracin, delta)
        logger.info(f"  Overall: Spearman r={sp_all:.4f} (p={sp_p_all:.4f}), Pearson r={pe_all:.4f}")

        # 逐组相关性
        group_results = []
        for g in range(args.n_groups):
            mask = groups == g
            n_g = int(mask.sum())
            if n_g < 3:
                group_results.append({"group": g, "n": n_g,
                                      "tracin_range": [float(tracin[mask].min()), float(tracin[mask].max())] if n_g else [],
                                      "member_ratio": float(membership[mask].mean()) if n_g else 0})
                logger.info(f"  Group {g}: only {n_g} points, skip")
                continue
            t_g, d_g = tracin[mask], delta[mask]
            sp_g, sp_p_g = spearmanr(t_g, d_g)
            pe_g, pe_p_g = pearsonr(t_g, d_g)
            group_results.append({"group": g, "n": n_g, "tracin_mean": float(t_g.mean()),
                                  "tracin_range": [float(t_g.min()), float(t_g.max())],
                                  "delta_mean": float(d_g.mean()), "member_ratio": float(membership[mask].mean()),
                                  "spearman_r": float(sp_g), "spearman_p": float(sp_p_g),
                                  "pearson_r": float(pe_g), "pearson_p": float(pe_p_g)})
            logger.info(f"  Group {g}: n={n_g}, member_ratio={membership[mask].mean():.2f}, "
                        f"Spearman r={sp_g:.4f} (p={sp_p_g:.4f})")

        # 画图: 总览 (按组着色) + 每组单独子图
        n_panels = 1 + args.n_groups
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        fig.suptitle(f"Per-Group LOO Diagnosis - Class {c} ({CLASS_NAMES.get(c, c)})\n"
                     f"Overall: Spearman r={sp_all:.3f}, Pearson r={pe_all:.3f}",
                     fontsize=13, fontweight="bold")
        ax = axes[0]
        for g in range(args.n_groups):
            mask = groups == g
            ax.scatter(tracin[mask], delta[mask], c=COLORS[g % len(COLORS)], s=40, alpha=0.7,
                       label=f"Group {g} (n={mask.sum()})", edgecolors="white", lw=0.5)
        ax.axhline(0, color="gray", ls="--", alpha=0.4); ax.axvline(0, color="gray", ls="--", alpha=0.4)
        ax.set_xlabel("TracIn Score"); ax.set_ylabel("Actual delta-loss")
        ax.set_title(f"All groups\nSpearman r={sp_all:.3f}"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        for g in range(args.n_groups):
            ax = axes[1 + g]
            mask = groups == g
            n_g = int(mask.sum())
            t_g, d_g = tracin[mask], delta[mask]
            ax.scatter(t_g, d_g, c=["#e74c3c" if m == 1 else "#3498db" for m in membership[mask]],
                       s=50, alpha=0.7, edgecolors="white", lw=0.5)
            ax.axhline(0, color="gray", ls="--", alpha=0.4)
            if n_g >= 3:
                gr = group_results[g]
                if n_g > 2 and np.std(t_g) > 1e-6:
                    z = np.polyfit(t_g, d_g, 1)
                    xl = np.linspace(t_g.min(), t_g.max(), 50)
                    ax.plot(xl, np.polyval(z, xl), "k--", alpha=0.5)
                ax.set_title(f"Group {g} (n={n_g})\nSpearman r={gr.get('spearman_r', 0):.3f} "
                             f"(p={gr.get('spearman_p', 1):.3f})")
            else:
                ax.set_title(f"Group {g} (n={n_g})\ntoo few points")
            ax.set_xlabel("TracIn Score"); ax.set_ylabel("Actual delta-loss"); ax.grid(True, alpha=0.3)
            ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Member'),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Non-member')],
                      fontsize=7, loc="upper left")
        plt.tight_layout()
        fig.savefig(str(out_dir / f"group_diagnosis_class{c}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {out_dir / f'group_diagnosis_class{c}.png'}")

        with open(out_dir / f"group_diagnosis_class{c}.json", "w") as f:
            json.dump({"class": int(c), "class_name": CLASS_NAMES.get(c, str(c)),
                       "overall_spearman_r": float(sp_all), "overall_spearman_p": float(sp_p_all),
                       "overall_pearson_r": float(pe_all), "overall_pearson_p": float(pe_p_all),
                       "loss_function": "CrossEntropyLoss", "groups": group_results}, f, indent=2)

    print("\nDone. Outputs in", args.output_dir)


if __name__ == "__main__":
    main()
