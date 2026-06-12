#!/usr/bin/env python3
"""
实验: LOO scatter 的 outlier 移除 + 相关性重算 [ours]。

读现有 loo_comprehensive_class{c}.json (50 个 LOO 点)，用两种方法移走 outlier，
分别重画 TracIn-vs-Δloss scatter 并重算 Spearman/Pearson，看相关性是否变干净:

  方法 1 (IQR):     按 Δloss 的 1.5×IQR 删统计离群点 (去掉异常大的 LOO 抖动)
  方法 2 (cluster): KMeans 按 TracIn score 聚成 n_clusters 组，删掉某个极端簇
                    (默认删 center 最负的簇，如 DME 最左 TracIn≈-58 那组)

纯 CPU、读小 JSON，可在现有数据上直接跑，也可在 l2=0 重跑后复用。

用法:
  python plot_loo_outliers.py --loo_dir ./results/oct_loo_full \
      --output_dir ./results/oct_loo_outlier --classes 0 1 2 3
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


def _corr(t, d):
    """安全算 Spearman/Pearson (点数 < 3 或零方差时返回 nan)。"""
    if len(t) < 3 or np.std(t) < 1e-12 or np.std(d) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sp_r, sp_p = spearmanr(t, d)
    pe_r, pe_p = pearsonr(t, d)
    return float(sp_r), float(sp_p), float(pe_r), float(pe_p)


def _iqr_keep_mask(delta, k=1.5):
    """方法 1: 按 Δloss 的 k×IQR 标记保留点 (True=保留)。"""
    q1, q3 = np.percentile(delta, [25, 75])
    iqr = q3 - q1
    return (delta >= q1 - k * iqr) & (delta <= q3 + k * iqr)


def _cluster_drop_mask(tracin, n_clusters=4, drop="extreme_low", seed=42):
    """方法 2: KMeans 按 TracIn 聚类，标记保留点 (True=保留)。
    drop: 'extreme_low' (center 最负簇) / 'extreme_high' (最正簇) /
          'both' (两端簇) / 整数 (按 center 升序的簇序号)。
    返回 (keep_mask, dropped_cluster_label, cluster_centers_sorted)。
    """
    n = len(tracin)
    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    raw = km.fit_predict(tracin.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.flatten())   # 簇按 center 升序
    label_map = {old: new for new, old in enumerate(order)}
    groups = np.array([label_map[g] for g in raw])      # 0=最负 … k-1=最正

    if drop == "extreme_low":
        drop_ids = [0]
    elif drop == "extreme_high":
        drop_ids = [k - 1]
    elif drop == "both":
        drop_ids = [0, k - 1]
    else:
        drop_ids = [int(drop)]

    keep = ~np.isin(groups, drop_ids)
    centers_sorted = km.cluster_centers_.flatten()[order]
    return keep, drop_ids, groups, centers_sorted


def _scatter(ax, t, d, membership, title, keep_mask=None):
    """画 TracIn-vs-Δloss scatter + 趋势线；keep_mask 给出的被删点画成灰色。"""
    colors = np.where(membership == 1, "#e74c3c", "#3498db")
    if keep_mask is None:
        keep_mask = np.ones(len(t), dtype=bool)
    # 被删点: 灰色空心
    ax.scatter(t[~keep_mask], d[~keep_mask], facecolors="none", edgecolors="#bbbbbb",
               s=45, lw=1.0, label="removed", zorder=2)
    # 保留点: 按 membership 着色
    ax.scatter(t[keep_mask], d[keep_mask], c=colors[keep_mask], s=45, alpha=0.8,
               edgecolors="white", lw=0.5, zorder=3)
    tk, dk = t[keep_mask], d[keep_mask]
    if len(tk) > 2 and np.std(tk) > 1e-9:   # 趋势线只用保留点拟合
        z = np.polyfit(tk, dk, 1)
        xl = np.linspace(tk.min(), tk.max(), 50)
        ax.plot(xl, np.polyval(z, xl), "k--", alpha=0.6, lw=1.2)
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.axvline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("TracIn Score"); ax.set_ylabel("Δloss (LOO - baseline)")
    ax.set_title(title); ax.grid(True, alpha=0.3)


def process_class(c, loo_dir, out_dir, iqr_k, n_clusters, drop_cluster, seed):
    """对一个 class 跑两种 outlier 移除并出图，返回结果 dict。"""
    loo_path = Path(loo_dir) / f"loo_comprehensive_class{c}.json"
    if not loo_path.exists():
        logger.warning(f"No LOO data for class {c} at {loo_path}")
        return None
    with open(loo_path) as f:
        points = json.load(f)["single_point_loo"]
    tracin = np.array([p["tracin_score"] for p in points])
    delta = np.array([p["delta_loss_mean"] for p in points])
    membership = np.array([p["membership"] for p in points])
    n = len(tracin)

    # 原始 + 两种移除后的相关性
    sp0, sp0p, pe0, pe0p = _corr(tracin, delta)
    iqr_keep = _iqr_keep_mask(delta, k=iqr_k)
    sp1, sp1p, pe1, pe1p = _corr(tracin[iqr_keep], delta[iqr_keep])
    clu_keep, drop_ids, groups, centers = _cluster_drop_mask(tracin, n_clusters, drop_cluster, seed)
    sp2, sp2p, pe2, pe2p = _corr(tracin[clu_keep], delta[clu_keep])

    logger.info(f"\n  CLASS {c} ({CLASS_NAMES.get(c, c)}) n={n}")
    logger.info(f"    original           : Spearman r={sp0:+.3f} (p={sp0p:.3f}), Pearson r={pe0:+.3f}")
    logger.info(f"    IQR(k={iqr_k}) kept {int(iqr_keep.sum())}/{n}: Spearman r={sp1:+.3f} (p={sp1p:.3f})")
    logger.info(f"    drop cluster {drop_ids} kept {int(clu_keep.sum())}/{n}: Spearman r={sp2:+.3f} (p={sp2p:.3f})")

    # 三联图: 原始 | IQR 移除 | 簇移除
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"LOO Outlier Removal — Class {c} ({CLASS_NAMES.get(c, c)}), n={n}",
                 fontsize=14, fontweight="bold")
    _scatter(axes[0], tracin, delta, membership,
             f"Original (all {n})\nSpearman r={sp0:+.3f} (p={sp0p:.3f})")
    _scatter(axes[1], tracin, delta, membership,
             f"IQR k={iqr_k}, kept {int(iqr_keep.sum())}\nSpearman r={sp1:+.3f} (p={sp1p:.3f})",
             keep_mask=iqr_keep)
    _scatter(axes[2], tracin, delta, membership,
             f"Drop cluster {drop_ids} (center≈{centers[drop_ids[0]]:.0f}), kept {int(clu_keep.sum())}\n"
             f"Spearman r={sp2:+.3f} (p={sp2p:.3f})", keep_mask=clu_keep)
    axes[0].legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Member'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Non-member'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#bbbbbb', markersize=8, label='Removed'),
    ], fontsize=8, loc="best")
    plt.tight_layout()
    fig_path = Path(out_dir) / f"loo_outlier_class{c}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    Plot saved: {fig_path}")

    return {
        "class": int(c), "class_name": CLASS_NAMES.get(c, str(c)), "n_total": int(n),
        "original": {"spearman_r": sp0, "spearman_p": sp0p, "pearson_r": pe0, "pearson_p": pe0p},
        "iqr": {"k": float(iqr_k), "n_kept": int(iqr_keep.sum()),
                "spearman_r": sp1, "spearman_p": sp1p, "pearson_r": pe1, "pearson_p": pe1p},
        "cluster": {"n_clusters": int(n_clusters), "dropped": [int(x) for x in drop_ids],
                    "dropped_center": float(centers[drop_ids[0]]), "n_kept": int(clu_keep.sum()),
                    "spearman_r": sp2, "spearman_p": sp2p, "pearson_r": pe2, "pearson_p": pe2p},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo_dir", type=str, default="./results/oct_loo_full")
    parser.add_argument("--output_dir", type=str, default="./results/oct_loo_outlier")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--iqr_k", type=float, default=1.5)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--drop_cluster", type=str, default="extreme_low",
                        help="extreme_low / extreme_high / both / 簇序号(按 center 升序)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for c in args.classes:
        r = process_class(c, args.loo_dir, out_dir, args.iqr_k, args.n_clusters, args.drop_cluster, args.seed)
        if r is not None:
            summary[c] = r

    with open(out_dir / "outlier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 汇总表: 原始 → IQR → 簇移除 的 Spearman r
    print(f"\n{'='*78}")
    print(f"  OUTLIER REMOVAL — Spearman r (original → IQR → drop-cluster)")
    print(f"{'='*78}")
    print(f"  {'Class':<10} {'orig':>10} {'IQR':>10} {'cluster':>10}   (p-values in JSON)")
    for c in args.classes:
        r = summary.get(c)
        if r is None:
            continue
        print(f"  {r['class_name']:<10} {r['original']['spearman_r']:>+10.3f} "
              f"{r['iqr']['spearman_r']:>+10.3f} {r['cluster']['spearman_r']:>+10.3f}")
    print(f"{'='*78}")
    print(f"  Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
