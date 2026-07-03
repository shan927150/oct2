#!/usr/bin/env python3
"""
LOO 簇内 (within-cluster) 相关性分析 [ours, 回应导师追问]。

配合 between-cluster 分析使用：把 LOO 点沿 TracIn 轴聚成带后，
分别在每个带内部算 TracIn-vs-Δloss 的 Spearman/Pearson，检验带内是否存在关系。

预期（也是要验证的点）：带内 TracIn 近似常数（这正是"竖线"的含义），
所以带内相关本质是拿几乎不变的 x 去相关 Δloss 噪声 —— r 应当不稳定、不显著、
符号在各带之间来回变。若结果如此，说明（微弱的）信号全在带与带之间，带内是噪声。

聚类逻辑与 loo_cluster_trend(_v2).py 一致，便于两者对照。
纯 CPU、读现成 JSON，可直接在登录节点跑。

用法:
  python loo_within_cluster_corr.py \
      --loo_dir ./results/oct_loo_full \
      --output_dir ./results/oct_within_cluster \
      --classes 0 1 2 3 --n_clusters 6
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans

np.seterr(all="ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
MIN_N = 3                 # 算相关所需的最少点数
TRACIN_STD_EPS = 1e-9     # TracIn 方差低于此值视为常数，相关无意义
PALETTE = ["#e74c3c", "#3498db", "#27ae60", "#e67e22", "#8e44ad", "#16a085",
           "#f39c12", "#2c3e50", "#c0392b", "#2980b9"]


def _effective_k(tracin, requested_k):
    """重复值多时把 k 卡在 unique TracIn 数以内，避免 KMeans 退化/警告。"""
    unique_x = np.unique(np.round(tracin, 8))
    return max(1, min(int(requested_k), len(tracin), len(unique_x)))


def _cluster_on_tracin(tracin, requested_k, seed):
    """1D KMeans on TracIn，按簇中心从负到正重排标签。"""
    k = _effective_k(tracin, requested_k)
    if k == 1:
        return np.zeros(len(tracin), dtype=int), 1
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    raw = km.fit_predict(tracin.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[g] for g in raw], dtype=int), k


def process_class(class_id, loo_dir, out_dir, n_clusters, seed, make_plot=True):
    loo_path = Path(loo_dir) / f"loo_comprehensive_class{class_id}.json"
    if not loo_path.exists():
        logger.warning("No LOO data for class %s at %s", class_id, loo_path)
        return None
    pts = json.load(open(loo_path))["single_point_loo"]
    tracin = np.array([p["tracin_score"] for p in pts], dtype=float)
    delta = np.array([p["delta_loss_mean"] for p in pts], dtype=float)
    membership = np.array([p["membership"] for p in pts], dtype=int)

    labels, k = _cluster_on_tracin(tracin, n_clusters, seed)

    clusters, usable_r, usable_w, n_sig = [], [], [], 0
    for g in range(k):
        m = labels == g
        n = int(m.sum())
        if n == 0:
            continue
        tm, ts = float(tracin[m].mean()), float(tracin[m].std())
        ds = float(delta[m].std())
        rec = {"cluster": int(g), "n": n, "tracin_mean": tm, "tracin_std": ts,
               "delta_std": ds, "member_ratio": float(membership[m].mean()),
               "within_spearman_r": None, "within_spearman_p": None,
               "within_pearson_r": None, "within_pearson_p": None, "usable": False}
        if n >= MIN_N and ts > TRACIN_STD_EPS and ds > 1e-12:
            sr, sp = spearmanr(tracin[m], delta[m])
            pr, pp = pearsonr(tracin[m], delta[m])
            rec.update({"within_spearman_r": float(sr), "within_spearman_p": float(sp),
                        "within_pearson_r": float(pr), "within_pearson_p": float(pp),
                        "usable": True})
            usable_r.append(sr); usable_w.append(n)
            if sp < 0.05:
                n_sig += 1
        clusters.append(rec)

    usable_r = np.array(usable_r); usable_w = np.array(usable_w, dtype=float)
    weighted_mean = float(np.sum(usable_w * usable_r) / np.sum(usable_w)) if len(usable_r) else float("nan")
    signs = set(np.sign(usable_r[np.abs(usable_r) > 1e-9]).tolist()) if len(usable_r) else set()
    sign_consistent = len(signs) <= 1

    logger.info("\n  CLASS %s (%s)  k=%d", class_id, CLASS_NAMES.get(class_id, class_id), k)
    logger.info("    %2s%5s%13s%12s%14s%9s%7s", "cl", "n", "TracIn_mean", "TracIn_std",
                "within Sp r", "p", "memb")
    for cc in clusters:
        if cc["usable"]:
            rs, ps = f"{cc['within_spearman_r']:+.3f}", f"{cc['within_spearman_p']:.3f}"
        else:
            rs, ps = "n/a", "-"
        logger.info("    %2d%5d%13.3f%12.4f%14s%9s%7.2f", cc["cluster"], cc["n"],
                    cc["tracin_mean"], cc["tracin_std"], rs, ps, cc["member_ratio"])
    logger.info("    -> size-weighted mean within-cluster Spearman = %s  (%d usable clusters; %d significant; signs %s)",
                f"{weighted_mean:+.3f}" if not np.isnan(weighted_mean) else "n/a",
                len(usable_r), n_sig, "consistent" if sign_consistent else "FLIP")

    fig_path = None
    if make_plot:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        for cc in clusters:
            g = cc["cluster"]; m = labels == g
            col = PALETTE[g % len(PALETTE)]
            ax.scatter(tracin[m], delta[m], c=col, s=30, alpha=0.7, edgecolors="white", lw=0.4)
            # 在簇上方标注簇内 Spearman（可用时）
            if cc["usable"]:
                txt = f"r={cc['within_spearman_r']:+.2f}\np={cc['within_spearman_p']:.2f}"
            else:
                txt = f"n={cc['n']}\n(n/a)"
            ax.annotate(txt, (cc["tracin_mean"], delta[m].max()), fontsize=7, ha="center",
                        va="bottom", color=col,
                        xytext=(0, 4), textcoords="offset points")
            # 簇内最佳拟合线（仅可用簇）
            if cc["usable"] and cc["tracin_std"] > TRACIN_STD_EPS:
                z = np.polyfit(tracin[m], delta[m], 1)
                xl = np.linspace(tracin[m].min(), tracin[m].max(), 20)
                ax.plot(xl, np.polyval(z, xl), color=col, ls="-", lw=1.0, alpha=0.6)
        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("TracIn Score"); ax.set_ylabel("Δloss (LOO - baseline)")
        ax.set_title(f"Within-cluster correlation — Class {class_id} ({CLASS_NAMES.get(class_id, class_id)}), k={k}\n"
                     f"size-weighted mean within-cluster Spearman = "
                     f"{weighted_mean:+.3f}" + ("" if not np.isnan(weighted_mean) else "n/a") +
                     f"  ({len(usable_r)} usable, {n_sig} sig, signs {'consistent' if sign_consistent else 'FLIP'})",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        fig_path = str(Path(out_dir) / f"within_cluster_class{class_id}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("    Plot saved: %s", fig_path)

    return {"class": int(class_id), "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "n_clusters": k, "weighted_mean_within_spearman": weighted_mean,
            "n_usable_clusters": int(len(usable_r)), "n_significant": int(n_sig),
            "sign_consistent": bool(sign_consistent), "clusters": clusters, "plot": fig_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo_dir", type=str, default="./results/oct_loo_full")
    ap.add_argument("--output_dir", type=str, default="./results/oct_within_cluster")
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c in args.classes:
        r = process_class(c, args.loo_dir, out_dir, args.n_clusters, args.seed)
        if r:
            summary[str(c)] = r
    json.dump(summary, open(out_dir / "within_cluster_summary.json", "w"), indent=2)

    print(f"\n{'='*96}")
    print("  WITHIN-CLUSTER CORRELATION SUMMARY")
    print("  (each LOO cluster has near-constant TracIn; within-cluster r should be unstable noise)")
    print(f"{'='*96}")
    print(f"  {'Class':<8}{'k':>4}{'usable':>8}{'#sig':>6}{'signs':>12}{'wMean within-Sp':>18}")
    for c in args.classes:
        r = summary.get(str(c))
        if not r:
            continue
        wm = r["weighted_mean_within_spearman"]
        print(f"  {r['class_name']:<8}{r['n_clusters']:>4}{r['n_usable_clusters']:>8}{r['n_significant']:>6}"
              f"{('consistent' if r['sign_consistent'] else 'FLIP'):>12}"
              f"{(f'{wm:+.3f}' if not np.isnan(wm) else 'n/a'):>18}")
    print(f"{'='*96}")
    print(f"  Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
