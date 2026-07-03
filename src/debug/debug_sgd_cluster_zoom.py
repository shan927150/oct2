#!/usr/bin/env python
"""
debug_sgd_cluster_zoom.py  (Experiment 1)

For each class and each TracIn vertical band: zoom in to test whether there is a
hidden within-cluster slope or just a vertical smear.

Per cluster produces a 3-panel figure:
  A. raw x zoom           (x = TracIn score)
  B. centered x           (x = TracIn - cluster_mean)
  C. jitter (viz only)    (x + small noise; correlation NOT computed on jitter)

Per-cluster stats include the metric the design flags as most critical:
  unique TracIn scores count + duplicate score ratio  (low unique => no x-resolution).
Also: x/y min/max/range, Spearman r/p, Pearson r/p, OLS slope, Theil-Sen robust slope.

Pure JSON reader (no torch/GPU).

  python debug_sgd_cluster_zoom.py \
    --loo_dir results/oct_loo_full_sgd \
    --output_dir results/oct_debug_sgd/cluster_zoom \
    --classes 0 1 2 3 --n_clusters 6
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress, theilslopes
from sklearn.cluster import KMeans

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def cluster_stats(x, y, member, min_n):
    n = len(x)
    uniq = np.unique(np.round(x, 6))
    rec = dict(
        n=n, member_ratio=float(member.mean()) if n else None,
        x_min=float(x.min()) if n else None, x_max=float(x.max()) if n else None,
        x_range=float(x.max() - x.min()) if n else 0.0,
        y_min=float(y.min()) if n else None, y_max=float(y.max()) if n else None,
        y_range=float(y.max() - y.min()) if n else 0.0,
        n_unique_scores=int(len(uniq)),
        duplicate_ratio=float(1.0 - len(uniq) / n) if n else None,
        spearman_r=None, spearman_p=None, pearson_r=None, pearson_p=None,
        ols_slope=None, theilsen_slope=None,
    )
    if n >= min_n and rec["x_range"] > 1e-9 and len(uniq) >= 2:
        sr, sp = spearmanr(x, y); pr, pp = pearsonr(x, y)
        ols = linregress(x, y)
        ts = theilslopes(y, x)
        rec.update(spearman_r=float(sr), spearman_p=float(sp),
                   pearson_r=float(pr), pearson_p=float(pp),
                   ols_slope=float(ols.slope), theilsen_slope=float(ts[0]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--min_n", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows, summary = [], {}
    for c in args.classes:
        path = os.path.join(args.loo_dir, f"loo_comprehensive_class{c}.json")
        if not os.path.exists(path):
            print(f"skip class {c}: {path} missing"); continue
        d = json.load(open(path))
        pts = d["single_point_loo"]
        tracin = np.array([p["tracin_score"] for p in pts], float)
        dloss = np.array([p["delta_loss_mean"] for p in pts], float)
        dstd = np.array([p["delta_loss_std"] for p in pts], float)
        member = np.array([p["membership"] for p in pts], int)
        name = d.get("class_name", CLASS_NAMES[c] if c < 4 else str(c))

        labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0).fit_predict(tracin.reshape(-1, 1))
        order = np.argsort([tracin[labels == k].mean() if (labels == k).any() else np.inf
                            for k in range(args.n_clusters)])

        print(f"\n=== Class {c} ({name}) ===")
        summary[str(c)] = dict(class_name=name, clusters=[])
        rng = np.random.default_rng(0)
        for k in order:
            m = labels == k
            if not m.any():
                continue
            x, y, e, mb = tracin[m], dloss[m], dstd[m], member[m]
            rec = cluster_stats(x, y, mb, args.min_n)
            rec["class"] = int(c); rec["cluster"] = int(k)
            all_rows.append(rec); summary[str(c)]["clusters"].append(rec)
            print(f"  clu{k}: n={rec['n']:2d} memb={rec['member_ratio']:.0%} "
                  f"x_range={rec['x_range']:6.2f} uniq={rec['n_unique_scores']:2d} "
                  f"dup={rec['duplicate_ratio']:.2f} spearman={rec['spearman_r']} "
                  f"theilsen={rec['theilsen_slope']}")

            # 3-panel zoom figure for this cluster
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
            col = "tab:red" if (rec["member_ratio"] or 0) > 0.5 else "tab:blue"
            # A raw
            ax[0].errorbar(x, y, yerr=e, fmt="o", ms=6, capsize=3, alpha=0.8, color=col)
            # B centered
            ax[1].errorbar(x - x.mean(), y, yerr=e, fmt="o", ms=6, capsize=3, alpha=0.8, color=col)
            # C jitter (viz only)
            jit = (rng.standard_normal(len(x)) * (rec["x_range"] or 1.0) * 0.05)
            ax[2].scatter(x + jit, y, s=40, alpha=0.7, color=col)
            for a in ax:
                a.set_ylabel("delta loss"); a.grid(alpha=0.3)
            ax[0].set_xlabel("TracIn score"); ax[0].set_title("A. raw x zoom")
            ax[1].set_xlabel("TracIn - cluster_mean"); ax[1].set_title("B. centered x")
            ax[2].set_xlabel("TracIn score + jitter (viz only)"); ax[2].set_title("C. jitter")
            if rec["ols_slope"] is not None:
                xs = np.linspace(x.min(), x.max(), 50)
                ax[0].plot(xs, linregress(x, y).intercept + rec["ols_slope"] * xs, "k--", lw=1.3)
            fig.suptitle(f"Class {c} ({name}) cluster {k}: n={rec['n']}, memb={rec['member_ratio']:.0%}, "
                         f"unique_scores={rec['n_unique_scores']}, x_range={rec['x_range']:.2f}, "
                         f"Spearman={rec['spearman_r']} (p={rec['spearman_p']})",
                         fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(args.output_dir, f"class{c}_cluster{k}_zoom.png"), dpi=120)
            plt.close(fig)

    # CSV + JSON
    csv_path = os.path.join(args.output_dir, "cluster_zoom_summary.csv")
    cols = ["class", "cluster", "n", "member_ratio", "x_min", "x_max", "x_range",
            "y_min", "y_max", "y_range", "n_unique_scores", "duplicate_ratio",
            "spearman_r", "spearman_p", "pearson_r", "pearson_p", "ols_slope", "theilsen_slope"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in cols})
    json.dump(summary, open(os.path.join(args.output_dir, "cluster_zoom_summary.json"), "w"), indent=2)
    print(f"\nSaved per-cluster pngs + {csv_path} + cluster_zoom_summary.json")
    print("Read: a cluster with many points but n_unique_scores=1-3 (duplicate_ratio near 1) "
          "has no x-resolution for within-cluster ranking -> the signal is between bands only.")


if __name__ == "__main__":
    main()
