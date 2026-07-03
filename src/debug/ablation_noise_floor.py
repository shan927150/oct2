#!/usr/bin/env python
"""
ablation_noise_floor.py  (Ablation 3 core - cheapest, most decisive)

Question: within a TracIn band, is the point-to-point variation in delta_loss
ABOVE the single-point LOO retraining noise floor? If not, within-band ranking is
fundamentally unresolvable, no matter how good the attribution method is.

Each LOO point already has delta_loss_mean (mean over n_seeds) and delta_loss_std
(std over those seeds). So per band B:
    between_point_var = Var over points of delta_loss_mean        (real spread of means)
    noise_floor_var   = mean over points of (delta_loss_std^2)/n_seeds
                        (sampling variance of each point's mean estimate)
    snr = between_point_var / noise_floor_var
snr ~ 1  -> the spread of band means is explained entirely by seed noise (no signal)
snr >> 1 -> there is real point-to-point variation to potentially rank

Pure JSON reader (no torch/GPU).

  python ablation_noise_floor.py --loo_dir results/oct_loo_full_sgd \
    --output_dir results/oct_debug_sgd/noise_floor --classes 0 1 2 3 --n_clusters 6
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--min_n", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for c in args.classes:
        p = os.path.join(args.loo_dir, f"loo_comprehensive_class{c}.json")
        if not os.path.exists(p):
            print(f"skip class {c}: {p} missing"); continue
        d = json.load(open(p))
        n_seeds = int(d.get("n_seeds", 5))
        pts = d["single_point_loo"]
        tracin = np.array([q["tracin_score"] for q in pts], float)
        dlmean = np.array([q["delta_loss_mean"] for q in pts], float)
        dlstd = np.array([q["delta_loss_std"] for q in pts], float)
        member = np.array([q["membership"] for q in pts], int)
        name = d.get("class_name", CLASS_NAMES[c] if c < 4 else str(c))

        labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0).fit_predict(tracin.reshape(-1, 1))
        order = np.argsort([tracin[labels == k].mean() if (labels == k).any() else np.inf
                            for k in range(args.n_clusters)])

        print(f"\n=== Class {c} ({name})  (n_seeds={n_seeds}) ===")
        print(f"  {'band':>4} {'n':>3} {'memb%':>6} {'between_var':>12} {'noise_floor':>12} {'SNR':>7} verdict")
        # also a class-wide (all 50 points) SNR
        for k in list(order) + ["ALL"]:
            if k == "ALL":
                m = np.ones(len(pts), bool); n = len(pts); tag = "ALL"
            else:
                m = labels == k; n = int(m.sum()); tag = str(int(k))
            if n < args.min_n:
                continue
            between = float(np.var(dlmean[m], ddof=1))
            noise = float(np.mean(dlstd[m] ** 2) / n_seeds)
            snr = between / noise if noise > 0 else float("inf")
            verdict = "NO within-band signal" if snr < 2 else ("weak" if snr < 5 else "some signal")
            print(f"  {tag:>4} {n:>3} {member[m].mean():>6.0%} {between:>12.2e} "
                  f"{noise:>12.2e} {snr:>7.2f}  {verdict}")
            rows.append(dict(cls=c, class_name=name, band=tag, n=n,
                             member_ratio=float(member[m].mean()),
                             between_point_var=between, noise_floor_var=noise,
                             snr=snr, verdict=verdict))

        # plot: per-band SNR bars
        bands = [r for r in rows if r["cls"] == c and r["band"] != "ALL"]
        if bands:
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = [f"b{r['band']}\n{r['member_ratio']:.0%}m" for r in bands]
            ax.bar(xs, [r["snr"] for r in bands],
                   color=["tab:red" if r["member_ratio"] > 0.5 else "tab:blue" for r in bands])
            ax.axhline(1, color="gray", ls="--", lw=1, label="noise floor (SNR=1)")
            ax.axhline(2, color="orange", ls=":", lw=1, label="SNR=2 threshold")
            ax.set_ylabel("between-point var / noise-floor var (SNR)")
            ax.set_title(f"Class {c} ({name}): within-band signal-to-noise")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, f"noise_floor_class{c}.png"), dpi=120)
            plt.close(fig)

    with open(os.path.join(args.output_dir, "noise_floor_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved noise_floor_summary.csv + per-class bars")
    print("Read: SNR ~ 1 means within-band delta_loss spread is just retraining noise -> "
          "within-band ranking is unresolvable (the decisive form of conclusion A).")


if __name__ == "__main__":
    main()
