#!/usr/bin/env python3
"""
LOO cluster-level / between-cluster trend analysis for OCT MIA TracIn validation.

Goal
----
The LOO scatter plots often form vertical bands: within one band, TracIn is nearly
constant while Δloss varies. Therefore, point-level correlation mixes two effects:
  1) between-cluster trend: do cluster centroids show a monotonic/linear trend?
  2) within-cluster noise: how much Δloss varies when TracIn is almost fixed?

This script:
  - clusters LOO points along the 1D TracIn axis;
  - plots raw points, full vertical cluster ranges, centroid SEM error bars, and
    an unweighted centroid regression line;
  - reports raw 50-point correlations, centroid-level correlations, weighted
    linear trend, and variance decomposition η²;
  - saves cluster-level min/max/std statistics for explaining whether a cluster
    can be treated as a vertical line;
  - optionally runs sensitivity analysis over several k values.

Important interpretation notes
------------------------------
- η²(TracIn) is a sanity check because clusters are defined along TracIn.
  It should not be presented as independent proof of structure.
- The more direct "vertical line" evidence is small within-cluster TracIn std/range
  together with large within-cluster Δloss spread.
- Centroid p-values are underpowered when k is small (e.g., 4-8 centroids), so
  treat the between-cluster trend as exploratory.

Example
-------
python loo_cluster_trend_v2.py \
  --loo_dir ./results/oct_loo_full \
  --output_dir ./results/oct_cluster_trend_v2 \
  --classes 0 1 2 3 \
  --n_clusters 6

Sensitivity over k:
python loo_cluster_trend_v2.py \
  --loo_dir ./results/oct_loo_full \
  --output_dir ./results/oct_cluster_trend_sensitivity \
  --classes 0 1 2 3 \
  --k_sensitivity 4 5 6 7 8
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


def _corr(x, y):
    """Return Spearman r/p and Pearson r/p with guardrails for tiny/constant arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    return float(sr), float(sp), float(pr), float(pp)


def _eta_squared(values, labels):
    """η² = SS_between / SS_total, i.e., group-level share of total variance."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_between = 0.0
    for g in np.unique(labels):
        v = values[labels == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _weighted_linear_fit(x, y, n):
    """Size-weighted linear fit on cluster centroids. Returns slope/intercept/R2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12 or np.sum(n) <= 0:
        return float("nan"), float("nan"), float("nan")

    # np.polyfit expects weights multiplying unsquared residuals; sqrt(n) gives WLS by cluster size.
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(n))
    y_hat = slope * x + intercept
    w = n / np.sum(n)
    y_bar_w = np.sum(w * y)
    ss_res = np.sum(w * (y - y_hat) ** 2)
    ss_tot = np.sum(w * (y - y_bar_w) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _effective_k(tracin, requested_k):
    """Avoid KMeans warnings/failures when many TracIn scores are duplicated."""
    tracin = np.asarray(tracin, dtype=float)
    unique_x = np.unique(np.round(tracin, 8))
    return max(1, min(int(requested_k), len(tracin), len(unique_x)))


def _cluster_on_tracin(tracin, requested_k, seed):
    """Run 1D KMeans on TracIn and remap labels left-to-right by centroid."""
    k = _effective_k(tracin, requested_k)
    if k == 1:
        return np.zeros(len(tracin), dtype=int), 1

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    raw = km.fit_predict(tracin.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[g] for g in raw], dtype=int)
    return labels, k


def _load_loo_points(loo_dir, class_id):
    loo_path = Path(loo_dir) / f"loo_comprehensive_class{class_id}.json"
    if not loo_path.exists():
        logger.warning("No LOO data for class %s at %s", class_id, loo_path)
        return None

    with open(loo_path, "r") as f:
        data = json.load(f)
    pts = data["single_point_loo"]
    tracin = np.array([p["tracin_score"] for p in pts], dtype=float)
    delta = np.array([p["delta_loss_mean"] for p in pts], dtype=float)
    membership = np.array([p["membership"] for p in pts], dtype=int)
    return pts, tracin, delta, membership


def process_class(class_id, loo_dir, out_dir, n_clusters, seed, make_plot=True, suffix=""):
    loaded = _load_loo_points(loo_dir, class_id)
    if loaded is None:
        return None
    pts, tracin, delta, membership = loaded
    n_total = len(tracin)

    labels, k = _cluster_on_tracin(tracin, n_clusters, seed)

    clusters = []
    for g in range(k):
        m = labels == g
        if m.sum() == 0:
            continue
        clusters.append({
            "cluster": int(g),
            "n": int(m.sum()),
            "tracin_mean": float(tracin[m].mean()),
            "tracin_std": float(tracin[m].std()),
            "tracin_min": float(tracin[m].min()),
            "tracin_max": float(tracin[m].max()),
            "tracin_range": float(tracin[m].max() - tracin[m].min()),
            "delta_mean": float(delta[m].mean()),
            "delta_std": float(delta[m].std()),
            "delta_sem": float(delta[m].std() / max(np.sqrt(m.sum()), 1.0)),
            "delta_min": float(delta[m].min()),
            "delta_max": float(delta[m].max()),
            "delta_range": float(delta[m].max() - delta[m].min()),
            "member_ratio": float(membership[m].mean()),
        })

    cl_tracin = np.array([cc["tracin_mean"] for cc in clusters], dtype=float)
    cl_delta = np.array([cc["delta_mean"] for cc in clusters], dtype=float)
    cl_n = np.array([cc["n"] for cc in clusters], dtype=float)

    # Raw point-level and between-cluster centroid correlations.
    rsr, rsp, rpr, rpp = _corr(tracin, delta)
    bsr, bsp, bpr, bpp = _corr(cl_tracin, cl_delta)

    # Variance decomposition. η²(TracIn) is only a sanity check, not independent evidence.
    eta_delta = _eta_squared(delta, labels)
    eta_tracin = _eta_squared(tracin, labels)

    within_tracin_std_mean = float(np.mean([cc["tracin_std"] for cc in clusters]))
    within_tracin_range_mean = float(np.mean([cc["tracin_range"] for cc in clusters]))
    within_delta_std_mean = float(np.mean([cc["delta_std"] for cc in clusters]))
    within_delta_range_mean = float(np.mean([cc["delta_range"] for cc in clusters]))

    weighted_slope, weighted_intercept, weighted_r2 = _weighted_linear_fit(cl_tracin, cl_delta, cl_n)

    # Deconfounding diagnostic: member-only / non-member-only centroid correlation if enough clusters exist.
    def _membership_between(mem_val):
        mask = membership == mem_val
        if mask.sum() < 3:
            return None
        labs = labels[mask]
        uniq = np.unique(labs)
        if len(uniq) < 3:
            return None
        ct = np.array([tracin[mask][labs == g].mean() for g in uniq], dtype=float)
        cd = np.array([delta[mask][labs == g].mean() for g in uniq], dtype=float)
        cn = np.array([(labs == g).sum() for g in uniq], dtype=float)
        sr, sp, pr, pp = _corr(ct, cd)
        ws, wi, wr2 = _weighted_linear_fit(ct, cd, cn)
        return {
            "n_clusters": int(len(uniq)),
            "spearman_r": sr,
            "spearman_p": sp,
            "pearson_r": pr,
            "pearson_p": pp,
            "weighted_slope": ws,
            "weighted_intercept": wi,
            "weighted_r2": wr2,
        }

    mem_between = _membership_between(1)
    nonmem_between = _membership_between(0)

    logger.info("\n  CLASS %s (%s) n=%d, requested_k=%d, effective_k=%d",
                class_id, CLASS_NAMES.get(class_id, class_id), n_total, n_clusters, k)
    logger.info("    raw 50-pt       : Spearman r=%+.3f (p=%.3f), Pearson r=%+.3f (p=%.3f)",
                rsr, rsp, rpr, rpp)
    logger.info("    between-cluster : Spearman r=%+.3f (p=%.3f), Pearson r=%+.3f (p=%.3f) [%d centroids]",
                bsr, bsp, bpr, bpp, k)
    logger.info("    weighted trend  : slope=%+.6f, R2=%.3f (cluster-size weighted)",
                weighted_slope, weighted_r2)
    logger.info("    eta2(Δloss)=%.3f  eta2(TracIn)=%.3f [TracIn eta2 is a sanity check]",
                eta_delta, eta_tracin)
    logger.info("    within-cluster  : TracIn std=%.4f, TracIn range=%.4f, Δloss std=%.4f, Δloss range=%.4f",
                within_tracin_std_mean, within_tracin_range_mean,
                within_delta_std_mean, within_delta_range_mean)

    fig_path = None
    if make_plot:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6.5))

        colors = np.where(membership == 1, "#e74c3c", "#3498db")
        ax.scatter(tracin, delta, c=colors, s=22, alpha=0.30,
                   edgecolors="none", zorder=1)

        # Full vertical range per cluster: this directly visualizes the teacher's "vertical line" idea.
        for cc in clusters:
            x = cc["tracin_mean"]
            ax.plot([x, x], [cc["delta_min"], cc["delta_max"]],
                    color="#888888", lw=1.0, alpha=0.75, zorder=2)

        # Centroid uncertainty: SEM of Δloss within the cluster.
        ax.errorbar(
            cl_tracin,
            cl_delta,
            yerr=[cc["delta_sem"] for cc in clusters],
            fmt="o",
            ms=0,
            ecolor="#333333",
            elinewidth=1.5,
            capsize=4,
            zorder=3,
        )
        ax.scatter(
            cl_tracin,
            cl_delta,
            s=40 + 12 * cl_n,
            c="#222222",
            marker="D",
            edgecolors="white",
            lw=1.0,
            zorder=4,
            label="cluster centroid (size∝n)",
        )

        # Unweighted centroid regression line for visual trend; weighted slope is reported in JSON/table.
        if len(cl_tracin) > 2 and np.std(cl_tracin) > 1e-9:
            slope, intercept = np.polyfit(cl_tracin, cl_delta, 1)
            xl = np.linspace(cl_tracin.min(), cl_tracin.max(), 50)
            ax.plot(xl, slope * xl + intercept, "k--", alpha=0.8, lw=1.5, zorder=3)

        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.axvline(0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("TracIn Score")
        ax.set_ylabel("Δloss (LOO - baseline)")
        ax.set_title(
            f"Between-cluster trend — Class {class_id} ({CLASS_NAMES.get(class_id, class_id)}), k={k}\n"
            f"centroid Spearman r={bsr:+.3f} (p={bsp:.3f}) | "
            f"η²(Δloss)={eta_delta:.2f}, within-x range={within_tracin_range_mean:.2g}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=7, label="Member (raw)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                   markersize=7, label="Non-member (raw)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#222222",
                   markersize=8, label="Cluster centroid"),
            Line2D([0], [0], color="#888888", lw=1,
                   label="within-cluster full Δloss range"),
        ], fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

        fig_path = out_dir / f"cluster_trend_class{class_id}{suffix}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("    Plot saved: %s", fig_path)

    return {
        "class": int(class_id),
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "n_total": int(n_total),
        "requested_n_clusters": int(n_clusters),
        "n_clusters": int(k),
        "raw_spearman_r": rsr,
        "raw_spearman_p": rsp,
        "raw_pearson_r": rpr,
        "raw_pearson_p": rpp,
        "between_cluster_spearman_r": bsr,
        "between_cluster_spearman_p": bsp,
        "between_cluster_pearson_r": bpr,
        "between_cluster_pearson_p": bpp,
        "weighted_slope": weighted_slope,
        "weighted_intercept": weighted_intercept,
        "weighted_r2": weighted_r2,
        "eta2_delta": eta_delta,
        "eta2_tracin": eta_tracin,
        "within_cluster_tracin_std_mean": within_tracin_std_mean,
        "within_cluster_tracin_range_mean": within_tracin_range_mean,
        "within_cluster_delta_std_mean": within_delta_std_mean,
        "within_cluster_delta_range_mean": within_delta_range_mean,
        "member_only_between": mem_between,
        "nonmember_only_between": nonmem_between,
        "plot": str(fig_path) if fig_path is not None else None,
        "clusters": clusters,
    }


def _print_summary(summary, title, output_dir):
    print(f"\n{'=' * 124}")
    print(f"  {title}")
    print("  raw Sp = raw 50-point Spearman; btw Sp = centroid-level Spearman; wSlope = cluster-size weighted slope")
    print(f"{'=' * 124}")
    print(
        f"  {'Class':<8}{'k':>4}{'raw Sp':>9}{'btw Sp':>10}{'btw p':>9}"
        f"{'wSlope':>12}{'wR2':>8}{'η²Δ':>8}{'xStd':>10}{'xRange':>10}{'yStd':>10}"
    )
    for _, r in sorted(summary.items(), key=lambda kv: int(kv[0])):
        print(
            f"  {r['class_name']:<8}{r['n_clusters']:>4}"
            f"{r['raw_spearman_r']:>+9.3f}"
            f"{r['between_cluster_spearman_r']:>+10.3f}"
            f"{r['between_cluster_spearman_p']:>9.3f}"
            f"{r['weighted_slope']:>+12.5f}"
            f"{r['weighted_r2']:>8.2f}"
            f"{r['eta2_delta']:>8.2f}"
            f"{r['within_cluster_tracin_std_mean']:>10.3g}"
            f"{r['within_cluster_tracin_range_mean']:>10.3g}"
            f"{r['within_cluster_delta_std_mean']:>10.4f}"
        )
    print(f"{'=' * 124}")
    print(f"  Outputs in {output_dir}")


def run_single(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for c in args.classes:
        r = process_class(
            class_id=c,
            loo_dir=args.loo_dir,
            out_dir=out_dir,
            n_clusters=args.n_clusters,
            seed=args.seed,
            make_plot=True,
            suffix="",
        )
        if r is not None:
            summary[str(c)] = r

    with open(out_dir / "cluster_trend_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(
        summary,
        title="BETWEEN-CLUSTER TREND",
        output_dir=args.output_dir,
    )


def run_sensitivity(args):
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for k in args.k_sensitivity:
        k_out = base_out / f"k{k}"
        k_out.mkdir(parents=True, exist_ok=True)
        logger.info("\n%s\nRunning k-sensitivity setting k=%d\n%s", "=" * 80, k, "=" * 80)
        summary = {}
        for c in args.classes:
            r = process_class(
                class_id=c,
                loo_dir=args.loo_dir,
                out_dir=k_out,
                n_clusters=k,
                seed=args.seed,
                make_plot=True,
                suffix=f"_k{k}",
            )
            if r is not None:
                summary[str(c)] = r
        with open(k_out / "cluster_trend_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_results[str(k)] = summary
        _print_summary(summary, title=f"BETWEEN-CLUSTER TREND | k={k}", output_dir=str(k_out))

    # Compact sensitivity table: one row per k/class.
    rows = []
    for k, summary in all_results.items():
        for c, r in summary.items():
            rows.append({
                "k_requested": int(k),
                "class": int(c),
                "class_name": r["class_name"],
                "k_effective": r["n_clusters"],
                "raw_spearman_r": r["raw_spearman_r"],
                "between_cluster_spearman_r": r["between_cluster_spearman_r"],
                "between_cluster_spearman_p": r["between_cluster_spearman_p"],
                "weighted_slope": r["weighted_slope"],
                "weighted_r2": r["weighted_r2"],
                "eta2_delta": r["eta2_delta"],
                "within_cluster_tracin_range_mean": r["within_cluster_tracin_range_mean"],
                "within_cluster_delta_std_mean": r["within_cluster_delta_std_mean"],
            })

    with open(base_out / "cluster_trend_sensitivity_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'=' * 124}")
    print("  K-SENSITIVITY COMPACT SUMMARY")
    print(f"{'=' * 124}")
    print(f"  {'k':>3}{'Class':>9}{'raw Sp':>9}{'btw Sp':>10}{'btw p':>9}{'wSlope':>12}{'wR2':>8}{'η²Δ':>8}")
    for r in rows:
        print(
            f"  {r['k_requested']:>3}{r['class_name']:>9}"
            f"{r['raw_spearman_r']:>+9.3f}"
            f"{r['between_cluster_spearman_r']:>+10.3f}"
            f"{r['between_cluster_spearman_p']:>9.3f}"
            f"{r['weighted_slope']:>+12.5f}"
            f"{r['weighted_r2']:>8.2f}"
            f"{r['eta2_delta']:>8.2f}"
        )
    print(f"{'=' * 124}")
    print(f"  Sensitivity outputs in {args.output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo_dir", type=str, default="./results/oct_loo_full")
    ap.add_argument("--output_dir", type=str, default="./results/oct_cluster_trend_v2")
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--k_sensitivity", type=int, nargs="*", default=None,
                    help="Optional: run multiple k values, e.g. --k_sensitivity 4 5 6 7 8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.k_sensitivity:
        run_sensitivity(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
