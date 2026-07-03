#!/usr/bin/env python
"""
debug_sgd_feature_case_study.py  (Experiment 2)

Explain what each TracIn band is, in prediction-vector space, organized BY CLUSTER.

Per class:
  A. cluster-level summary  -> feature_cluster_summary.csv
       class, cluster, n, member_ratio, mean_tracin, mean_delta_loss,
       mean_self_influence, mean_p_true, mean_max_prob, mean_entropy,
       mean_margin, correct_rate
  B. case study samples (per cluster: max/min/median delta_loss, max/min self_influence)
       -> case_study_samples.csv
  C. same-score-different-delta pairs (same cluster, same membership, |dscore| tiny,
       |ddelta| large) with both prediction vectors -> same_score_different_delta_pairs.csv
  D. boxplots: entropy / p_true / margin / self_influence / delta_loss by cluster
       -> class{c}_feature_boxplots.png

Joins scores_class{c}.npz (scores, self_influence, train_x, train_y, test_y) with
loo_comprehensive_class{c}.json (the 50 LOO points) via train_idx.

  python debug_sgd_feature_case_study.py \
    --scores_dir results/oct_mia_tracin_sgd/tracin \
    --loo_dir results/oct_loo_full_sgd \
    --output_dir results/oct_debug_sgd/features \
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
from sklearn.cluster import KMeans

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _get(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def pv_stats(x_row):
    eps = 1e-12
    p = np.clip(x_row, eps, 1.0)
    srt = np.sort(x_row)[::-1]
    return dict(max_prob=float(x_row.max()), entropy=float(-(p * np.log(p)).sum()),
                margin=float(srt[0] - srt[1]), pred_class=int(x_row.argmax()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--loo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--pair_score_tol", type=float, default=0.5, help="max |dscore| to call 'same score'")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cluster_rows, case_rows, pair_rows = [], [], []
    for c in args.classes:
        sp = os.path.join(args.scores_dir, f"scores_class{c}.npz")
        lp = os.path.join(args.loo_dir, f"loo_comprehensive_class{c}.json")
        if not (os.path.exists(sp) and os.path.exists(lp)):
            print(f"skip class {c}: missing {sp} or {lp}"); continue
        npz = np.load(sp, allow_pickle=True)
        train_x = np.asarray(_get(npz, "train_x", "X_train", "attack_train_x"), float)
        train_y = np.asarray(_get(npz, "train_y", "y_train", "attack_train_y"), int)
        self_inf_all = _get(npz, "self_influence", "self_influence_raw")
        self_inf_all = np.asarray(self_inf_all, float) if self_inf_all is not None else None
        name = CLASS_NAMES[c] if c < 4 else str(c)

        d = json.load(open(lp))
        pts = d["single_point_loo"]
        idx = np.array([p["train_idx"] for p in pts], int)
        tracin = np.array([p["tracin_score"] for p in pts], float)
        dloss = np.array([p["delta_loss_mean"] for p in pts], float)
        member = np.array([p["membership"] for p in pts], int)

        # features for the 50 LOO points (join by train_idx)
        feats = train_x[idx]                       # (50, 4)
        p_true = feats[:, c]
        st = [pv_stats(feats[i]) for i in range(len(idx))]
        max_prob = np.array([s["max_prob"] for s in st])
        entropy = np.array([s["entropy"] for s in st])
        margin = np.array([s["margin"] for s in st])
        pred_class = np.array([s["pred_class"] for s in st])
        correct = (pred_class == c).astype(int)
        self_inf = self_inf_all[idx] if self_inf_all is not None else np.full(len(idx), np.nan)

        labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0).fit_predict(tracin.reshape(-1, 1))
        order = np.argsort([tracin[labels == k].mean() if (labels == k).any() else np.inf
                            for k in range(args.n_clusters)])

        print(f"\n=== Class {c} ({name}) cluster feature summary ===")
        box = {"entropy": [], "p_true": [], "margin": [], "self_influence": [], "delta_loss": [], "label": []}
        for k in order:
            m = labels == k
            if not m.any():
                continue
            row = dict(cls=c, cluster=int(k), n=int(m.sum()),
                       member_ratio=float(member[m].mean()),
                       mean_tracin=float(tracin[m].mean()), mean_delta_loss=float(dloss[m].mean()),
                       mean_self_influence=float(np.nanmean(self_inf[m])),
                       mean_p_true=float(p_true[m].mean()), mean_max_prob=float(max_prob[m].mean()),
                       mean_entropy=float(entropy[m].mean()), mean_margin=float(margin[m].mean()),
                       correct_rate=float(correct[m].mean()))
            cluster_rows.append(row)
            print(f"  clu{k}: n={row['n']:2d} memb={row['member_ratio']:.0%} "
                  f"p_true={row['mean_p_true']:.3f} entropy={row['mean_entropy']:.3f} "
                  f"margin={row['mean_margin']:.3f} selfI={row['mean_self_influence']:.1f} "
                  f"dloss={row['mean_delta_loss']:.4f}")
            box["entropy"].append(entropy[m]); box["p_true"].append(p_true[m])
            box["margin"].append(margin[m]); box["self_influence"].append(self_inf[m])
            box["delta_loss"].append(dloss[m]); box["label"].append(f"c{k}\n{member[m].mean():.0%}m")

            # B. case study: 5 picks within the cluster
            sub = np.where(m)[0]
            picks = {
                "max_dloss": sub[np.argmax(dloss[sub])],
                "min_dloss": sub[np.argmin(dloss[sub])],
                "median_dloss": sub[np.argsort(dloss[sub])[len(sub) // 2]],
                "max_selfI": sub[np.nanargmax(self_inf[sub])] if not np.all(np.isnan(self_inf[sub])) else sub[0],
                "min_selfI": sub[np.nanargmin(self_inf[sub])] if not np.all(np.isnan(self_inf[sub])) else sub[0],
            }
            for tag, j in picks.items():
                case_rows.append(dict(cls=c, cluster=int(k), kind=tag, train_idx=int(idx[j]),
                                      membership=int(member[j]), tracin_score=float(tracin[j]),
                                      delta_loss=float(dloss[j]), self_influence=float(self_inf[j]),
                                      p_true=float(p_true[j]), max_prob=float(max_prob[j]),
                                      entropy=float(entropy[j]), margin=float(margin[j]),
                                      prediction_vector=";".join(f"{v:.4f}" for v in feats[j])))

            # C. same-score-different-delta pairs within this cluster, same membership
            for a in range(len(sub)):
                for b in range(a + 1, len(sub)):
                    ia, ib = sub[a], sub[b]
                    if member[ia] != member[ib]:
                        continue
                    if abs(tracin[ia] - tracin[ib]) <= args.pair_score_tol:
                        pair_rows.append(dict(
                            cls=c, cluster=int(k), membership=int(member[ia]),
                            train_idx_A=int(idx[ia]), train_idx_B=int(idx[ib]),
                            score_A=float(tracin[ia]), score_B=float(tracin[ib]),
                            dscore=float(abs(tracin[ia] - tracin[ib])),
                            dloss_A=float(dloss[ia]), dloss_B=float(dloss[ib]),
                            ddelta=float(abs(dloss[ia] - dloss[ib])),
                            pv_A=";".join(f"{v:.4f}" for v in feats[ia]),
                            pv_B=";".join(f"{v:.4f}" for v in feats[ib])))

        # D. boxplots
        fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
        for a, key in zip(ax, ["entropy", "p_true", "margin", "self_influence", "delta_loss"]):
            a.boxplot(box[key], labels=box["label"], showmeans=True)
            a.set_title(f"{key} by cluster"); a.tick_params(axis="x", labelsize=8)
        fig.suptitle(f"Class {c} ({name}) feature distributions by TracIn band", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(args.output_dir, f"class{c}_feature_boxplots.png"), dpi=120)
        plt.close(fig)

    def dump(rows, fn):
        if not rows:
            print(f"(no rows for {fn})"); return
        cols = list(rows[0].keys())
        with open(os.path.join(args.output_dir, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"Saved {fn} ({len(rows)} rows)")

    dump(cluster_rows, "feature_cluster_summary.csv")
    dump(case_rows, "case_study_samples.csv")
    # sort pairs by largest delta-loss gap so the most striking collapses are on top
    pair_rows.sort(key=lambda r: -r["ddelta"])
    dump(pair_rows, "same_score_different_delta_pairs.csv")
    if pair_rows:
        print("\nTop same-score / different-delta pairs (TracIn collapse hides prediction-vector heterogeneity):")
        for r in pair_rows[:5]:
            print(f"  class{r['cls']} clu{r['cluster']} memb={r['membership']} "
                  f"score {r['score_A']:.2f}~{r['score_B']:.2f} (d={r['dscore']:.3f}) "
                  f"dloss {r['dloss_A']:.4f} vs {r['dloss_B']:.4f} (gap={r['ddelta']:.4f})  "
                  f"pv_A=[{r['pv_A']}] pv_B=[{r['pv_B']}]")


if __name__ == "__main__":
    main()
