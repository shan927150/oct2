#!/usr/bin/env python
"""
ablation_loo_nseeds_sweep.py  (Phase D1, merged)

Answers the reviewer's Re.1: "can averaging each LOO effect over MORE retraining runs
push single-point LOO above the noise floor, or does the signal vanish (redundancy)?"

For a set of LOO points, retrain the attack model up to --max_seeds times with that
single point removed (each paired with a same-seed baseline), record member-test loss,
then for each seed budget m in --grid recompute per-point mean/std of delta_loss using
only the first m seeds and report:

  raw_SNR       = between_point_var / (noise_floor_var)
                  NOTE: noise_floor_var ~ 1/m, so raw_SNR = 1 + m * (true_signal/noise).
                  It climbs with m whenever there is ANY signal and asymptotes to 1 when
                  there is none. Do NOT read raw_SNR alone.
  debiased_var  = between_point_var - noise_floor_var
                  Unbiased estimate of the TRUE point-to-point spread of the means. Does
                  not grow mechanically with m. ~0 (or <0) => no point-level signal.
  spearman      = Spearman(tracin, delta_loss_mean) with a point-bootstrap 95% CI.

Reading: debiased_var clearly > 0 AND the Spearman CI settles off 0 => averaging
recovers a real point-level signal. debiased_var ~ 0 AND the CI keeps straddling 0 at
max seeds => the effect is redundancy-limited, not just noisy; more seeds will not help.

Data source: the per-class scores_class{c}.npz (self-contained: scores, train_x/train_y/
test_x/test_y), so the score axis and the removal axis are identical by construction.

Point selection:
  default            : stratify by score from the npz (same algorithm as the LOO script,
                       so it picks the same points), then take --n_points.
  --loo_dir <dir>    : reuse the exact points from loo_comprehensive_class{c}.json. This
                       path is GUARDED: each json point's stored tracin_score must match
                       the score recomputed from the npz within --align_tol, otherwise the
                       json was built on a different train axis (the data-binding bug) and
                       the script aborts instead of silently correlating mismatched points.

  python ablation_loo_nseeds_sweep.py \
    --scores_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/nseeds_sweep \
    --classes 2 --n_points 20 --max_seeds 40
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

from models import build_attack_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _get(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def member_loss_after_training(X, y, keep_mask, mtx, mty, lr, mom, wd, epochs, batch, seed):
    g = torch.Generator(); g.manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_attack_model("nn", X.shape[1], 64).to(DEVICE)
    Xk, yk = X[keep_mask], y[keep_mask]
    loader = DataLoader(TensorDataset(torch.tensor(Xk, dtype=torch.float32),
                                      torch.tensor(yk, dtype=torch.long)),
                        batch_size=min(batch, len(Xk)), shuffle=True, generator=g)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(mtx, dtype=torch.float32, device=DEVICE))
        return float(nn.functional.cross_entropy(
            out, torch.tensor(mty, dtype=torch.long, device=DEVICE)))


def stratify_by_score(cand_idx, cand_scores, n):
    """Pick n indices from cand_idx spread across their score range (extremes + middle)."""
    cand_idx = np.asarray(cand_idx)
    if len(cand_idx) <= n:
        return cand_idx
    order = np.argsort(cand_scores)[::-1]
    cand_idx = cand_idx[order]
    n_extreme = min(max(1, n // 5), len(cand_idx) // 2)
    n_mid = n - 2 * n_extreme
    top, bot = cand_idx[:n_extreme], cand_idx[-n_extreme:]
    mid_range = cand_idx[n_extreme:-n_extreme]
    if n_mid <= 0 or len(mid_range) == 0:
        mid = np.array([], dtype=int)
    else:
        step = len(mid_range) / n_mid
        mid = mid_range[[int(i * step) for i in range(n_mid)]]
    return np.unique(np.concatenate([top, mid, bot]))


def bootstrap_spearman_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        rs[b] = np.nan if (np.std(x[i]) == 0 or np.std(y[i]) == 0) else spearmanr(x[i], y[i])[0]
    rs = rs[~np.isnan(rs)]
    if len(rs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--loo_dir", default=None,
                    help="optional: reuse exact points from loo_comprehensive_class{c}.json (guarded)")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_points", type=int, default=20)
    ap.add_argument("--max_seeds", type=int, default=40)
    ap.add_argument("--grid", type=int, nargs="+", default=[5, 10, 20, 30, 40])
    ap.add_argument("--n_test_points", type=int, default=50)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--align_tol", type=float, default=1e-3, help="tol for json-vs-npz score alignment check")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    grid = sorted(n for n in args.grid if n <= args.max_seeds)

    rows = []
    for c in args.classes:
        sp_path = os.path.join(args.scores_dir, f"scores_class{c}.npz")
        if not os.path.exists(sp_path):
            print(f"skip class {c}: {sp_path} missing"); continue
        npz = np.load(sp_path, allow_pickle=True)
        X = np.asarray(_get(npz, "train_x", "X_train", "attack_train_x"), np.float32)
        y = np.asarray(_get(npz, "train_y", "y_train", "attack_train_y"), int)
        test_x = np.asarray(_get(npz, "test_x", "X_test", "attack_test_x"), np.float32)
        test_y = np.asarray(_get(npz, "test_y", "y_test", "attack_test_y"), int)
        scores = np.asarray(_get(npz, "scores", "scores_raw"), float)
        name = CLASS_NAMES[c] if c < 4 else str(c)
        assert scores.shape == (len(test_x), len(X)), \
            f"class {c}: scores {scores.shape} != {(len(test_x), len(X))}"

        mt = np.where(test_y == 1)[0][:args.n_test_points]
        mtx, mty = test_x[mt], test_y[mt]
        mean_scores = scores[mt].mean(axis=0)

        # ---- choose points ----
        if args.loo_dir:
            lp = os.path.join(args.loo_dir, f"loo_comprehensive_class{c}.json")
            d = json.load(open(lp))
            jpts = d["single_point_loo"]
            jidx = np.array([p["train_idx"] for p in jpts], int)
            jscore = np.array([p["tracin_score"] for p in jpts], float)
            # GUARD: json points must live on the same train axis as the npz scores
            max_err = float(np.max(np.abs(mean_scores[jidx] - jscore)))
            if max_err > args.align_tol:
                raise SystemExit(
                    f"class {c}: LOO json points do not align with scores_class{c}.npz "
                    f"(max score error {max_err:.3g} > {args.align_tol}). The json was built on a "
                    f"different train axis (the attack_data/scores binding bug). Regenerate the LOO "
                    f"json with the fixed run_loo_comprehensive_sgd.py, or drop --loo_dir.")
            pts = stratify_by_score(jidx, jscore, args.n_points)
        else:
            pts = stratify_by_score(np.arange(len(X)), mean_scores, args.n_points)
        tracin = mean_scores[pts]
        n_train = len(X)
        t0 = time.time()
        print(f"\n=== Class {c} ({name}): N={n_train}, {len(pts)} points x {args.max_seeds} seeds "
              f"(+{args.max_seeds} baseline), source={'loo_json' if args.loo_dir else 'stratified'}, "
              f"device={DEVICE} ===")

        base = np.array([member_loss_after_training(X, y, np.ones(n_train, bool), mtx, mty,
                                                    args.lr, args.momentum, args.weight_decay,
                                                    args.epochs, args.batch_size, args.base_seed + s)
                         for s in range(args.max_seeds)])
        loo = np.empty((len(pts), args.max_seeds), float)
        for pi, ti in enumerate(pts):
            keep = np.ones(n_train, bool); keep[ti] = False
            for si in range(args.max_seeds):
                loo[pi, si] = member_loss_after_training(
                    X, y, keep, mtx, mty, args.lr, args.momentum, args.weight_decay,
                    args.epochs, args.batch_size, args.base_seed + si)
            if (pi + 1) % 5 == 0:
                print(f"  point {pi+1}/{len(pts)} done ({time.time()-t0:.0f}s)")

        print(f"  {'m':>4} {'debiased_var':>13} {'raw_SNR':>9} {'spearman':>9} {'ci_lo':>7} {'ci_hi':>7}")
        for m in grid:
            base_m = base[:m].mean()
            dl_mean = loo[:, :m].mean(axis=1) - base_m
            dl_std = loo[:, :m].std(axis=1, ddof=1) if m > 1 else np.zeros(len(pts))
            between = float(np.var(dl_mean, ddof=1))
            noise = float(np.mean(dl_std ** 2) / m)
            raw_snr = between / noise if noise > 0 else float("inf")
            debiased = between - noise
            if np.std(dl_mean) > 0 and np.std(tracin) > 0:
                sr = float(spearmanr(tracin, dl_mean)[0])
                lo, hi = bootstrap_spearman_ci(tracin, dl_mean, args.n_boot, seed=c)
            else:
                sr, lo, hi = float("nan"), float("nan"), float("nan")
            print(f"  {m:>4} {debiased:>13.2e} {raw_snr:>9.2f} {sr:>9.3f} {lo:>7.3f} {hi:>7.3f}")
            rows.append(dict(cls=c, class_name=name, n_points=len(pts), m_seeds=m,
                             between_point_var=between, noise_floor_var=noise,
                             debiased_between_var=debiased, raw_snr=raw_snr,
                             spearman=sr, spearman_ci_lo=lo, spearman_ci_hi=hi))

        # figure: (1) debiased var + raw SNR vs m ; (2) Spearman vs m with CI
        cr = [r for r in rows if r["cls"] == c]
        ms = [r["m_seeds"] for r in cr]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
        fig.suptitle(f"Class {c} ({name}): does averaging over more retrains rescue single-point LOO?",
                     fontsize=12, fontweight="bold")
        a1.plot(ms, [r["raw_snr"] for r in cr], "o-", color="tab:gray", label="raw SNR (climbs by construction)")
        a1.axhline(1, color="gray", ls="--", lw=1)
        a1b = a1.twinx()
        a1b.plot(ms, [r["debiased_between_var"] for r in cr], "s-", color="tab:red", label="debiased between-var")
        a1b.axhline(0, color="tab:red", ls=":", lw=1)
        a1.set_xlabel("n_seeds averaged (m)"); a1.set_ylabel("raw SNR")
        a1b.set_ylabel("debiased between-point var", color="tab:red"); a1.set_title("noise-floor view")
        h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        a2.plot(ms, [r["spearman"] for r in cr], "o-", color="tab:blue")
        a2.fill_between(ms, [r["spearman_ci_lo"] for r in cr], [r["spearman_ci_hi"] for r in cr],
                        color="tab:blue", alpha=0.2, label="95% bootstrap CI")
        a2.axhline(0, color="gray", ls="--", lw=1)
        a2.set_xlabel("n_seeds averaged (m)"); a2.set_ylabel("Spearman(TracIn, delta_loss_mean)")
        a2.set_title("point-wise correlation view"); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, f"nseeds_sweep_class{c}.png"), dpi=120)
        plt.close(fig)

        # per-seed raw dump so this never has to be rerun for a different grid
        with open(os.path.join(args.output_dir, f"nseeds_raw_class{c}.json"), "w") as f:
            json.dump({"class": c, "class_name": name,
                       "point_indices": [int(i) for i in pts],
                       "tracin_score": [float(v) for v in tracin],
                       "base_seed": args.base_seed, "max_seeds": args.max_seeds,
                       "baseline_losses_per_seed": [float(v) for v in base],
                       "loo_losses_per_seed": loo.tolist()}, f, indent=2)
        print(f"  class {c} done in {time.time()-t0:.0f}s")

    if rows:
        with open(os.path.join(args.output_dir, "nseeds_sweep_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\nSaved nseeds_sweep_summary.csv + per-class figures + per-seed raw json.")
    print("Read: debiased_between_var ~0 and Spearman CI straddling 0 at max seeds => averaging "
          "does NOT rescue single-point LOO (redundancy-limited). debiased_var clearly >0 and "
          "Spearman converging off 0 => it does.")


if __name__ == "__main__":
    main()
