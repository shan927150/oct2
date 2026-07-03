#!/usr/bin/env python
"""
ablation_band_level_loo.py  (Ablation 2 - the positive experiment, needs retraining)

Single-point LOO is too fine for this score. This evaluates attribution at the level
the score actually resolves: remove a whole SEMANTIC band from the full per-class
training set, retrain the attack model, and measure the change in member-test loss.

Semantic bands (defined on the full train set, not just the 50 LOO points):
  member_highconf      (y=1, entropy below class median)   -> expected strong proponents (loss up when removed)
  member_ambiguous     (y=1, entropy above class median)   -> the high self-influence band
  nonmember_highconf   (y=0, entropy below class median)   -> expected opponents (loss down when removed)
  nonmember_ambiguous  (y=0, entropy above class median)
The same number k of points is removed from each band (k = min band size, capped),
so the comparison is size-controlled. n_seeds random draws; report mean +/- std delta.

  python ablation_band_level_loo.py \
    --scores_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/band_loo \
    --classes 0 1 2 3 --n_seeds 5 --n_test_points 50
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import build_attack_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _get(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def entropy(X):
    eps = 1e-12
    p = np.clip(X, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def train_and_member_loss(X, y, keep_mask, member_test_x, member_test_y,
                          lr, mom, wd, epochs, batch, seed):
    g = torch.Generator(); g.manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
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
        out = model(torch.tensor(member_test_x, dtype=torch.float32, device=DEVICE))
        loss = nn.functional.cross_entropy(
            out, torch.tensor(member_test_y, dtype=torch.long, device=DEVICE)).item()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_test_points", type=int, default=50)
    ap.add_argument("--k_cap", type=int, default=300, help="cap on band-removal size")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for c in args.classes:
        sp = os.path.join(args.scores_dir, f"scores_class{c}.npz")
        if not os.path.exists(sp):
            print(f"skip class {c}: {sp} missing"); continue
        npz = np.load(sp, allow_pickle=True)
        X = np.asarray(_get(npz, "train_x", "X_train", "attack_train_x"), np.float32)
        y = np.asarray(_get(npz, "train_y", "y_train", "attack_train_y"), int)
        test_x = np.asarray(_get(npz, "test_x", "X_test", "attack_test_x"), np.float32)
        test_y = np.asarray(_get(npz, "test_y", "y_test", "attack_test_y"), int)
        name = CLASS_NAMES[c] if c < 4 else str(c)

        mt = np.where(test_y == 1)[0][:args.n_test_points]
        m_test_x, m_test_y = test_x[mt], test_y[mt]
        ent = entropy(X)
        ent_med = np.median(ent)
        bands = {
            "member_highconf": np.where((y == 1) & (ent <= ent_med))[0],
            "member_ambiguous": np.where((y == 1) & (ent > ent_med))[0],
            "nonmember_highconf": np.where((y == 0) & (ent <= ent_med))[0],
            "nonmember_ambiguous": np.where((y == 0) & (ent > ent_med))[0],
        }
        k = min(args.k_cap, min(len(v) for v in bands.values()))
        print(f"\n=== Class {c} ({name}): N={len(X)}, remove k={k} per band, "
              f"{args.n_seeds} seeds, device={DEVICE} ===")

        # baseline (full train), n_seeds
        base = np.array([train_and_member_loss(X, y, np.ones(len(X), bool), m_test_x, m_test_y,
                                                args.lr, args.momentum, args.weight_decay,
                                                args.epochs, args.batch_size, s)
                         for s in range(args.n_seeds)])
        print(f"  baseline member-test loss = {base.mean():.4f} +/- {base.std():.4f}")

        for band, pool in bands.items():
            deltas = []
            for s in range(args.n_seeds):
                rng = np.random.default_rng(1000 + s)
                rm = rng.choice(pool, size=k, replace=False)
                keep = np.ones(len(X), bool); keep[rm] = False
                loss = train_and_member_loss(X, y, keep, m_test_x, m_test_y,
                                             args.lr, args.momentum, args.weight_decay,
                                             args.epochs, args.batch_size, s)
                deltas.append(loss - base[s])
            deltas = np.array(deltas)
            sign = "loss UP (proponent-like)" if deltas.mean() > 0 else "loss DOWN (opponent-like)"
            print(f"  remove {band:>20}: delta={deltas.mean():+.4f} +/- {deltas.std():.4f}  {sign}")
            rows.append(dict(cls=c, class_name=name, band=band, k=k,
                             baseline_loss=float(base.mean()),
                             delta_loss_mean=float(deltas.mean()), delta_loss_std=float(deltas.std())))

        # bar plot
        br = [r for r in rows if r["cls"] == c]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([r["band"] for r in br], [r["delta_loss_mean"] for r in br],
               yerr=[r["delta_loss_std"] for r in br], capsize=4,
               color=["tab:red" if "member" in r["band"] else "tab:blue" for r in br])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("member-test loss delta (removed - baseline)")
        ax.set_title(f"Class {c} ({name}): band-level LOO (k={br[0]['k']})")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, f"band_loo_class{c}.png"), dpi=120)
        plt.close(fig)

    with open(os.path.join(args.output_dir, "band_loo_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved band_loo_summary.csv + per-class bars")
    print("Read: a clean monotone pattern (remove member_highconf -> loss up; remove nonmember -> loss down) "
          "means attribution IS reliable at band/group level even though single-point LOO is not.")


if __name__ == "__main__":
    main()
