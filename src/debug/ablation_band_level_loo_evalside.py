#!/usr/bin/env python
"""
ablation_band_level_loo_evalside.py  (Phase D2, merged)

Same band-level LOO as C3 (ablation_band_level_loo.py) but the evaluation test side is
selectable, to test the symmetry the reviewer asked about (Re.3 / Re.6):

  --eval_side member     reproduces the original C3 result (sanity):
                         remove member bands   -> member-test loss UP   (proponents)
                         remove nonmember bands -> member-test loss DOWN (opponents)
  --eval_side nonmember  the NEW symmetry test, expect the mirror:
                         remove nonmember bands -> nonmember-test loss UP
                         remove member bands    -> nonmember-test loss DOWN

Bands are the four C3 bands (membership x entropy median), k points removed per band
(size-controlled), n_seeds retrains. Retraining is plain SGD, matched to the SGD scores.
Band-removal RNG uses seed 1000+s, identical to C3, so the member run reproduces C3
point-for-point rather than merely being statistically equivalent.

Re.6 note: in C3, member_highconf ~= member_ambiguous, i.e. the entropy split separates
SELF-influence, not the cross-influence contribution to the test loss. So this script
also prints, for the MATCHED side, the highconf-minus-ambiguous gap. If that gap is ~0
on the matched side too, "low-entropy samples are the influential ones" is a
self-influence effect, not a cross-influence direction.

  # member side (sanity, reproduces C3)
  python ablation_band_level_loo_evalside.py --scores_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/band_loo_member --classes 0 1 2 3 --eval_side member
  # non-member side (symmetry test)
  python ablation_band_level_loo_evalside.py --scores_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/band_loo_nonmember --classes 0 1 2 3 --eval_side nonmember
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


def eval_loss_after_training(X, y, keep_mask, etx, ety, lr, mom, wd, epochs, batch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
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
        out = model(torch.tensor(etx, dtype=torch.float32, device=DEVICE))
        return float(nn.functional.cross_entropy(
            out, torch.tensor(ety, dtype=torch.long, device=DEVICE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--eval_side", choices=["member", "nonmember"], default="member")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_test_points", type=int, default=50)
    ap.add_argument("--k_cap", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    side_label = 1 if args.eval_side == "member" else 0

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

        et = np.where(test_y == side_label)[0][:args.n_test_points]
        if len(et) == 0:
            print(f"skip class {c}: no {args.eval_side}-test points"); continue
        etx, ety = test_x[et], test_y[et]
        ent = entropy(X); em = np.median(ent)
        bands = {
            "member_highconf": np.where((y == 1) & (ent <= em))[0],
            "member_ambiguous": np.where((y == 1) & (ent > em))[0],
            "nonmember_highconf": np.where((y == 0) & (ent <= em))[0],
            "nonmember_ambiguous": np.where((y == 0) & (ent > em))[0],
        }
        k = min(args.k_cap, min(len(v) for v in bands.values()))
        print(f"\n=== Class {c} ({name}) eval_side={args.eval_side}: remove k={k}, "
              f"{args.n_seeds} seeds, n_eval={len(et)}, device={DEVICE} ===")
        base = np.array([eval_loss_after_training(X, y, np.ones(len(X), bool), etx, ety,
                                                  args.lr, args.momentum, args.weight_decay,
                                                  args.epochs, args.batch_size, s)
                         for s in range(args.n_seeds)])
        print(f"  baseline {args.eval_side}-test loss = {base.mean():.4f} +/- {base.std():.4f}")
        for band, pool in bands.items():
            deltas = []
            for s in range(args.n_seeds):
                rng = np.random.default_rng(1000 + s)   # aligned to C3 (ablation_band_level_loo.py)
                rm = rng.choice(pool, size=k, replace=False)
                keep = np.ones(len(X), bool); keep[rm] = False
                deltas.append(eval_loss_after_training(X, y, keep, etx, ety, args.lr, args.momentum,
                                                       args.weight_decay, args.epochs, args.batch_size, s) - base[s])
            deltas = np.array(deltas)
            role = "proponent (loss UP)" if deltas.mean() > 0 else "opponent (loss DOWN)"
            print(f"  remove {band:>20}: delta={deltas.mean():+.4f} +/- {deltas.std():.4f}  {role}")
            rows.append(dict(cls=c, class_name=name, eval_side=args.eval_side, band=band, k=k,
                             baseline_loss=float(base.mean()),
                             delta_loss_mean=float(deltas.mean()), delta_loss_std=float(deltas.std())))

        # Re.6 check: highconf - ambiguous gap on the MATCHED side
        matched = "member" if args.eval_side == "member" else "nonmember"
        cr = [r for r in rows if r["cls"] == c]
        hc = next((r for r in cr if r["band"] == f"{matched}_highconf"), None)
        am = next((r for r in cr if r["band"] == f"{matched}_ambiguous"), None)
        if hc and am:
            gap = hc["delta_loss_mean"] - am["delta_loss_mean"]
            print(f"  [Re.6 check] matched-side ({matched}) highconf - ambiguous = {gap:+.4f}  "
                  f"(near 0 => entropy split is self-influence, not a cross-influence direction)")

        # per-class bar (correct coloring: member bands red, nonmember bands blue)
        br = [r for r in cr]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([r["band"] for r in br], [r["delta_loss_mean"] for r in br],
               yerr=[r["delta_loss_std"] for r in br], capsize=4,
               color=["tab:red" if r["band"].startswith("member") else "tab:blue" for r in br])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel(f"{args.eval_side}-test loss delta (removed - baseline)")
        ax.set_title(f"Class {c} ({name}) band-level LOO, eval_side={args.eval_side} (k={br[0]['k']})")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"band_loo_{args.eval_side}_class{c}.png"), dpi=120)
        plt.close(fig)

    if rows:
        with open(os.path.join(args.output_dir, f"band_loo_{args.eval_side}_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nSaved band_loo_{args.eval_side}_summary.csv + per-class bars")
    print(f"Read (eval_side={args.eval_side}): under nonmember evaluation the pattern should MIRROR "
          "the member run, with nonmember bands as proponents (loss up) and member bands as opponents. "
          "Watch [Re.6 check]: highconf ~= ambiguous on the matched side => 'low-entropy is more "
          "influential' is a self-influence effect, not a cross-influence one.")


if __name__ == "__main__":
    main()
