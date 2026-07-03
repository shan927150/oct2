#!/usr/bin/env python
"""
debug_sgd_gradient_checkpoint.py  (Experiment 3)

Decompose the cross-influence TracIn score into per-checkpoint gradient pieces, to
tell whether large scores come from gradient NORM or directional ALIGNMENT, and
whether a single checkpoint dominates.

For each selected LOO train sample, at each checkpoint i:
    train_grad_norm_i = ||grad_train||
    test_grad_norm_i  = mean over selected member-test points of ||grad_test||
    dot_i             = < mean_test grad_test , grad_train >     (= mean inner product)
    cos_i             = dot_i / (train_grad_norm_i * test_grad_norm_i + eps)
    self_i            = ||grad_train||^2
Aggregates per sample:
    score_sum, norm_mean, norm_max, cos_mean, dominant_checkpoint, checkpoint_concentration

Checkpoints: loads sgd_checkpoints_class{c}.pt if present; otherwise retrains with the
same plain-SGD recipe (deterministic with seed 42) to regenerate identical checkpoints.

  python debug_sgd_gradient_checkpoint.py \
    --scores_dir results/oct_mia_tracin_sgd/tracin \
    --loo_dir results/oct_loo_full_sgd \
    --checkpoint_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/gradients \
    --classes 0 1 2 --n_test_points 50
"""
import argparse
import copy
import csv
import json
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


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def retrain_checkpoints(X, y, lr, mom, wd, epochs, batch, every, seed):
    set_seed(seed)
    model = build_attack_model("nn", X.shape[1], 64).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                      torch.tensor(y, dtype=torch.long)),
                        batch_size=min(batch, len(X)), shuffle=True)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    ck = []
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if ep % every == 0 or ep == epochs:
            ck.append({"epoch": ep, "state_dict": copy.deepcopy(model.state_dict())})
    return ck


def load_checkpoints(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "checkpoints" in obj:
        obj = obj["checkpoints"]
    out = []
    for i, el in enumerate(obj):
        if isinstance(el, dict) and "state_dict" in el:
            out.append({"epoch": el.get("epoch", i + 1), "state_dict": el["state_dict"]})
        else:
            out.append({"epoch": i + 1, "state_dict": el})
    return out


def flat_grad(model, crit, x_row, y_row):
    model.zero_grad(set_to_none=True)
    crit(model(x_row), y_row).backward()
    return torch.cat([(p.grad.detach().flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=DEVICE)) for p in model.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--loo_dir", required=True)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_test_points", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--checkpoint_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    eps = 1e-12

    for c in args.classes:
        sp = os.path.join(args.scores_dir, f"scores_class{c}.npz")
        lp = os.path.join(args.loo_dir, f"loo_comprehensive_class{c}.json")
        if not (os.path.exists(sp) and os.path.exists(lp)):
            print(f"skip class {c}: missing {sp} or {lp}"); continue
        name = CLASS_NAMES[c] if c < 4 else str(c)
        npz = np.load(sp, allow_pickle=True)
        train_x = np.asarray(_get(npz, "train_x", "X_train", "attack_train_x"), np.float32)
        train_y = np.asarray(_get(npz, "train_y", "y_train", "attack_train_y"), int)
        test_x = np.asarray(_get(npz, "test_x", "X_test", "attack_test_x"), np.float32)
        test_y = np.asarray(_get(npz, "test_y", "y_test", "attack_test_y"), int)

        d = json.load(open(lp))
        pts = d["single_point_loo"]
        sel = np.array([p["train_idx"] for p in pts], int)
        json_score = np.array([p["tracin_score"] for p in pts], float)
        member = np.array([p["membership"] for p in pts], int)

        mt = np.where(test_y == 1)[0][:args.n_test_points]
        print(f"\n=== Class {c} ({name}): {len(sel)} LOO train pts, {len(mt)} member-test pts, device={DEVICE} ===")

        # checkpoints
        ckpt_path = os.path.join(args.checkpoint_dir or args.scores_dir, f"sgd_checkpoints_class{c}.pt")
        if os.path.exists(ckpt_path):
            ckpts = load_checkpoints(ckpt_path); print(f"loaded {len(ckpts)} checkpoints from {ckpt_path}")
        else:
            print(f"{ckpt_path} not found -> retraining (deterministic, seed {args.seed})")
            ckpts = retrain_checkpoints(train_x, train_y, args.lr, args.momentum, args.weight_decay,
                                        args.epochs, args.batch_size, args.checkpoint_every, args.seed)
            print(f"retrained {len(ckpts)} checkpoints")

        nC, nS = len(ckpts), len(sel)
        DOT = np.zeros((nC, nS)); TN = np.zeros((nC, nS)); COS = np.zeros((nC, nS))
        crit = nn.CrossEntropyLoss()
        tx = torch.tensor(train_x, device=DEVICE); ty = torch.tensor(train_y, device=DEVICE)
        ex = torch.tensor(test_x, device=DEVICE); ey = torch.tensor(test_y, device=DEVICE)

        for k, ck in enumerate(ckpts):
            model = build_attack_model("nn", train_x.shape[1], 64).to(DEVICE)
            model.load_state_dict(ck["state_dict"]); model.eval()
            # test grads
            gt = torch.stack([flat_grad(model, crit, ex[t:t + 1], ey[t:t + 1]) for t in mt])  # (T,P)
            gt_mean = gt.mean(0)
            test_norm_mean = float(gt.norm(dim=1).mean())
            for s, j in enumerate(sel):
                g = flat_grad(model, crit, tx[j:j + 1], ty[j:j + 1])
                tn = float(g.norm())
                dot = float(torch.dot(gt_mean, g))
                DOT[k, s] = dot; TN[k, s] = tn
                COS[k, s] = dot / (tn * test_norm_mean + eps)
            print(f"  ckpt {k+1}/{nC} (ep {ck['epoch']}): test_norm_mean={test_norm_mean:.4f}")

        score_sum = DOT.sum(0)
        norm_mean = TN.mean(0); norm_max = TN.max(0); cos_mean = COS.mean(0)
        absdot = np.abs(DOT)
        dom_ck = absdot.argmax(0)
        concentration = absdot.max(0) / (absdot.sum(0) + eps)

        # diagnostics: is score driven by norm or by direction?
        from scipy.stats import spearmanr
        r_norm = spearmanr(score_sum, norm_mean)[0]
        r_cos = spearmanr(score_sum, cos_mean)[0]
        r_json = spearmanr(score_sum, json_score)[0]
        print(f"  Spearman(score_sum, json tracin) = {r_json:+.3f}  (sanity, should be high)")
        print(f"  Spearman(score_sum, norm_mean)   = {r_norm:+.3f}")
        print(f"  Spearman(score_sum, cos_mean)    = {r_cos:+.3f}")
        print(f"  -> if |norm| corr >> |cos| corr: score is NORM-dominated (diagnosis D); "
              f"mean checkpoint_concentration={concentration.mean():.3f}")

        # per-sample summary CSV
        with open(os.path.join(args.output_dir, f"gradient_summary_class{c}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train_idx", "membership", "json_tracin", "score_sum", "norm_mean",
                        "norm_max", "cos_mean", "dominant_checkpoint_epoch", "checkpoint_concentration"])
            for s in range(nS):
                w.writerow([int(sel[s]), int(member[s]), f"{json_score[s]:.4f}", f"{score_sum[s]:.4f}",
                            f"{norm_mean[s]:.4f}", f"{norm_max[s]:.4f}", f"{cos_mean[s]:.4f}",
                            int(ckpts[dom_ck[s]]["epoch"]), f"{concentration[s]:.4f}"])

        # top by gradient norm
        topn = np.argsort(norm_max)[::-1][:10]
        with open(os.path.join(args.output_dir, f"top_gradient_samples_class{c}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train_idx", "membership", "norm_max", "score_sum", "cos_mean"])
            for s in topn:
                w.writerow([int(sel[s]), int(member[s]), f"{norm_max[s]:.4f}",
                            f"{score_sum[s]:.4f}", f"{cos_mean[s]:.4f}"])

        # case-study samples: 2 top proponents, 2 top opponents, 2 near-zero
        o = np.argsort(score_sum)
        case = list(o[-2:]) + list(o[:2]) + list(o[np.argsort(np.abs(score_sum[o]))[:2]])
        ep_axis = [ck["epoch"] for ck in ckpts]
        for arr, title, fn, ylab in [
            (DOT, "checkpoint dot contribution", "checkpoint_contrib", "dot contribution"),
            (TN, "train gradient norm", "grad_norm_trajectory", "||grad_train||"),
            (COS, "cosine(train, mean test)", "cosine_trajectory", "cosine")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for s in case:
                ax.plot(ep_axis, arr[:, s], "-o", ms=4,
                        label=f"idx{sel[s]} {'M' if member[s] else 'N'} score={score_sum[s]:+.2f}")
            ax.axhline(0, color="gray", lw=0.7, ls=":")
            ax.set_xlabel("checkpoint epoch"); ax.set_ylabel(ylab)
            ax.set_title(f"Class {c} ({name}): {title}"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, f"{fn}_class{c}.png"), dpi=120)
            plt.close(fig)
        print(f"  saved gradient_summary / top_gradient / 3 trajectory pngs for class {c}")


if __name__ == "__main__":
    main()
