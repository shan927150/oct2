#!/usr/bin/env python3
"""
Comprehensive LOO / Leave-K-Out validation for SGD-trained attack models.

Important:
  The ground-truth retraining in this script also uses SGD. This keeps the
  validation dynamics matched to the SGD TracIn scores.

  [FIX] Per-class train/test data are now loaded from the SAME scores_class{c}.npz
  that the TracIn scores come from (it already stores train_x/train_y/test_x/test_y).
  This guarantees the score axis and the LOO-removal axis are identical, removing the
  attack_data / scores mismatch risk. --attack_data_path is kept for CLI compat but
  is no longer used.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_attack_sgd(model, train_x, train_y, *, epochs, lr, batch_size, weight_decay=0.0,
                     momentum=0.0, seed=42):
    set_seed(seed)
    model = model.to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)),
        batch_size=min(batch_size, len(train_x)), shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def compute_test_loss(model, test_x, test_y):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        x_t = torch.tensor(test_x, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(test_y, dtype=torch.long, device=DEVICE)
        return criterion(model(x_t), y_t).cpu().numpy()


def train_and_eval_sgd(train_x, train_y, test_x, test_y, n_in, cfg, args, seed):
    from models import build_attack_model
    model = build_attack_model("nn", n_in, cfg.attack_n_hidden)
    model = train_attack_sgd(
        model, train_x, train_y,
        epochs=args.epochs,
        lr=args.sgd_lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        momentum=args.sgd_momentum,
        seed=seed,
    )
    return float(compute_test_loss(model, test_x, test_y).mean())


def select_stratified_points(mean_scores, n_points=50):
    """Select points across the score range: extremes + middle."""
    ranked = np.argsort(mean_scores)[::-1]
    n_extreme = min(10, n_points // 5)
    n_mid = n_points - 2 * n_extreme
    top_idx, bottom_idx = ranked[:n_extreme], ranked[-n_extreme:]
    mid_range = ranked[n_extreme:-n_extreme]
    if n_mid <= 0:
        mid_idx = np.array([], dtype=int)
    elif len(mid_range) > n_mid:
        step = len(mid_range) / n_mid
        mid_idx = mid_range[[int(i * step) for i in range(n_mid)]]
    else:
        mid_idx = mid_range
    return np.unique(np.concatenate([top_idx, mid_idx, bottom_idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str, default="./results/oct_mia_tracin/tracin/attack_data.npz",
                        help="[deprecated] no longer used; data is read from scores_class{c}.npz")
    parser.add_argument("--scores_dir", type=str, default="./results/oct_mia_tracin_sgd/tracin")
    parser.add_argument("--output_dir", type=str, default="./results/oct_loo_full_sgd")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--n_loo_points", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_test_points", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sgd_lr", type=float, default=0.03)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    from config import preset_oct
    cfg = preset_oct()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [FIX] data now loaded per-class from scores_class{c}.npz inside the loop (see below).
    seeds = [args.base_seed + s for s in range(args.n_seeds)]
    all_class_results = {}

    logger.info(f"SGD LOO setting: lr={args.sgd_lr}, momentum={args.sgd_momentum}, weight_decay={args.weight_decay}")

    for c in args.classes:
        t0 = time.time()
        logger.info(f"\n{'='*70}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*70}")

        # [FIX] load scores AND the exact train/test arrays the scores were computed on,
        # from the same npz. Guarantees the TracIn score axis == the LOO removal axis.
        score_path = Path(args.scores_dir) / f"scores_class{c}.npz"
        sp = np.load(score_path, allow_pickle=True)
        scores = sp["scores"]
        c_tr_x, c_tr_y = np.asarray(sp["train_x"], np.float32), np.asarray(sp["train_y"], int)
        c_te_x, c_te_y = np.asarray(sp["test_x"], np.float32), np.asarray(sp["test_y"], int)
        assert scores.shape == (len(c_te_x), len(c_tr_x)), (
            f"class {c}: scores shape {scores.shape} != {(len(c_te_x), len(c_tr_x))}")
        n_in, n_train = c_tr_x.shape[1], len(c_tr_x)

        member_test_idx = np.where(c_te_y == 1)[0][:args.n_test_points]
        test_x_eval, test_y_eval = c_te_x[member_test_idx], c_te_y[member_test_idx]

        mean_scores = scores[member_test_idx].mean(axis=0)

        selected = select_stratified_points(mean_scores, args.n_loo_points)
        logger.info(f"Single-point LOO: {len(selected)} points x {args.n_seeds} seeds")
        baseline_losses = [train_and_eval_sgd(c_tr_x, c_tr_y, test_x_eval, test_y_eval, n_in, cfg, args, s)
                           for s in seeds]
        baseline_mean = float(np.mean(baseline_losses))
        baseline_std = float(np.std(baseline_losses))
        logger.info(f"Baseline SGD loss: {baseline_mean:.6f} +/- {baseline_std:.6f}")

        loo_results = []
        for i, idx in enumerate(selected):
            keep = np.ones(n_train, dtype=bool)
            keep[idx] = False
            loo_losses = [train_and_eval_sgd(c_tr_x[keep], c_tr_y[keep], test_x_eval, test_y_eval, n_in, cfg, args, s)
                          for s in seeds]
            loo_results.append({
                "train_idx": int(idx),
                "tracin_score": float(mean_scores[idx]),
                "membership": int(c_tr_y[idx]),
                "delta_loss_mean": float(np.mean(loo_losses) - baseline_mean),
                "delta_loss_std": float(np.std(loo_losses)),
            })
            if (i + 1) % 10 == 0:
                logger.info(f"LOO {i+1}/{len(selected)}: delta={loo_results[-1]['delta_loss_mean']:+.6f}")

        tracin_arr = np.array([r["tracin_score"] for r in loo_results])
        delta_arr = np.array([r["delta_loss_mean"] for r in loo_results])
        sp_r, sp_p = spearmanr(tracin_arr, delta_arr)
        pe_r, pe_p = pearsonr(tracin_arr, delta_arr)
        logger.info(f"Spearman r={sp_r:.4f} p={sp_p:.4f}; Pearson r={pe_r:.4f} p={pe_p:.4f}")

        ranked_desc = np.argsort(mean_scores)[::-1]
        ranked_asc = np.argsort(mean_scores)
        lko_results = []
        for k in [10, 20, 50]:
            if k > n_train // 2:
                continue
            for direction, name, indices in [
                ("proponent", f"top-{k} proponents", ranked_desc[:k]),
                ("opponent", f"bottom-{k} opponents", ranked_asc[:k]),
            ]:
                keep = np.ones(n_train, dtype=bool)
                keep[indices] = False
                lko_losses = [train_and_eval_sgd(c_tr_x[keep], c_tr_y[keep], test_x_eval, test_y_eval, n_in, cfg, args, s)
                              for s in seeds]
                delta = float(np.mean(lko_losses) - baseline_mean)
                lko_results.append({
                    "condition": name,
                    "direction": direction,
                    "k": int(k),
                    "delta_loss_mean": delta,
                    "delta_loss_std": float(np.std(lko_losses)),
                    "removed_member_ratio": float(c_tr_y[indices].mean()),
                })
                logger.info(f"{name}: delta={delta:+.6f}")

        elapsed = float(time.time() - t0)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f"SGD TracIn LOO Validation — Class {c} ({CLASS_NAMES.get(c, c)}) | n_train={n_train}\n"
            f"Spearman r={sp_r:.3f} (p={sp_p:.3f}), Pearson r={pe_r:.3f} (p={pe_p:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        from matplotlib.lines import Line2D
        ax = axes[0]
        colors = ["#e74c3c" if r["membership"] == 1 else "#3498db" for r in loo_results]
        ax.scatter(tracin_arr, delta_arr, c=colors, alpha=0.6, s=35, edgecolors="white", lw=0.5)
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.axvline(0, color="gray", ls="--", alpha=0.4)
        if len(tracin_arr) > 2 and np.std(tracin_arr) > 0:
            z = np.polyfit(tracin_arr, delta_arr, 1)
            xl = np.linspace(tracin_arr.min(), tracin_arr.max(), 100)
            ax.plot(xl, np.polyval(z, xl), "k--", alpha=0.5, lw=1)
        ax.set_xlabel("SGD raw TracIn score")
        ax.set_ylabel("delta loss = LOO loss - baseline loss")
        ax.set_title("Single-point LOO")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Member'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Non-member'),
        ], fontsize=8)

        ax = axes[1]
        pro = [r for r in lko_results if r["direction"] == "proponent"]
        opp = [r for r in lko_results if r["direction"] == "opponent"]
        x_pos = np.arange(len(pro))
        ax.bar(x_pos - 0.175, [r["delta_loss_mean"] for r in pro], 0.35, label="Remove proponents", edgecolor="white")
        ax.bar(x_pos + 0.175, [r["delta_loss_mean"] for r in opp], 0.35, label="Remove opponents", edgecolor="white")
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"k={r['k']}" for r in pro])
        ax.set_xlabel("Samples removed")
        ax.set_ylabel("delta loss")
        ax.set_title("Leave-K-Out")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.hist(mean_scores, bins=50, alpha=0.4, density=True, label="All training")
        ax.hist(mean_scores[selected], bins=30, alpha=0.6, density=True, label="Selected LOO")
        ax.set_xlabel("Mean SGD raw TracIn score")
        ax.set_ylabel("Density")
        ax.set_title("Selected LOO points")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / f"loo_comprehensive_class{c}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        out = {
            "class": int(c),
            "class_name": CLASS_NAMES.get(c, str(c)),
            "n_train": int(n_train),
            "n_test_eval": int(len(member_test_idx)),
            "n_loo_points": int(len(selected)),
            "n_seeds": int(args.n_seeds),
            "optimizer": "sgd",
            "sgd_lr": float(args.sgd_lr),
            "sgd_momentum": float(args.sgd_momentum),
            "weight_decay": float(args.weight_decay),
            "baseline_loss_mean": baseline_mean,
            "baseline_loss_std": baseline_std,
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "elapsed_sec": elapsed,
            "single_point_loo": loo_results,
            "leave_k_out": lko_results,
        }
        with open(out_dir / f"loo_comprehensive_class{c}.json", "w") as f:
            json.dump(out, f, indent=2)
        all_class_results[str(c)] = {
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
        }
        logger.info(f"Class {c} done in {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("  SGD COMPREHENSIVE LOO — ALL CLASSES")
    print("=" * 80)
    print(f"  {'Class':<10} {'Spearman r':>12} {'p':>8} {'Pearson r':>12} {'p':>8}")
    for c in args.classes:
        r = all_class_results[str(c)]
        print(f"  {CLASS_NAMES.get(c, str(c)):<10} {r['spearman_r']:>+12.4f} {r['spearman_p']:>8.4f} "
              f"{r['pearson_r']:>+12.4f} {r['pearson_p']:>8.4f}")
    with open(out_dir / "loo_summary.json", "w") as f:
        json.dump(all_class_results, f, indent=2)


if __name__ == "__main__":
    main()
