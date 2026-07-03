#!/usr/bin/env python3
"""
Mislabel detection with plain-SGD attack-model training.

Flip 10% of attack-training labels, train the per-class attack model with SGD,
compute TracIn self-influence on SGD checkpoints, and measure recovery of flipped labels.
"""
import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_attack_sgd_with_checkpoints(model, train_x, train_y, *, epochs, lr, batch_size,
                                      weight_decay=0.0, momentum=0.0, checkpoint_every=5,
                                      seed=42, label=""):
    set_seed(seed)
    model = model.to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)),
        batch_size=min(batch_size, len(train_x)), shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    checkpoints = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch % checkpoint_every == 0 or epoch == epochs:
            checkpoints.append({
                "epoch": int(epoch),
                "state_dict": copy.deepcopy(model.state_dict()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(total_loss),
                "optimizer": "sgd",
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
            })
    model.eval()
    logger.info(f"[{label}] trained {epochs} epochs with SGD, checkpoints={len(checkpoints)}")
    return model, checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str, default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--output_dir", type=str, default="./results/oct_mislabel_sgd")
    parser.add_argument("--flip_ratio", type=float, default=0.10)
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--sgd_lr", type=float, default=0.03)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from config import preset_oct
    from models import build_attack_model
    from attribution import tracin_self_influence

    cfg = preset_oct()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.attack_data_path)
    attack_train_x, attack_train_y, train_classes = ad["attack_train_x"], ad["attack_train_y"], ad["train_classes"]
    n_in = attack_train_x.shape[1]
    rng = np.random.default_rng(args.seed)
    all_results = {}

    for c in args.classes:
        logger.info(f"\n{'='*70}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*70}")
        mask = train_classes == c
        c_x = attack_train_x[mask]
        c_y_clean = attack_train_y[mask].copy()
        n = len(c_x)
        n_flip = int(n * args.flip_ratio)
        flip_indices = rng.choice(n, size=n_flip, replace=False)
        is_flipped = np.zeros(n, dtype=bool)
        is_flipped[flip_indices] = True
        c_y_noisy = c_y_clean.copy()
        c_y_noisy[flip_indices] = 1 - c_y_noisy[flip_indices]
        logger.info(f"Total={n}, flipped={n_flip} ({args.flip_ratio:.0%}), SGD lr={args.sgd_lr}")

        def make_model():
            return build_attack_model("nn", n_in, cfg.attack_n_hidden).to(DEVICE)

        _, checkpoints = train_attack_sgd_with_checkpoints(
            make_model(), c_x, c_y_noisy,
            epochs=args.epochs, lr=args.sgd_lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, momentum=args.sgd_momentum,
            checkpoint_every=args.checkpoint_every, seed=args.seed + 1000 + c,
            label=f"SGD-Mislabel-c{c}",
        )
        torch.save(checkpoints, out_dir / f"sgd_mislabel_checkpoints_class{c}.pt")

        self_inf = tracin_self_influence(checkpoints, c_x, c_y_noisy, make_model, equal_weight=True)
        rank_order = np.argsort(self_inf)[::-1]
        recovery_rate = np.cumsum(is_flipped[rank_order]) / max(n_flip, 1)
        recovery_at_10 = recovery_rate[max(1, int(n * 0.10)) - 1]
        recovery_at_20 = recovery_rate[max(1, int(n * 0.20)) - 1]
        si_flipped, si_clean = self_inf[is_flipped], self_inf[~is_flipped]
        logger.info(f"Recovery@10%={recovery_at_10:.4f}, @20%={recovery_at_20:.4f}, SI ratio={si_flipped.mean()/max(si_clean.mean(),1e-12):.2f}x")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"SGD Mislabel Detection — Class {c} ({CLASS_NAMES.get(c, c)})\n"
                     f"Flipped {n_flip}/{n} | Recovery@10%={recovery_at_10:.3f}",
                     fontsize=13, fontweight="bold")
        fractions = np.arange(1, n + 1) / n
        ax = axes[0]
        ax.plot(fractions * 100, recovery_rate * 100, lw=2, label="SGD TracIn Self-Influence")
        ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.5, label="Random")
        ax.axvline(10, ls=":", alpha=0.7, label=f"@10%: {recovery_at_10*100:.1f}%")
        ax.set_xlabel("% Training Data Inspected"); ax.set_ylabel("% Mislabeled Recovered")
        ax.set_title("Recovery Curve"); ax.legend(fontsize=9)
        ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

        ax = axes[1]
        upper = np.percentile(self_inf, 99)
        if upper <= 0:
            upper = self_inf.max() + 1e-8
        bins = np.linspace(0, upper, 40)
        ax.hist(si_clean, bins=bins, alpha=0.6, label=f"Clean (n={len(si_clean)})", density=True)
        ax.hist(si_flipped, bins=bins, alpha=0.6, label=f"Flipped (n={len(si_flipped)})", density=True)
        ax.set_xlabel("Self-Influence"); ax.set_ylabel("Density")
        ax.set_title("Self-Influence Distribution"); ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / f"mislabel_sgd_class{c}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        all_results[str(c)] = {
            "class": int(c),
            "class_name": CLASS_NAMES.get(c, str(c)),
            "optimizer": "sgd",
            "sgd_lr": float(args.sgd_lr),
            "sgd_momentum": float(args.sgd_momentum),
            "weight_decay": float(args.weight_decay),
            "n_total": int(n),
            "n_flipped": int(n_flip),
            "flip_ratio": float(args.flip_ratio),
            "recovery_at_10pct": float(recovery_at_10),
            "recovery_at_20pct": float(recovery_at_20),
            "self_inf_flipped_mean": float(si_flipped.mean()),
            "self_inf_clean_mean": float(si_clean.mean()),
            "self_inf_flip_clean_ratio": float(si_flipped.mean() / max(si_clean.mean(), 1e-12)),
        }

    with open(out_dir / "mislabel_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  {'Class':<10} {'N':>6} {'Flipped':>8} {'Recovery@10%':>14} {'Recovery@20%':>14} {'SI Flip/Clean':>14}")
    for c in map(str, args.classes):
        r = all_results[c]
        print(f"  {r['class_name']:<10} {r['n_total']:>6} {r['n_flipped']:>8} "
              f"{r['recovery_at_10pct']:>13.3f} {r['recovery_at_20pct']:>13.3f} {r['self_inf_flip_clean_ratio']:>13.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
