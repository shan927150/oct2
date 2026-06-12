#!/usr/bin/env python3
"""
实验: Mislabel Detection via TracIn Self-Influence。

复现 [Pruthi 2020 Sec 4.1] 评估法，应用到 attack model 训练数据:
  翻转 10% membership 标签 → 含噪重训 → 算 self-influence → 降序排 → 看 top-x% 找回多少翻转样本。
错误标注样本是自身的强 proponent (离群)，故 self-influence 高 → 排在前面。
flip_ratio=0.10 来源: [Pruthi Sec 4.1] "for 10% of the training data, we changed the label"。

用法:
  python run_mislabel_detection.py --attack_data_path .../attack_data.npz \
      --output_dir ./results/oct_mislabel --flip_ratio 0.10 --classes 0 1 2 3
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str,
                        default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--output_dir", type=str, default="./results/oct_mislabel")
    parser.add_argument("--flip_ratio", type=float, default=0.10)   # [Pruthi Sec 4.1]
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.attack_data_path)
    attack_train_x, attack_train_y, train_classes = ad["attack_train_x"], ad["attack_train_y"], ad["train_classes"]
    logger.info(f"Loaded attack data: train={attack_train_x.shape}")

    from models import build_attack_model
    from attribution import train_attack_with_checkpoints, tracin_self_influence
    from config import preset_oct
    cfg = preset_oct()
    n_in = attack_train_x.shape[1]
    rng = np.random.default_rng(args.seed)
    all_results = {}

    for c in args.classes:
        logger.info(f"\n{'='*60}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*60}")
        mask = train_classes == c
        c_x = attack_train_x[mask]
        c_y_clean = attack_train_y[mask].copy()
        n = len(c_x)

        # Step 1: 翻转 flip_ratio 比例的标签
        n_flip = int(n * args.flip_ratio)
        flip_indices = rng.choice(n, size=n_flip, replace=False)
        is_flipped = np.zeros(n, dtype=bool)
        is_flipped[flip_indices] = True
        c_y_noisy = c_y_clean.copy()
        c_y_noisy[flip_indices] = 1 - c_y_noisy[flip_indices]
        logger.info(f"  Total={n}, Flipped={n_flip} ({args.flip_ratio*100:.0f}%)")

        # Step 2: 含噪标签训练 attack model (带 checkpoint)
        def make_model():
            return build_attack_model("nn", n_in, cfg.attack_n_hidden)
        torch.manual_seed(args.seed + c)
        np.random.seed(args.seed + c)
        _, ckpts = train_attack_with_checkpoints(
            make_model(), c_x, c_y_noisy, epochs=cfg.attack_epochs, lr=cfg.attack_lr,
            batch_size=cfg.attack_batch_size, l2_ratio=cfg.attack_l2,
            checkpoint_every=args.checkpoint_every, label=f"Mislabel-c{c}")

        # Step 3: self-influence → 降序 → recovery curve
        self_inf = tracin_self_influence(ckpts, c_x, c_y_noisy, make_model)
        rank_order = np.argsort(self_inf)[::-1]
        recovery_rate = np.cumsum(is_flipped[rank_order]) / n_flip
        recovery_at_10 = recovery_rate[max(1, int(n * 0.10)) - 1]
        recovery_at_20 = recovery_rate[max(1, int(n * 0.20)) - 1]
        logger.info(f"  Recovery@10%={recovery_at_10:.4f} (random=0.10), @20%={recovery_at_20:.4f}")

        si_flipped, si_clean = self_inf[is_flipped], self_inf[~is_flipped]
        logger.info(f"  Self-inf flipped mean={si_flipped.mean():.4f}, clean mean={si_clean.mean():.4f}")

        # Step 4: 画 recovery curve + self-influence 分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Mislabel Detection — Class {c} ({CLASS_NAMES.get(c, c)})\n"
                     f"Flipped {n_flip}/{n} | Recovery@10%={recovery_at_10:.3f}",
                     fontsize=13, fontweight="bold")
        fractions = np.arange(1, n + 1) / n
        ax = axes[0]
        ax.plot(fractions * 100, recovery_rate * 100, color="#e74c3c", lw=2, label="TracIn Self-Influence")
        ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.5, label="Random")
        ax.axvline(10, color="#3498db", ls=":", alpha=0.7, label=f"@10%: {recovery_at_10*100:.1f}%")
        ax.set_xlabel("% Training Data Inspected"); ax.set_ylabel("% Mislabeled Recovered")
        ax.set_title("Recovery Curve"); ax.legend(fontsize=9)
        ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

        ax = axes[1]
        bins = np.linspace(0, np.percentile(self_inf, 99), 40)
        ax.hist(si_clean, bins=bins, alpha=0.6, color="#3498db", label=f"Clean (n={len(si_clean)})", density=True)
        ax.hist(si_flipped, bins=bins, alpha=0.6, color="#e74c3c", label=f"Flipped (n={len(si_flipped)})", density=True)
        ax.set_xlabel("Self-Influence"); ax.set_ylabel("Density")
        ax.set_title("Self-Influence Distribution"); ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(str(out_dir / f"mislabel_class{c}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        all_results[c] = {
            "class": int(c), "class_name": CLASS_NAMES.get(c, str(c)),
            "n_total": int(n), "n_flipped": int(n_flip), "flip_ratio": float(args.flip_ratio),
            "recovery_at_10pct": float(recovery_at_10), "recovery_at_20pct": float(recovery_at_20),
            "self_inf_flipped_mean": float(si_flipped.mean()), "self_inf_clean_mean": float(si_clean.mean()),
        }

    with open(out_dir / "mislabel_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  {'Class':<10} {'N':>6} {'Flipped':>8} {'Recovery@10%':>14} {'Recovery@20%':>14} {'SI Flip/Clean':>14}")
    for c in args.classes:
        r = all_results.get(c)
        if r is None:
            continue
        ratio = r["self_inf_flipped_mean"] / max(r["self_inf_clean_mean"], 1e-8)
        print(f"  {r['class_name']:<10} {r['n_total']:>6} {r['n_flipped']:>8} "
              f"{r['recovery_at_10pct']:>13.3f} {r['recovery_at_20pct']:>13.3f} {ratio:>13.1f}×")
    print("=" * 70)


if __name__ == "__main__":
    main()
