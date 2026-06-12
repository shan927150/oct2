#!/usr/bin/env python3
"""
实验 A: OCT Utility Baseline [ours]。
只训练分类模型，不跑 shadow/attack；用于确认 OCT 分类正常 + 观察过拟合程度
(train-test gap 是后续 MIA 的关键指标)。

输出: confusion matrix (PNG)、per-class P/R/F1、macro-F1、balanced acc、gap、预测分布 npz。

用法:
  python run_oct_baseline.py                     # 默认 oct preset
  python run_oct_baseline.py --preset oct_smoke
  python run_oct_baseline.py --target_epochs 100
"""
import argparse
import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OCT_CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]   # 与 data.py class_names 顺序一致


def parse_args():
    parser = argparse.ArgumentParser(description="OCT Utility Baseline (Experiment A)")
    parser.add_argument("--preset", type=str, default="oct", choices=["oct_smoke", "oct", "oct_large"])
    parser.add_argument("--target_model_type", type=str, default=None, choices=["cnn", "paper_cnn"])
    parser.add_argument("--target_epochs", type=int, default=None)
    parser.add_argument("--target_lr", type=float, default=None)
    parser.add_argument("--target_batch_size", type=int, default=None)
    parser.add_argument("--target_n_hidden", type=int, default=None)
    parser.add_argument("--n_total_samples", type=int, default=None)
    parser.add_argument("--target_data_size", type=int, default=None)
    parser.add_argument("--optimizer_type", type=str, default=None, choices=["sgd", "adam"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, output_path, train_acc, test_acc):
    """保存 confusion matrix 图 (左: 计数, 右: 行归一化)。"""
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"OCT Baseline — train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
                 f"gap={train_acc - test_acc:.4f}", fontsize=13, fontweight="bold")
    for ax, mat, title, fmt in [(axes[0], cm, "Confusion Matrix (counts)", "d"),
                                (axes[1], cm_norm, "Confusion Matrix (normalized)", ".3f")]:
        ax.imshow(mat, cmap="Blues", **({"vmin": 0, "vmax": 1} if fmt == ".3f" else {}))
        ax.set_title(title)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = (mat.max() * 0.5) if fmt == "d" else 0.5
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=11,
                        color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")


def main():
    args = parse_args()
    from config import preset_oct_smoke, preset_oct, preset_oct_large
    cfg = {"oct_smoke": preset_oct_smoke, "oct": preset_oct, "oct_large": preset_oct_large}[args.preset]()

    # CLI 覆盖
    for attr, val in [("target_model_type", args.target_model_type), ("target_epochs", args.target_epochs),
                      ("target_lr", args.target_lr), ("target_batch_size", args.target_batch_size),
                      ("target_n_hidden", args.target_n_hidden), ("n_total_samples", args.n_total_samples),
                      ("target_data_size", args.target_data_size), ("optimizer_type", args.optimizer_type),
                      ("random_seed", args.seed), ("data_dir", args.data_dir)]:
        if val is not None:
            setattr(cfg, attr, val)
    cfg.output_dir = args.output_dir if args.output_dir else cfg.output_dir.rstrip("/") + "_baseline"
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.random.seed(cfg.random_seed)

    logger.info("=" * 60 + "\nOCT UTILITY BASELINE (Experiment A)\n" + "=" * 60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    # 加载数据 + target split (忽略 shadow)
    from data import load_dataset, split_data
    X, y, groups = load_dataset(cfg)
    target_data, _ = split_data(X, y, cfg, groups=groups)
    train_X, train_y, test_X, test_y = target_data
    logger.info(f"Train: {train_X.shape}, Test: {test_X.shape}")

    # 训练
    from models import build_model, train_model, get_predictions
    n_classes = len(np.unique(y))
    n_in = X.shape[1]   # CNN 取通道数；OCT 全是 cnn
    model = build_model(cfg.target_model_type, n_in, cfg.target_n_hidden, n_classes)
    model, train_acc, test_acc = train_model(
        model, target_data, cfg.target_epochs, cfg.target_lr, cfg.target_batch_size,
        cfg.target_l2, verbose=cfg.verbose, label="Baseline",
        optimizer_type=cfg.optimizer_type, lr_decay=cfg.lr_decay)

    # 评估
    test_probs, train_probs = get_predictions(model, test_X), get_predictions(model, train_X)
    test_pred, train_pred = test_probs.argmax(axis=1), train_probs.argmax(axis=1)
    class_names = OCT_CLASS_NAMES if (cfg.dataset == "oct" and n_classes == 4) else [str(i) for i in range(n_classes)]
    report = classification_report(test_y, test_pred, target_names=class_names, zero_division=0)
    macro_f1 = f1_score(test_y, test_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(test_y, test_pred)
    cm = confusion_matrix(test_y, test_pred)

    print("\n" + "=" * 60 + "\n  OCT UTILITY BASELINE RESULTS\n" + "=" * 60)
    print(f"  Model: {cfg.target_model_type}, Epochs: {cfg.target_epochs}")
    print(f"  Train/Test samples: {len(train_y)} / {len(test_y)}")
    print(f"  Train/Test acc: {train_acc:.4f} / {test_acc:.4f}, gap={train_acc - test_acc:.4f}")
    print(f"  Balanced acc: {balanced_acc:.4f}, Macro-F1: {macro_f1:.4f}")
    print("-" * 60 + "\n" + report + "-" * 60 + f"\nConfusion Matrix:\n{cm}\n" + "=" * 60)

    # 保存
    plot_confusion_matrix(cm, class_names, os.path.join(cfg.output_dir, "baseline_confusion_matrix.png"),
                          train_acc, test_acc)
    with open(os.path.join(cfg.output_dir, "baseline_report.txt"), "w") as f:
        f.write(f"Model: {cfg.target_model_type}\nEpochs: {cfg.target_epochs}\n"
                f"Train/Test samples: {len(train_y)}/{len(test_y)}\n"
                f"Train/Test acc: {train_acc:.4f}/{test_acc:.4f}\nGap: {train_acc - test_acc:.4f}\n"
                f"Balanced acc: {balanced_acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}\n"
                f"Confusion Matrix:\n{cm}\n")
    np.savez(os.path.join(cfg.output_dir, "baseline_predictions.npz"),
             train_probs=train_probs, train_y=train_y, train_pred=train_pred,
             test_probs=test_probs, test_y=test_y, test_pred=test_pred)
    logger.info(f"Outputs saved to {cfg.output_dir}")
    return {"train_acc": train_acc, "test_acc": test_acc, "gap": train_acc - test_acc,
            "balanced_acc": balanced_acc, "macro_f1": macro_f1}


if __name__ == "__main__":
    main()
