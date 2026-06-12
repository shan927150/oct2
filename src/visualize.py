"""
Phase 1 MI Attack 结果 dashboard (5 panel) [ours 绘图]。
  1. per-class attack precision  (cf. paper Fig 5/6)
  2. precision & recall CDF       (cf. paper Fig 5)
  3. target/shadow/attack 准确率总览
  4. attack confusion matrix
  5. per-class precision vs recall 散点
"""
import os
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

C_GREEN, C_GREEN_L = "#27ae60", "#2ecc71"
C_BLUE, C_BLUE_L = "#2980b9", "#3498db"
C_RED, C_ORANGE, C_PURPLE, C_GRAY = "#e74c3c", "#e67e22", "#8e44ad", "#95a5a6"


def plot_dashboard(results: dict, output_dir: str = "./results") -> str:
    """生成 5-panel dashboard PNG，返回保存路径。"""
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Membership Inference Attack — Results Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs = GridSpec(2, 6, figure=fig, hspace=0.38, wspace=0.30, left=0.06, right=0.96, top=0.92, bottom=0.06)

    pcr = results.get("per_class_results", {})
    _panel_precision_dist(fig.add_subplot(gs[0, 0:2]), pcr)
    _panel_precision_recall_cdf(fig.add_subplot(gs[0, 2:4]), pcr)
    _panel_model_accuracy(fig.add_subplot(gs[0, 4:6]), results)
    _panel_confusion(fig.add_subplot(gs[1, 1:3]), results)
    _panel_pr_scatter(fig.add_subplot(gs[1, 3:5]), pcr)

    path = os.path.join(output_dir, "mi_attack_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dashboard saved to {path}")
    return path


def _panel_precision_dist(ax, pcr):
    """Panel 1: per-class precision (类别多时画直方图，否则条形)。"""
    if not pcr:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    classes = sorted(pcr.keys())
    precisions = [pcr[c]["precision"] for c in classes]
    if len(classes) > 30:
        ax.hist(precisions, bins=20, color=C_BLUE, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Precision"); ax.set_ylabel("# Classes"); ax.set_title("Attack Precision Distribution")
        ax.axvline(0.5, color=C_RED, ls="--", alpha=0.7, label="Random baseline")
    else:
        colors = [C_RED if p > 0.7 else C_ORANGE if p > 0.55 else C_GRAY for p in precisions]
        ax.bar(range(len(classes)), precisions, color=colors, edgecolor="white", lw=0.5)
        ax.axhline(0.5, color=C_RED, ls="--", alpha=0.7, label="Random baseline")
        ax.set_xlabel("Class"); ax.set_ylabel("Precision"); ax.set_title("Per-Class Attack Precision")
        ax.set_xticks(range(0, len(classes), max(1, len(classes) // 10)))
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)


def _panel_precision_recall_cdf(ax, pcr):
    """Panel 2: precision & recall CDF (cf. paper Fig 5/6)。"""
    if not pcr:
        return
    precs = sorted(pcr[c]["precision"] for c in pcr)
    recs = sorted(pcr[c]["recall"] for c in pcr)
    cdf = np.arange(1, len(precs) + 1) / len(precs)
    ax.plot(precs, cdf, "o-", color=C_BLUE, ms=3, label="Precision")
    ax.plot(recs, cdf, "s-", color=C_ORANGE, ms=3, label="Recall")
    ax.axvline(0.5, color=C_RED, ls="--", alpha=0.4, label="Random")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("Cumulative Fraction of Classes")
    ax.set_title("Precision & Recall CDF\n(cf. Paper Fig 5)")
    ax.legend(fontsize=8); ax.set_xlim(0, 1.05); ax.grid(True, alpha=0.3)


def _panel_model_accuracy(ax, r):
    """Panel 3: target/shadow/attack 准确率条形图。"""
    labels = ["Target\nTrain", "Target\nTest", "Shadow\nTrain", "Shadow\nTest", "Attack\nAccuracy"]
    values = [r.get(k, 0) for k in ("target_train_acc", "target_test_acc",
                                    "shadow_train_acc", "shadow_test_acc", "attack_accuracy")]
    bars = ax.bar(range(len(labels)), values, color=[C_GREEN, C_GREEN_L, C_BLUE, C_BLUE_L, C_RED],
                  edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15); ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax.set_title("Model Accuracy Overview")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
                fontsize=8, fontweight="bold")


def _panel_confusion(ax, r):
    """Panel 4: attack 归一化混淆矩阵 (Out/In)。"""
    true, pred = r.get("all_true", np.array([])), r.get("all_pred", np.array([]))
    if len(true) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    cm = confusion_matrix(true, pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Out", "Pred: In"]); ax.set_yticklabels(["True: Out", "True: In"])
    ax.set_title("Attack Confusion Matrix\n(Normalized)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.3f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=10, color="white" if cm_norm[i, j] > 0.5 else "black")


def _panel_pr_scatter(ax, pcr):
    """Panel 5: per-class precision vs recall 散点 (点大小 ~ 测试样本数)。"""
    if not pcr:
        return
    precs = [pcr[c]["precision"] for c in pcr]
    recs = [pcr[c]["recall"] for c in pcr]
    sizes = [pcr[c]["n_test"] for c in pcr]
    max_s = max(sizes) if sizes else 1
    ax.scatter(recs, precs, s=[30 + 100 * s / max_s for s in sizes], alpha=0.6,
               c=C_PURPLE, edgecolors="white", lw=0.5)
    ax.axhline(0.5, color=C_RED, ls="--", alpha=0.4); ax.axvline(0.5, color=C_RED, ls="--", alpha=0.4)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision vs Recall\n(size ~ test samples)")
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1); ax.grid(True, alpha=0.3)
