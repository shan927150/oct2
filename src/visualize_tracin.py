"""
Phase 2 TracIn 归因结果 dashboard (每类 6 panel) [ours 绘图]。
  1. self-influence 分布 (member vs non-member)
  2. top-K proponent 的 membership 构成
  3. top 影响样本的 prediction entropy
  4. mean influence 按 (test_label × train_label) 2×2 热图
  5. mean influence 分布 (member vs non-member)
  6. prediction entropy vs influence 散点

用法:
  python visualize_tracin.py --tracin_dir ./results/oct_mia_tracin/tracin
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


def load_class_data(tracin_dir: str, c: int):
    """读取某类的 scores_class{c}.npz + tracin_class{c}.json。"""
    scores_path = Path(tracin_dir) / f"scores_class{c}.npz"
    if not scores_path.exists():
        return None
    data = np.load(str(scores_path))
    with open(Path(tracin_dir) / f"tracin_class{c}.json") as f:
        meta = json.load(f)
    return {"scores": data["scores"], "self_influence": data["self_influence"],
            "train_y": data["train_y"], "test_y": data["test_y"],
            "train_x": data["train_x"], "test_x": data["test_x"], "meta": meta}


def plot_class_dashboard(data: dict, c: int, output_dir: str, top_k: int = 50):
    """生成某类 6-panel TracIn dashboard。"""
    scores, self_inf = data["scores"], data["self_influence"]
    tr_y, te_y, tr_x, meta = data["train_y"], data["test_y"], data["train_x"], data["meta"]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"TracIn Attribution — Class {c} ({CLASS_NAMES.get(c, c)})\n"
                 f"Attack: acc={meta['attack_acc']:.3f}, prec={meta['attack_prec']:.3f}, "
                 f"rec={meta['attack_rec']:.3f} | Train={meta['n_train']}, Test={meta['n_test']}",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30, left=0.06, right=0.96, top=0.90, bottom=0.08)
    _panel_self_influence(fig.add_subplot(gs[0, 0]), self_inf, tr_y)
    _panel_proponent_composition(fig.add_subplot(gs[0, 1]), scores, tr_y, te_y, top_k)
    _panel_pred_vec_patterns(fig.add_subplot(gs[0, 2]), scores, tr_x, top_k)
    _panel_influence_by_membership(fig.add_subplot(gs[1, 0]), scores, tr_y, te_y)
    _panel_influence_distribution(fig.add_subplot(gs[1, 1]), scores, tr_y)
    _panel_entropy_vs_influence(fig.add_subplot(gs[1, 2]), scores, tr_x, tr_y)

    path = os.path.join(output_dir, f"tracin_dashboard_class{c}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved: {path}")
    return path


def _panel_self_influence(ax, self_inf, tr_y):
    """self-influence 分布: member vs non-member。"""
    si_m, si_nm = self_inf[tr_y == 1], self_inf[tr_y == 0]
    bins = np.linspace(0, np.percentile(self_inf, 99), 40)
    ax.hist(si_m, bins=bins, alpha=0.6, color="#e74c3c", label=f"Members (n={len(si_m)})", density=True)
    ax.hist(si_nm, bins=bins, alpha=0.6, color="#3498db", label=f"Non-members (n={len(si_nm)})", density=True)
    ax.axvline(np.median(si_m), color="#e74c3c", ls="--", alpha=0.7)
    ax.axvline(np.median(si_nm), color="#3498db", ls="--", alpha=0.7)
    ax.set_xlabel("Self-Influence (||∇ℓ||²)"); ax.set_ylabel("Density")
    ax.set_title("Self-Influence Distribution"); ax.legend(fontsize=8)


def _panel_proponent_composition(ax, scores, tr_y, te_y, top_k):
    """按 membership 分组的测试点，其 top-K proponents 中 member 占比 (按 te_y 真标签分，非预测对错)。"""
    categories = ["Member\nTest", "Non-member\nTest", "All\nTest"]
    member_fracs = []
    for label_val in [1, 0, None]:
        mask = (te_y == label_val) if label_val is not None else np.ones(len(te_y), dtype=bool)
        if mask.sum() == 0:
            member_fracs.append(0); continue
        fracs = [tr_y[np.argsort(scores[idx])[::-1][:top_k]].mean() for idx in np.where(mask)[0]]
        member_fracs.append(np.mean(fracs))
    bars = ax.bar(categories, member_fracs, color=["#e74c3c", "#3498db", "#95a5a6"], edgecolor="white")
    ax.axhline(tr_y.mean(), color="gray", ls="--", alpha=0.5, label=f"Base rate: {tr_y.mean():.2f}")
    ax.set_ylabel(f"Member fraction in top-{top_k}"); ax.set_title(f"Top-{top_k} Proponent Composition")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    for bar, v in zip(bars, member_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")


def _panel_pred_vec_patterns(ax, scores, tr_x, top_k):
    """top proponents / opponents / 全体 的 prediction entropy 对比。"""
    eps = 1e-12
    entropy = -np.sum(np.clip(tr_x, eps, 1.0) * np.log(np.clip(tr_x, eps, 1.0)), axis=1)
    mean_score = scores.mean(axis=0)
    top_pro, top_opp = np.argsort(mean_score)[::-1][:top_k], np.argsort(mean_score)[:top_k]
    ent_means = [entropy[top_pro].mean(), entropy[top_opp].mean(), entropy.mean()]
    ent_stds = [entropy[top_pro].std(), entropy[top_opp].std(), entropy.std()]
    bars = ax.bar(["Top Proponents", "Top Opponents", "All Training"], ent_means, yerr=ent_stds,
                  color=["#27ae60", "#8e44ad", "#95a5a6"], edgecolor="white", capsize=4)
    ax.set_ylabel("Prediction Entropy"); ax.set_title(f"Prediction Entropy of Top-{top_k}")
    for bar, v in zip(bars, ent_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)


def _panel_influence_by_membership(ax, scores, tr_y, te_y):
    """2×2 热图: mean influence 按 (test_label, train_label)。"""
    mat = np.zeros((2, 2))
    for ti in [0, 1]:
        for tr in [0, 1]:
            te_mask, tr_mask = te_y == ti, tr_y == tr
            if te_mask.sum() and tr_mask.sum():
                mat[ti, tr] = scores[np.ix_(te_mask, tr_mask)].mean()
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Train: Out", "Train: In"]); ax.set_yticklabels(["Test: Out", "Test: In"])
    ax.set_title("Mean Influence by Membership")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if abs(mat[i, j]) > abs(mat).max() * 0.5 else "black")


def _panel_influence_distribution(ax, scores, tr_y):
    """mean influence (跨测试点) 分布: member vs non-member。"""
    mean_score = scores.mean(axis=0)
    lo, hi = np.percentile(mean_score, [1, 99])
    bins = np.linspace(lo, hi, 40)
    ax.hist(mean_score[tr_y == 1], bins=bins, alpha=0.6, color="#e74c3c", label="Members", density=True)
    ax.hist(mean_score[tr_y == 0], bins=bins, alpha=0.6, color="#3498db", label="Non-members", density=True)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Mean TracIn Score"); ax.set_ylabel("Density")
    ax.set_title("Influence Distribution"); ax.legend(fontsize=8)


def _panel_entropy_vs_influence(ax, scores, tr_x, tr_y):
    """散点: prediction entropy vs mean influence，按 membership 着色。"""
    eps = 1e-12
    entropy = -np.sum(np.clip(tr_x, eps, 1.0) * np.log(np.clip(tr_x, eps, 1.0)), axis=1)
    ax.scatter(entropy, scores.mean(axis=0), c=np.where(tr_y == 1, "#e74c3c", "#3498db"),
               alpha=0.3, s=8, edgecolors="none")
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Prediction Entropy"); ax.set_ylabel("Mean TracIn Score"); ax.set_title("Entropy vs Influence")
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Member'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Non-member')],
              fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracin_dir", type=str, default="./results/oct_mia_tracin/tracin")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    args = parser.parse_args()
    for c in args.classes:
        data = load_class_data(args.tracin_dir, c)
        if data is None:
            print(f"No data for class {c}, skipping")
            continue
        plot_class_dashboard(data, c, args.tracin_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
