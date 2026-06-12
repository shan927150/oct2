"""
Phase 2: TracIn Attribution for Attack Model Training Data。

实现 TracInCP (Pruthi et al. 2020, arXiv:2002.08484)，应用于 Shokri MI attack
的 attack-model 训练阶段。

核心公式 (Pruthi Eq. 1):
    TracInCP(z, z') = Σ_i η_i · ∇ℓ(w_i, z) · ∇ℓ(w_i, z')
其中 z=attack 训练样本, z'=attack 测试样本, w_i=第 i 个 checkpoint 参数,
ℓ=CrossEntropyLoss, η_i=1 (equal_weight, 见 Pruthi Sec 4.2)。
"""

import copy
import logging
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_attack_with_checkpoints(
    model: nn.Module,
    train_X: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 64,
    l2_ratio: float = 0.0,
    checkpoint_every: int = 5,
    label: str = "",
) -> Tuple[nn.Module, List[Dict]]:
    """训练 attack model (CrossEntropyLoss + Adam) 并周期性保存 checkpoint。
    返回 (model, checkpoints)，checkpoints 供 TracInCP 重放各 w_i。
    l2_ratio 默认 0: weight_decay>0 会把 wd*param 混入梯度，与 TracIn 纯 ∇ℓ 不一致。
    """
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=min(batch_size, len(X_t)), shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_ratio)

    checkpoints: List[Dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 在 epoch 边界采样 checkpoint；Pruthi Sec 3.3 "两 checkpoint 间每样本访问一次"
        # 的假设在 checkpoint_every>1 时只是近似 (论文亦如此，如每 30 个 checkpoint 取一次)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            checkpoints.append({
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
                "lr": optimizer.param_groups[0]["lr"],
                "loss": epoch_loss,
            })

    model.eval()
    logger.info(f"[{label}] trained {epochs} epochs, {len(checkpoints)} checkpoints")
    return model, checkpoints


def _compute_per_sample_gradients(model: nn.Module, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
    """逐样本计算 ∇_w ℓ(w, z_i)，返回 (N, P) CPU 张量 (P=参数总数)。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    grads = torch.empty(len(X), n_params, dtype=torch.float32)
    for i in range(len(X)):
        model.zero_grad()
        loss = criterion(model(X_t[i:i + 1]), y_t[i:i + 1])
        loss.backward()
        grads[i] = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
    return grads


def tracin_cp(
    checkpoints: List[Dict],
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    model_fn: Callable[[], nn.Module],
    equal_weight: bool = True,
) -> np.ndarray:
    """TracInCP 交叉影响 (Pruthi Eq. 1)。返回 (N_test, N_train) 分数矩阵。
    >0 → proponent (降 test loss); <0 → opponent (增 test loss)。
    equal_weight=True → η=1 (Pruthi Sec 4.2 表明等权与按 lr 加权效果相当)。
    """
    t0 = time.time()
    scores = None
    for i, ckpt in enumerate(checkpoints):
        model = model_fn()
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        eta = 1.0 if equal_weight else ckpt["lr"]

        train_grads = _compute_per_sample_gradients(model, train_X, train_y)
        test_grads = _compute_per_sample_gradients(model, test_X, test_y)
        # (N_test, P) @ (P, N_train) → (N_test, N_train)
        contrib = eta * (test_grads @ train_grads.T).numpy()
        scores = contrib if scores is None else scores + contrib
        logger.info(f"  ckpt {i+1}/{len(checkpoints)} (epoch {ckpt['epoch']}): "
                    f"range=[{contrib.min():.6f}, {contrib.max():.6f}]")

    logger.info(f"TracInCP done in {time.time() - t0:.1f}s")
    return scores


def tracin_self_influence(
    checkpoints: List[Dict], X: np.ndarray, y: np.ndarray,
    model_fn: Callable[[], nn.Module], equal_weight: bool = True,
) -> np.ndarray:
    """Self-influence (Pruthi Sec 4.1): TracInCP(z,z) = Σ_i η_i ||∇ℓ(w_i,z)||² ≥ 0。
    值越大表示模型对该样本记忆越深 (用于 mislabel detection)。
    """
    self_inf = np.zeros(len(X), dtype=np.float64)
    for ckpt in checkpoints:
        model = model_fn()
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        eta = 1.0 if equal_weight else ckpt["lr"]
        grads = _compute_per_sample_gradients(model, X, y)
        self_inf += eta * (grads ** 2).sum(dim=1).numpy()
    return self_inf


# ---- 分析辅助 [ours] ----

def top_k_proponents(scores: np.ndarray, test_idx: int, k: int = 20):
    """对某测试点，返回 TracIn 最高的 k 个训练样本索引 (proponents)。"""
    return np.argsort(scores[test_idx])[::-1][:k]


def top_k_opponents(scores: np.ndarray, test_idx: int, k: int = 20):
    """对某测试点，返回 TracIn 最负的 k 个训练样本索引 (opponents)。"""
    return np.argsort(scores[test_idx])[:k]


def aggregate_influence_by_group(scores: np.ndarray, group_labels: np.ndarray, axis: str = "train") -> Dict:
    """按类别变量 (shadow_id / membership 等) 聚合平均 |influence|。
    axis='train' 按列分组, 'test' 按行分组。
    """
    result = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        block = scores[:, mask] if axis == "train" else scores[mask, :]
        result[str(g)] = float(np.mean(np.abs(block)))
    return result


def prediction_vector_stats(pred_vecs: np.ndarray) -> Dict[str, np.ndarray]:
    """计算 prediction vector 的逐样本统计 (max_conf / entropy / margin / predicted_class)，
    用于分析高影响样本的模式 [ours]。"""
    eps = 1e-12
    p = np.clip(pred_vecs, eps, 1.0)
    sorted_p = np.sort(pred_vecs, axis=1)[:, ::-1]
    return {
        "max_conf": np.max(pred_vecs, axis=1),
        "entropy": -np.sum(p * np.log(p), axis=1),
        "margin": sorted_p[:, 0] - sorted_p[:, 1],
        "predicted_class": np.argmax(pred_vecs, axis=1),
    }
