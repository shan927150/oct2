#!/usr/bin/env python3
"""
LESS-style Adam-aware TracIn ablation for OCT MI attack attribution.

This script reuses existing attack_data.npz, retrains each per-class attack model with Adam,
saves optimizer-state checkpoints, and computes an Adam-aware influence score:

    S_adam(z, z') = sum_i eta_i < grad_loss(z'; theta_i), Gamma(z; theta_i) >

where Gamma is the Adam update direction computed from the checkpoint's optimizer states
(m, v) and the per-sample gradient of training point z.

By default we DO NOT apply cosine normalization, because LESS uses normalization mainly to
handle variable-length LLM instruction sequences; our attack samples are fixed 4D prediction
vectors. For diagnostics, the script also saves a normalized/cosine version.

Outputs are compatible with run_loo_comprehensive_from_scores.py:
  <output_dir>/tracin/scores_class{c}.npz  # key "scores" is selected variant, default Adam-dot
  <output_dir>/tracin/adam_tracin_class{c}.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_attack_with_adam_state_checkpoints(
    model: nn.Module,
    train_X: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    l2_ratio: float,
    checkpoint_every: int,
    label: str = "",
) -> Tuple[nn.Module, List[Dict]]:
    """Train attack model with Adam and save both model + optimizer states at checkpoints."""
    model = model.to(DEVICE)
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
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % checkpoint_every == 0 or epoch == epochs:
            checkpoints.append({
                "epoch": int(epoch),
                "state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(epoch_loss),
            })

    model.eval()
    logger.info(f"[{label}] trained {epochs} epochs, {len(checkpoints)} Adam-state checkpoints")
    return model, checkpoints


def _flatten_param_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().flatten().cpu() for t in tensors]).float()


def _adam_gamma_for_current_grads(model: nn.Module, optimizer: optim.Optimizer) -> torch.Tensor:
    """Compute Gamma = m_hat_next / (sqrt(v_hat_next)+eps) for current per-sample grads.

    Uses the checkpoint optimizer state as the prior Adam state. This is the post-hoc analogue
    of LESS's Adam update feature Gamma(z, theta_i).
    """
    gammas: List[torch.Tensor] = []
    group = optimizer.param_groups[0]
    beta1, beta2 = group.get("betas", (0.9, 0.999))
    eps = group.get("eps", 1e-8)

    for p in model.parameters():
        if p.grad is None:
            gammas.append(torch.zeros_like(p, device=DEVICE))
            continue
        g = p.grad.detach()
        state = optimizer.state.get(p, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(p)).to(DEVICE)
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(p)).to(DEVICE)
        step_raw = state.get("step", torch.tensor(0.0, device=DEVICE))
        step = int(step_raw.item()) if torch.is_tensor(step_raw) else int(step_raw)
        step_next = max(step + 1, 1)

        m_next = beta1 * exp_avg + (1.0 - beta1) * g
        v_next = beta2 * exp_avg_sq + (1.0 - beta2) * (g * g)
        m_hat = m_next / (1.0 - beta1 ** step_next)
        v_hat = v_next / (1.0 - beta2 ** step_next)
        gammas.append(m_hat / (torch.sqrt(v_hat) + eps))

    return _flatten_param_tensors(gammas)


def compute_grad_and_adam_features(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return raw gradients G and Adam update features Gamma for each sample: both (N, P)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    grads = torch.empty(len(X), n_params, dtype=torch.float32)
    gammas = torch.empty(len(X), n_params, dtype=torch.float32)

    for i in range(len(X)):
        model.zero_grad(set_to_none=True)
        loss = criterion(model(X_t[i:i + 1]), y_t[i:i + 1])
        loss.backward()
        grads[i] = _flatten_param_tensors([p.grad if p.grad is not None else torch.zeros_like(p)
                                           for p in model.parameters()])
        gammas[i] = _adam_gamma_for_current_grads(model, optimizer)
    return grads, gammas


def _safe_cosine_dot(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    A_norm = A / (A.norm(dim=1, keepdim=True) + eps)
    B_norm = B / (B.norm(dim=1, keepdim=True) + eps)
    return (A_norm @ B_norm.T).numpy()


def compute_scores_for_checkpoints(
    checkpoints: List[Dict],
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    model_fn: Callable[[], nn.Module],
    lr: float,
    equal_weight: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute raw TracIn, Adam-dot, and Adam-cosine scores from the same checkpoints."""
    raw_scores = None
    adam_dot_scores = None
    adam_cos_scores = None
    self_raw = np.zeros(len(train_X), dtype=np.float64)
    self_adam_dot = np.zeros(len(train_X), dtype=np.float64)
    self_adam_cos = np.zeros(len(train_X), dtype=np.float64)

    for i, ckpt in enumerate(checkpoints):
        model = model_fn().to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        eta = 1.0 if equal_weight else float(ckpt.get("lr", lr))

        train_grads, train_gamma = compute_grad_and_adam_features(model, optimizer, train_X, train_y)
        test_grads, _ = compute_grad_and_adam_features(model, optimizer, test_X, test_y)

        raw = eta * (test_grads @ train_grads.T).numpy()
        adam_dot = eta * (test_grads @ train_gamma.T).numpy()
        adam_cos = eta * _safe_cosine_dot(test_grads, train_gamma)

        raw_scores = raw if raw_scores is None else raw_scores + raw
        adam_dot_scores = adam_dot if adam_dot_scores is None else adam_dot_scores + adam_dot
        adam_cos_scores = adam_cos if adam_cos_scores is None else adam_cos_scores + adam_cos

        self_raw += eta * (train_grads ** 2).sum(dim=1).numpy()
        self_adam_dot += eta * (train_grads * train_gamma).sum(dim=1).numpy()
        self_adam_cos += eta * np.sum(
            (train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-12)).numpy()
            * (train_gamma / (train_gamma.norm(dim=1, keepdim=True) + 1e-12)).numpy(),
            axis=1,
        )
        logger.info(f"  ckpt {i+1}/{len(checkpoints)} epoch={ckpt['epoch']}: "
                    f"adam_dot range=[{adam_dot.min():.4f}, {adam_dot.max():.4f}]")

    return {
        "scores_raw": raw_scores,
        "scores_adam_dot": adam_dot_scores,
        "scores_adam_cos": adam_cos_scores,
        "self_influence_raw": self_raw,
        "self_influence_adam_dot": self_adam_dot,
        "self_influence_adam_cos": self_adam_cos,
    }


def main():
    parser = argparse.ArgumentParser(description="LESS-style Adam-aware TracIn ablation")
    parser.add_argument("--attack_data_path", type=str, default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--output_dir", type=str, default="./results/oct_mia_tracin_adam")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--variant", type=str, default="adam_dot",
                        choices=["adam_dot", "adam_cos", "raw_same_ckpt"])
    parser.add_argument("--equal_weight", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from config import preset_oct
    from models import build_attack_model, predict_attack

    cfg = preset_oct()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir) / "tracin"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.attack_data_path)
    attack_train_x, attack_train_y = ad["attack_train_x"], ad["attack_train_y"]
    train_classes = ad["train_classes"]
    attack_test_x, attack_test_y = ad["attack_test_x"], ad["attack_test_y"]
    test_classes = ad["test_classes"]
    n_in = attack_train_x.shape[1]
    logger.info(f"Loaded attack data: train={attack_train_x.shape}, test={attack_test_x.shape}")
    logger.info(f"Outputs: {out_dir}; selected score variant for key 'scores' = {args.variant}")

    for c in args.classes:
        t0 = time.time()
        logger.info(f"\n{'='*70}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*70}")
        tr_mask, te_mask = train_classes == c, test_classes == c
        c_tr_x, c_tr_y = attack_train_x[tr_mask], attack_train_y[tr_mask]
        c_te_x, c_te_y = attack_test_x[te_mask], attack_test_y[te_mask]

        def make_model():
            return build_attack_model("nn", n_in, cfg.attack_n_hidden)

        model, checkpoints = train_attack_with_adam_state_checkpoints(
            make_model(), c_tr_x, c_tr_y,
            epochs=cfg.attack_epochs, lr=cfg.attack_lr, batch_size=cfg.attack_batch_size,
            l2_ratio=cfg.attack_l2, checkpoint_every=args.checkpoint_every,
            label=f"AdamTracIn-c{c}",
        )
        pred = predict_attack(model, c_te_x)
        acc = accuracy_score(c_te_y, pred)
        prec = precision_score(c_te_y, pred, zero_division=0)
        rec = recall_score(c_te_y, pred, zero_division=0)
        logger.info(f"  Attack: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}")

        score_dict = compute_scores_for_checkpoints(
            checkpoints, c_tr_x, c_tr_y, c_te_x, c_te_y,
            make_model, lr=cfg.attack_lr, equal_weight=args.equal_weight,
        )
        if args.variant == "adam_dot":
            selected_scores = score_dict["scores_adam_dot"]
            selected_self = score_dict["self_influence_adam_dot"]
        elif args.variant == "adam_cos":
            selected_scores = score_dict["scores_adam_cos"]
            selected_self = score_dict["self_influence_adam_cos"]
        else:
            selected_scores = score_dict["scores_raw"]
            selected_self = score_dict["self_influence_raw"]

        np.savez_compressed(
            out_dir / f"scores_class{c}.npz",
            scores=selected_scores,
            self_influence=selected_self,
            train_y=c_tr_y, test_y=c_te_y, train_x=c_tr_x, test_x=c_te_x,
            **score_dict,
        )
        torch.save(checkpoints, out_dir / f"adam_checkpoints_class{c}.pt")
        with open(out_dir / f"adam_tracin_class{c}.json", "w") as f:
            json.dump({
                "class": int(c), "class_name": CLASS_NAMES.get(c, str(c)),
                "n_train": int(len(c_tr_x)), "n_test": int(len(c_te_x)),
                "variant_saved_as_scores": args.variant,
                "attack_acc": float(acc), "attack_prec": float(prec), "attack_rec": float(rec),
                "checkpoint_every": int(args.checkpoint_every),
                "n_checkpoints": int(len(checkpoints)),
                "elapsed_sec": float(time.time() - t0),
                "note": "LESS-style Adam-aware influence without cosine normalization by default; cosine scores are saved for diagnostics.",
            }, f, indent=2)
        logger.info(f"  Saved class {c} scores → {out_dir / f'scores_class{c}.npz'}")

    logger.info("Adam-aware TracIn ablation complete.")


if __name__ == "__main__":
    main()
