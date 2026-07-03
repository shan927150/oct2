#!/usr/bin/env python3
"""
Plain-SGD TracIn experiment for OCT MI attack models.

Purpose:
  Train each per-class attack model with plain SGD (momentum=0 by default),
  compute standard raw TracIn scores on the SGD checkpoints, and save outputs
  in the same format expected by the LOO scripts.

Default input:
  ./results/oct_mia_tracin/tracin/attack_data.npz

Default output:
  ./results/oct_mia_tracin_sgd/tracin/
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

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
    """Train attack model with plain SGD and save epoch-boundary checkpoints."""
    set_seed(seed)
    model = model.to(DEVICE)
    x_t = torch.tensor(train_x, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=min(batch_size, len(x_t)), shuffle=True)
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
            logger.info(f"[{label}] epoch={epoch}/{epochs}, loss={total_loss:.4f}")
    model.eval()
    logger.info(f"[{label}] trained {epochs} epochs with SGD, checkpoints={len(checkpoints)}")
    return model, checkpoints


def predict_attack(model, x):
    model.eval()
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=min(512, len(x)), shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_data_path", type=str, default="./results/oct_mia_tracin/tracin/attack_data.npz")
    parser.add_argument("--output_dir", type=str, default="./results/oct_mia_tracin_sgd")
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
    from attribution import tracin_cp, tracin_self_influence

    cfg = preset_oct()
    tracin_dir = Path(args.output_dir) / "tracin"
    tracin_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.attack_data_path)
    attack_train_x, attack_train_y, train_classes = ad["attack_train_x"], ad["attack_train_y"], ad["train_classes"]
    attack_test_x, attack_test_y, test_classes = ad["attack_test_x"], ad["attack_test_y"], ad["test_classes"]
    n_in = attack_train_x.shape[1]
    logger.info(f"Loaded attack data: train={attack_train_x.shape}, test={attack_test_x.shape}")
    logger.info(f"SGD setting: lr={args.sgd_lr}, momentum={args.sgd_momentum}, weight_decay={args.weight_decay}")

    for c in args.classes:
        t0 = time.time()
        logger.info(f"\n{'='*70}\n  CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*70}")
        tr_mask, te_mask = train_classes == c, test_classes == c
        c_tr_x, c_tr_y = attack_train_x[tr_mask], attack_train_y[tr_mask]
        c_te_x, c_te_y = attack_test_x[te_mask], attack_test_y[te_mask]
        if len(c_tr_x) < 20 or len(c_te_x) < 10:
            logger.warning(f"Skipping class {c}: too few samples")
            continue

        def make_model():
            return build_attack_model("nn", n_in, cfg.attack_n_hidden).to(DEVICE)

        model, checkpoints = train_attack_sgd_with_checkpoints(
            make_model(), c_tr_x, c_tr_y,
            epochs=args.epochs, lr=args.sgd_lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, momentum=args.sgd_momentum,
            checkpoint_every=args.checkpoint_every, seed=args.seed + c,
            label=f"SGD-Attack-c{c}",
        )
        torch.save(checkpoints, tracin_dir / f"sgd_checkpoints_class{c}.pt")

        pred = predict_attack(model, c_te_x)
        acc = accuracy_score(c_te_y, pred)
        prec = precision_score(c_te_y, pred, zero_division=0)
        rec = recall_score(c_te_y, pred, zero_division=0)
        logger.info(f"Attack perf: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}")

        self_inf = tracin_self_influence(checkpoints, c_tr_x, c_tr_y, make_model, equal_weight=True)
        scores = tracin_cp(checkpoints, c_tr_x, c_tr_y, c_te_x, c_te_y, make_model, equal_weight=True)
        si_m = self_inf[c_tr_y == 1]
        si_nm = self_inf[c_tr_y == 0]

        np.savez_compressed(
            tracin_dir / f"scores_class{c}.npz",
            scores=scores,
            self_influence=self_inf,
            train_y=c_tr_y,
            test_y=c_te_y,
            train_x=c_tr_x,
            test_x=c_te_x,
        )
        with open(tracin_dir / f"sgd_tracin_class{c}.json", "w") as f:
            json.dump({
                "class": int(c),
                "class_name": CLASS_NAMES.get(c, str(c)),
                "n_train": int(len(c_tr_x)),
                "n_test": int(len(c_te_x)),
                "optimizer": "sgd",
                "sgd_lr": float(args.sgd_lr),
                "sgd_momentum": float(args.sgd_momentum),
                "weight_decay": float(args.weight_decay),
                "epochs": int(args.epochs),
                "checkpoint_every": int(args.checkpoint_every),
                "n_checkpoints": int(len(checkpoints)),
                "attack_acc": float(acc),
                "attack_prec": float(prec),
                "attack_rec": float(rec),
                "self_influence_member_mean": float(si_m.mean()) if len(si_m) else None,
                "self_influence_nonmember_mean": float(si_nm.mean()) if len(si_nm) else None,
                "elapsed_sec": float(time.time() - t0),
            }, f, indent=2)
        logger.info(f"Saved class {c} outputs -> {tracin_dir}")

    logger.info(f"Done. Outputs in {tracin_dir}")


if __name__ == "__main__":
    main()
