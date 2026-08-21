#!/usr/bin/env python3
"""End-to-end patient LOO pilot across the OCT shadow -> attack pipeline.

This experiment is intentionally independent of all previous TracIn and attack-
only LOO artifacts.  It creates a fresh patient-level split and measures

    patient removal -> shadow retraining -> prediction-vector drift
                    -> attack retraining -> fixed target-query performance.

For each removed patient it evaluates the four counterfactuals from the
continuous-pathway derivation:

    J00: baseline prediction vectors, baseline membership labels
    J10: LOO prediction vectors,      baseline membership labels (value only)
    J01: baseline prediction vectors, LOO membership labels      (relabel only)
    J11: LOO prediction vectors,      LOO membership labels      (full effect)

The attack-table row universe and the target-query set are fixed.  Images from
the removed patient remain interface queries, but their attack-training label
changes from member to nonmember in J01/J11.  The primary endpoint is the
matched OCT-class attack cross-entropy change, not a macro average.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if (SRC_DIR / "config.py").is_file():
    MODULE_DIR = SRC_DIR
elif (REPO_ROOT / "config.py").is_file():
    # Delta's historical deployment has config.py/data.py/models.py/split.py
    # directly under ~/oct2 instead of under ~/oct2/src.
    MODULE_DIR = REPO_ROOT
else:
    raise RuntimeError(
        f"Cannot find project modules under {SRC_DIR} or {REPO_ROOT}")
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import preset_oct  # noqa: E402
from data import load_dataset  # noqa: E402
from models import DEVICE, build_attack_model, build_model, get_predictions  # noqa: E402
from split import split_data as indexed_split_data  # noqa: E402


LOGGER = logging.getLogger("cross_stage_patient_loo")
CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
CONDITIONS = ("J00", "J10", "J01", "J11")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fresh end-to-end OCT patient LOO with an interface checkpoint")
    ap.add_argument("--output_dir", default="./results/cross_stage_patient_loo_pilot")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_total_samples", type=int, default=12000)
    ap.add_argument("--target_data_size", type=int, default=800)
    ap.add_argument("--shadow_data_size", type=int, default=800)
    ap.add_argument("--n_shadow", type=int, default=5)
    ap.add_argument("--affected_shadow", type=int, default=0)
    ap.add_argument("--n_patients", type=int, default=8)
    ap.add_argument("--min_patient_images", type=int, default=5)
    ap.add_argument("--max_patient_images", type=int, default=15)
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--split_seed", type=int, default=42005)
    ap.add_argument("--selection_seed", type=int, default=42006)
    ap.add_argument("--target_seed", type=int, default=42007)
    ap.add_argument("--fixed_shadow_seed", type=int, default=42100)
    ap.add_argument("--shadow_epochs", type=int, default=50)
    ap.add_argument("--attack_epochs", type=int, default=50)
    ap.add_argument("--shadow_batch_size", type=int, default=128)
    ap.add_argument("--attack_batch_size", type=int, default=128)
    ap.add_argument("--shadow_lr", type=float, default=1e-3)
    ap.add_argument("--attack_lr", type=float, default=1e-2)
    ap.add_argument("--noop_replays", type=int, default=1,
                    help="same-data replay count per seed; verifies deterministic no-op floor")
    ap.add_argument("--gate_min_queries_per_label", type=int, default=50)
    ap.add_argument("--gate_min_class_auc", type=float, default=0.55)
    ap.add_argument("--gate_min_class_balanced_accuracy", type=float, default=0.53)
    ap.add_argument(
        "--enforce_attack_gate", action="store_true",
        help=("run LOO only for classes whose baseline attack passes the "
              "per-class gate for every seed; fail if no class qualifies"))
    ap.add_argument("--noop_tolerance", type=float, default=0.0)
    ap.add_argument(
        "--enforce_noop_gate", action="store_true",
        help="stop before LOO unless Stage-1 and Stage-2 no-op replays are within tolerance")
    ap.add_argument(
        "--baseline_only", action="store_true",
        help="run baseline attack qualification and complete no-op replay, then stop")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute completed patient/seed result JSON files")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # PyTorch 2.8 has no deterministic CUDA implementation for
        # adaptive_avg_pool2d_backward.  Keep all other deterministic checks
        # active and measure the residual floor with a full no-op replay.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=json_default)
    tmp.replace(path)


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def make_epoch_orders(indices: Sequence[int], epochs: int, seed: int) -> np.ndarray:
    """Explicit raw-index order; LOO filters this same order instead of reshuffling."""
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(indices) for _ in range(epochs)], axis=0)


def _iter_batches(order: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def train_classifier_from_orders(
    X: np.ndarray,
    y: np.ndarray,
    train_orders: np.ndarray,
    eval_indices: Sequence[int],
    seed: int,
    n_hidden: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    deterministic: bool,
    excluded_indices: Sequence[int] = (),
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Train SmallCNN using a recorded baseline raw-index order.

    A LOO run removes the excluded raw indices from every recorded epoch.  All
    remaining examples keep the same relative order as the paired baseline.
    """
    seed_everything(seed, deterministic)
    n_classes = int(np.max(y)) + 1
    model = build_model("cnn", X.shape[1], n_hidden, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    excluded = set(int(v) for v in excluded_indices)

    for base_order in train_orders:
        if excluded:
            order = np.asarray([v for v in base_order if int(v) not in excluded], dtype=np.int64)
        else:
            order = base_order
        if len(order) == 0:
            raise ValueError("LOO removed every Stage-1 training image")
        model.train()
        for batch_idx in _iter_batches(order, batch_size):
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    effective_train = np.asarray(
        [v for v in train_orders[0] if int(v) not in excluded], dtype=np.int64)
    metrics = {
        "train_accuracy": classification_accuracy(model, X, y, effective_train),
        "heldout_accuracy": classification_accuracy(model, X, y, eval_indices),
        "n_train_images": int(len(effective_train)),
        "n_heldout_images": int(len(eval_indices)),
    }
    model.eval()
    return model, metrics


def classification_accuracy(model, X, y, indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return float("nan")
    pred = get_predictions(model, X[idx]).argmax(axis=1)
    return float(accuracy_score(y[idx], pred))


def save_model(path: Path, model: torch.nn.Module, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_model(path: Path, n_in: int, n_hidden: int, n_classes: int) -> torch.nn.Module:
    model = build_model("cnn", n_in, n_hidden, n_classes)
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def choose_patients(
    train_indices: Sequence[int],
    y: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[int],
    n_patients: int,
    min_images: int,
    max_images: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Exactly class-balanced random selection, independent of old scores."""
    if n_patients % len(classes) != 0:
        raise ValueError(
            f"--n_patients={n_patients} must be divisible by "
            f"the number of classes ({len(classes)})")
    idx = np.asarray(train_indices, dtype=np.int64)
    rows: List[Dict[str, object]] = []
    for pid in np.unique(groups[idx]):
        patient_idx = idx[groups[idx] == pid]
        vals, counts = np.unique(y[patient_idx], return_counts=True)
        cls = int(vals[counts.argmax()])
        if cls not in classes or not min_images <= len(patient_idx) <= max_images:
            continue
        rows.append({
            "patient_id": int(pid),
            "oct_class": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "n_images": int(len(patient_idx)),
            "raw_indices": patient_idx.tolist(),
        })

    rng = np.random.default_rng(seed)
    selected: List[Dict[str, object]] = []
    quota = n_patients // len(classes)
    for cls in classes:
        pool = [r for r in rows if r["oct_class"] == cls]
        rng.shuffle(pool)
        if len(pool) < quota:
            counts = {int(c): sum(r["oct_class"] == c for r in rows) for c in classes}
            raise RuntimeError(
                f"Class {cls} has only {len(pool)} eligible patients; needs {quota}. "
                f"Eligible by class={counts}. Widen --min/--max_patient_images.")
        selected.extend(pool[:quota])
    rng.shuffle(selected)
    return selected


def make_interface(
    models: Sequence[torch.nn.Module],
    splits: Mapping[str, object],
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, np.ndarray]:
    vectors, labels, classes, shadow_ids, raw_indices = [], [], [], [], []
    for sid, (model, split) in enumerate(zip(models, splits["shadow_models"])):
        tr = np.asarray(split["train_idx"], dtype=np.int64)
        te = np.asarray(split["test_idx"], dtype=np.int64)
        rows = np.concatenate([tr, te])
        vectors.append(get_predictions(model, X[rows]))
        labels.append(np.concatenate([np.ones(len(tr), dtype=np.int64),
                                      np.zeros(len(te), dtype=np.int64)]))
        classes.append(y[rows].astype(np.int64))
        shadow_ids.append(np.full(len(rows), sid, dtype=np.int64))
        raw_indices.append(rows)
    return {
        "x": np.concatenate(vectors),
        "membership": np.concatenate(labels),
        "classes": np.concatenate(classes),
        "shadow_id": np.concatenate(shadow_ids),
        "raw_index": np.concatenate(raw_indices),
    }


def make_target_queries(model, splits, X, y) -> Dict[str, np.ndarray]:
    tr = np.asarray(splits["target_train_idx"], dtype=np.int64)
    te = np.asarray(splits["target_test_idx"], dtype=np.int64)
    rows = np.concatenate([tr, te])
    return {
        "x": get_predictions(model, X[rows]),
        "membership": np.concatenate([np.ones(len(tr), dtype=np.int64),
                                      np.zeros(len(te), dtype=np.int64)]),
        "classes": y[rows].astype(np.int64),
        "raw_index": rows,
    }


def replace_affected_vectors(
    baseline: Mapping[str, np.ndarray],
    loo_model: torch.nn.Module,
    X: np.ndarray,
    affected_shadow: int,
) -> np.ndarray:
    out = baseline["x"].copy()
    mask = baseline["shadow_id"] == affected_shadow
    out[mask] = get_predictions(loo_model, X[baseline["raw_index"][mask]])
    return out


def membership_after_removal(
    baseline: Mapping[str, np.ndarray], removed_raw_indices: Sequence[int],
    affected_shadow: int,
) -> np.ndarray:
    out = baseline["membership"].copy()
    removed = ((baseline["shadow_id"] == affected_shadow) &
               np.isin(baseline["raw_index"], np.asarray(removed_raw_indices, dtype=np.int64)))
    if not np.all(out[removed] == 1):
        raise RuntimeError("A selected patient has a nonmember interface row; split/provenance is invalid")
    out[removed] = 0
    return out


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(p: np.ndarray) -> np.ndarray:
    ordered = np.sort(p, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * ((p * np.log(p / m)).sum(axis=1) +
                  (q * np.log(q / m)).sum(axis=1))


def drift_summary(
    baseline: Mapping[str, np.ndarray],
    loo_x: np.ndarray,
    patient_id: int,
    groups: np.ndarray,
    affected_shadow: int,
) -> Dict[str, Dict[str, float]]:
    affected = baseline["shadow_id"] == affected_shadow
    deleted = affected & (groups[baseline["raw_index"]] == patient_id)
    retained_member = affected & (baseline["membership"] == 1) & ~deleted
    nonmember = affected & (baseline["membership"] == 0)
    masks = {
        "deleted_patient": deleted,
        "retained_members": retained_member,
        "nonmembers": nonmember,
        "all_affected_shadow_rows": affected,
    }
    result: Dict[str, Dict[str, float]] = {}
    for name, mask in masks.items():
        p0, p1 = baseline["x"][mask], loo_x[mask]
        cls = baseline["classes"][mask]
        if len(p0) == 0:
            result[name] = {"n": 0}
            continue
        row = np.arange(len(p0))
        result[name] = {
            "n": int(len(p0)),
            "mean_js": float(js_divergence(p0, p1).mean()),
            "mean_l1": float(np.abs(p1 - p0).sum(axis=1).mean()),
            "max_abs": float(np.abs(p1 - p0).max()),
            "mean_delta_true_probability": float((p1[row, cls] - p0[row, cls]).mean()),
            "mean_delta_entropy": float((entropy(p1) - entropy(p0)).mean()),
            "mean_delta_margin": float((margin(p1) - margin(p0)).mean()),
        }
    return result


def train_attack_model_deterministic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    n_hidden: int,
    deterministic: bool,
) -> torch.nn.Module:
    seed_everything(seed, deterministic)
    model = build_attack_model("nn", X.shape[1], n_hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    orders = make_epoch_orders(np.arange(len(X)), epochs, seed + 900000)
    for order in orders:
        model.train()
        for batch in _iter_batches(order, batch_size):
            xb = torch.as_tensor(X[batch], dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(y[batch], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_attack_prob(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.as_tensor(X[start:start + 512], dtype=torch.float32, device=DEVICE)
            chunks.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)),
        "cross_entropy": float(log_loss(y_true, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
        "member_recall": float(recall_score(y_true, pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_member": int((y_true == 1).sum()),
        "n_nonmember": int((y_true == 0).sum()),
    }


def evaluate_attack_condition(
    train_x: np.ndarray,
    train_membership: np.ndarray,
    train_classes: np.ndarray,
    target: Mapping[str, np.ndarray],
    classes: Sequence[int],
    seed: int,
    args: argparse.Namespace,
    return_probabilities: bool = False,
):
    per_class: Dict[str, object] = {}
    probabilities: Dict[str, np.ndarray] = {}
    for cls in classes:
        tr = train_classes == cls
        te = target["classes"] == cls
        if tr.sum() < 20 or te.sum() < 10:
            raise RuntimeError(f"Class {cls} has too few attack rows: train={tr.sum()}, test={te.sum()}")
        if len(np.unique(train_membership[tr])) != 2 or len(np.unique(target["membership"][te])) != 2:
            raise RuntimeError(f"Class {cls} does not contain both membership labels")
        attack_seed = int(seed * 100 + cls)
        model = train_attack_model_deterministic(
            train_x[tr], train_membership[tr], attack_seed,
            args.attack_epochs, args.attack_lr, args.attack_batch_size,
            n_hidden=64, deterministic=args.deterministic)
        prob = predict_attack_prob(model, target["x"][te])
        probabilities[str(cls)] = prob
        per_class[str(cls)] = {
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "n_train": int(tr.sum()),
            "n_train_member": int((train_membership[tr] == 1).sum()),
            "n_train_nonmember": int((train_membership[tr] == 0).sum()),
            **binary_metrics(target["membership"][te], prob),
        }
    numeric = ("accuracy", "balanced_accuracy", "auc", "cross_entropy", "member_recall")
    macro = {key: float(np.mean([per_class[str(c)][key] for c in classes])) for key in numeric}
    metrics = {"per_class": per_class, "macro": macro}
    if return_probabilities:
        return metrics, probabilities
    return metrics


def attack_gate(metrics: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    per_class = {}
    for cls, row in metrics["per_class"].items():
        failures = []
        if min(row["n_member"], row["n_nonmember"]) < args.gate_min_queries_per_label:
            failures.append(
                f"min(member,nonmember)={min(row['n_member'], row['n_nonmember'])} "
                f"< {args.gate_min_queries_per_label}")
        if row["auc"] < args.gate_min_class_auc:
            failures.append(
                f"AUC {row['auc']:.4f} < {args.gate_min_class_auc:.4f}")
        if row["balanced_accuracy"] < args.gate_min_class_balanced_accuracy:
            failures.append(
                f"balanced accuracy {row['balanced_accuracy']:.4f} "
                f"< {args.gate_min_class_balanced_accuracy:.4f}")
        per_class[cls] = {
            "class_name": row["class_name"],
            "passed": not failures,
            "failures": failures,
        }
    return {
        "policy": "each class is qualified independently on every baseline seed",
        "per_class": per_class,
        "passed_all_classes": bool(all(r["passed"] for r in per_class.values())),
        "passed_any_class": bool(any(r["passed"] for r in per_class.values())),
    }


def classes_passing_all_seeds(
    baseline_records: Mapping[str, object], classes: Sequence[int]
) -> List[int]:
    return [
        int(cls) for cls in classes
        if all(
            record["attack_gate"]["per_class"][str(cls)]["passed"]
            for record in baseline_records.values()
        )
    ]


def subset_attack_metrics(
    metrics: Mapping[str, object], classes: Sequence[int]
) -> Dict[str, object]:
    per_class = {str(cls): metrics["per_class"][str(cls)] for cls in classes}
    numeric = ("accuracy", "balanced_accuracy", "auc", "cross_entropy", "member_recall")
    macro = {
        key: float(np.mean([per_class[str(cls)][key] for cls in classes]))
        for key in numeric
    }
    return {"per_class": per_class, "macro": macro}


def endpoint_deltas(
    conditions: Mapping[str, object], matched_class: int
) -> Dict[str, object]:
    keys = ("accuracy", "balanced_accuracy", "auc", "cross_entropy", "member_recall")

    def diff(a: str, b: str, level: str, cls: str | None = None) -> Dict[str, float]:
        aa = conditions[a][level] if cls is None else conditions[a][level][cls]
        bb = conditions[b][level] if cls is None else conditions[b][level][cls]
        return {key: float(aa[key] - bb[key]) for key in keys}

    out = {
        "macro": {
            "value_J10_minus_J00": diff("J10", "J00", "macro"),
            "relabel_J01_minus_J00": diff("J01", "J00", "macro"),
            "full_J11_minus_J00": diff("J11", "J00", "macro"),
        },
        "per_class": {},
    }
    classes = conditions["J00"]["per_class"].keys()
    for cls in classes:
        out["per_class"][cls] = {
            "value_J10_minus_J00": diff("J10", "J00", "per_class", cls),
            "relabel_J01_minus_J00": diff("J01", "J00", "per_class", cls),
            "full_J11_minus_J00": diff("J11", "J00", "per_class", cls),
        }
    for level, rows in (("macro", out["macro"]),):
        rows["interaction_J11_minus_J10_minus_J01_plus_J00"] = {
            key: float(conditions["J11"][level][key] - conditions["J10"][level][key]
                       - conditions["J01"][level][key] + conditions["J00"][level][key])
            for key in keys
        }
    for cls, rows in out["per_class"].items():
        rows["interaction_J11_minus_J10_minus_J01_plus_J00"] = {
            key: float(conditions["J11"]["per_class"][cls][key]
                       - conditions["J10"]["per_class"][cls][key]
                       - conditions["J01"]["per_class"][cls][key]
                       + conditions["J00"]["per_class"][cls][key])
            for key in keys
        }
    matched_key = str(matched_class)
    if matched_key not in out["per_class"]:
        raise RuntimeError(
            f"Patient class {matched_class} was not qualified for LOO")
    out["matched_class"] = {
        "oct_class": matched_class,
        "class_name": CLASS_NAMES.get(matched_class, str(matched_class)),
        "primary_endpoint": "cross_entropy",
        **out["per_class"][matched_key],
    }
    return out


def build_config(args: argparse.Namespace):
    cfg = preset_oct()
    cfg.output_dir = str(Path(args.output_dir).resolve())
    cfg.data_dir = args.data_dir
    cfg.n_total_samples = args.n_total_samples
    cfg.target_data_size = args.target_data_size
    cfg.shadow_data_size = args.shadow_data_size
    cfg.n_shadow = args.n_shadow
    cfg.random_seed = args.split_seed
    cfg.target_epochs = args.shadow_epochs
    cfg.shadow_epochs = args.shadow_epochs
    cfg.target_batch_size = args.shadow_batch_size
    cfg.shadow_batch_size = args.shadow_batch_size
    cfg.target_lr = args.shadow_lr
    cfg.shadow_lr = args.shadow_lr
    cfg.attack_epochs = args.attack_epochs
    cfg.attack_batch_size = args.attack_batch_size
    cfg.attack_lr = args.attack_lr
    return cfg


def prepare_split(cfg, X, y, groups, out_dir: Path, overwrite: bool):
    split_path = out_dir / "splits" / "fresh_patient_split.json"
    _, _, splits = indexed_split_data(
        X, y, cfg, split_path=str(split_path), reuse_if_exists=not overwrite,
        shadow_data_size=cfg.get_shadow_data_size(),
        disjoint_shadow_models=False, balanced=False,
        return_splits=True, groups=groups)
    return split_path, splits


def validate_patient_split(splits, groups: np.ndarray) -> Dict[str, object]:
    target_train = set(groups[np.asarray(splits["target_train_idx"], dtype=np.int64)].tolist())
    target_test = set(groups[np.asarray(splits["target_test_idx"], dtype=np.int64)].tolist())
    target_overlap = target_train & target_test
    shadow_reports = []
    all_shadow_patients = set()
    for sid, split in enumerate(splits["shadow_models"]):
        train_patients = set(groups[np.asarray(split["train_idx"], dtype=np.int64)].tolist())
        test_patients = set(groups[np.asarray(split["test_idx"], dtype=np.int64)].tolist())
        overlap = train_patients & test_patients
        all_shadow_patients.update(train_patients)
        all_shadow_patients.update(test_patients)
        shadow_reports.append({
            "shadow_id": sid,
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "train_test_overlap": len(overlap),
        })
    target_shadow_overlap = (target_train | target_test) & all_shadow_patients
    passed = not target_overlap and not target_shadow_overlap and all(
        r["train_test_overlap"] == 0 for r in shadow_reports)
    report = {
        "passed": bool(passed),
        "target_train_test_overlap": len(target_overlap),
        "target_shadow_overlap": len(target_shadow_overlap),
        "shadows": shadow_reports,
    }
    if not passed:
        raise RuntimeError(f"Patient-disjointness validation failed: {report}")
    return report


def train_or_load_stage1(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    seed: int,
    args: argparse.Namespace,
    excluded_indices: Sequence[int] = (),
) -> Tuple[torch.nn.Module, Dict[str, float], np.ndarray]:
    orders = make_epoch_orders(train_indices, args.shadow_epochs, seed + 700000)
    if path.exists() and not args.overwrite:
        model = load_model(path, X.shape[1], 128, int(np.max(y)) + 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return model, dict(payload["metadata"]["metrics"]), orders
    model, metrics = train_classifier_from_orders(
        X, y, orders, eval_indices, seed=seed, n_hidden=128,
        lr=args.shadow_lr, batch_size=args.shadow_batch_size,
        weight_decay=1e-5, deterministic=args.deterministic,
        excluded_indices=excluded_indices)
    save_model(path, model, {
        "seed": seed,
        "excluded_indices": list(map(int, excluded_indices)),
        "metrics": metrics,
    })
    return model, metrics, orders


def write_flat_summary(out_dir: Path, results: Sequence[Mapping[str, object]]) -> None:
    rows = []
    for result in results:
        drifts = result["interface_drift"]
        matched = result["endpoint_deltas"]["matched_class"]
        full = matched["full_J11_minus_J00"]
        value = matched["value_J10_minus_J00"]
        relabel = matched["relabel_J01_minus_J00"]
        macro_full = result["endpoint_deltas"]["macro"]["full_J11_minus_J00"]
        rows.append({
            "seed": result["seed"],
            "patient_id": result["patient"]["patient_id"],
            "oct_class": result["patient"]["oct_class"],
            "class_name": result["patient"]["class_name"],
            "n_images": result["patient"]["n_images"],
            "deleted_mean_js": drifts["deleted_patient"]["mean_js"],
            "retained_member_mean_js": drifts["retained_members"]["mean_js"],
            "nonmember_mean_js": drifts["nonmembers"]["mean_js"],
            "all_affected_mean_js": drifts["all_affected_shadow_rows"]["mean_js"],
            "all_affected_mean_l1": drifts["all_affected_shadow_rows"]["mean_l1"],
            "all_affected_max_abs": drifts["all_affected_shadow_rows"]["max_abs"],
            "primary_delta_full_matched_cross_entropy": full["cross_entropy"],
            "delta_full_matched_accuracy": full["accuracy"],
            "delta_full_matched_balanced_accuracy": full["balanced_accuracy"],
            "delta_full_matched_auc": full["auc"],
            "delta_value_matched_cross_entropy": value["cross_entropy"],
            "delta_relabel_matched_cross_entropy": relabel["cross_entropy"],
            "delta_full_macro_cross_entropy": macro_full["cross_entropy"],
            "delta_full_macro_auc": macro_full["auc"],
        })
    path = out_dir / "patient_seed_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def aggregate_analysis(
    results: Sequence[Mapping[str, object]],
    baseline_records: Mapping[str, object],
    noop_records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Small-pilot descriptive analysis; no confirmatory p-value claim."""
    from scipy.stats import spearmanr

    by_patient: Dict[int, List[Mapping[str, object]]] = {}
    for result in results:
        by_patient.setdefault(int(result["patient"]["patient_id"]), []).append(result)
    patient_rows = []
    for pid, runs in sorted(by_patient.items()):
        drift_arrays = {
            "deleted_patient_mean_js": np.asarray([
                r["interface_drift"]["deleted_patient"]["mean_js"] for r in runs]),
            "retained_members_mean_js": np.asarray([
                r["interface_drift"]["retained_members"]["mean_js"] for r in runs]),
            "nonmembers_mean_js": np.asarray([
                r["interface_drift"]["nonmembers"]["mean_js"] for r in runs]),
            "all_affected_mean_js": np.asarray([
                r["interface_drift"]["all_affected_shadow_rows"]["mean_js"] for r in runs]),
        }

        def matched_effect(pathway: str, metric: str) -> np.ndarray:
            return np.asarray([
                r["endpoint_deltas"]["matched_class"][pathway][metric]
                for r in runs
            ])

        d_ce = matched_effect("full_J11_minus_J00", "cross_entropy")
        d_auc = matched_effect("full_J11_minus_J00", "auc")
        d_bacc = matched_effect("full_J11_minus_J00", "balanced_accuracy")
        d_acc = matched_effect("full_J11_minus_J00", "accuracy")
        d_value_ce = matched_effect("value_J10_minus_J00", "cross_entropy")
        d_relabel_ce = matched_effect("relabel_J01_minus_J00", "cross_entropy")

        def sign_concordance(values: np.ndarray) -> float | None:
            if len(values) < 2:
                return None
            return float(max((values > 0).mean(), (values < 0).mean()))

        patient_rows.append({
            "patient_id": pid,
            "oct_class": int(runs[0]["patient"]["oct_class"]),
            "class_name": runs[0]["patient"]["class_name"],
            "n_images": int(runs[0]["patient"]["n_images"]),
            "n_seeds": len(runs),
            **{key: float(values.mean()) for key, values in drift_arrays.items()},
            "mean_delta_full_matched_cross_entropy": float(d_ce.mean()),
            "mean_delta_full_matched_auc": float(d_auc.mean()),
            "mean_delta_full_matched_balanced_accuracy": float(d_bacc.mean()),
            "mean_delta_full_matched_accuracy": float(d_acc.mean()),
            "mean_delta_value_matched_cross_entropy": float(d_value_ce.mean()),
            "mean_delta_relabel_matched_cross_entropy": float(d_relabel_ce.mean()),
            "seed_sign_concordance_matched_cross_entropy": sign_concordance(d_ce),
            "seed_sign_concordance_matched_auc": sign_concordance(d_auc),
        })

    def rho(drift_key: str, effect_key: str) -> Dict[str, float | int | None]:
        if len(patient_rows) < 3:
            return {"n": len(patient_rows), "rho": None, "pvalue_descriptive_only": None}
        x = [r[drift_key] for r in patient_rows]
        yv = [abs(r[effect_key]) for r in patient_rows]
        value, pvalue = spearmanr(x, yv)
        return {
            "n": len(patient_rows),
            "rho": float(value) if np.isfinite(value) else None,
            "pvalue_descriptive_only": float(pvalue) if np.isfinite(pvalue) else None,
        }

    def incremental_r2(drift_key: str, effect_key: str) -> Dict[str, object]:
        """Incremental R2 of drift after controlling image count and OCT class."""
        n = len(patient_rows)
        if n < 4:
            return {"n": n, "incremental_r2": None, "reason": "fewer than 4 patients"}
        n_images = np.asarray([r["n_images"] for r in patient_rows], dtype=float)
        classes = np.asarray([r["oct_class"] for r in patient_rows], dtype=int)
        drift = np.asarray([r[drift_key] for r in patient_rows], dtype=float)
        outcome = np.abs(np.asarray([r[effect_key] for r in patient_rows], dtype=float))
        if np.std(drift) == 0 or np.std(outcome) == 0:
            return {"n": n, "incremental_r2": None, "reason": "constant drift or outcome"}

        def zscore(values: np.ndarray) -> np.ndarray:
            sd = values.std()
            return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)

        class_levels = sorted(set(classes.tolist()))
        class_columns = [
            (classes == cls).astype(float) for cls in class_levels[1:]
        ]
        base_columns = [np.ones(n), zscore(n_images), *class_columns]
        x_base = np.column_stack(base_columns)
        x_full = np.column_stack([*base_columns, zscore(drift)])

        def r_squared(design: np.ndarray) -> float:
            fitted = design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
            total = float(np.sum((outcome - outcome.mean()) ** 2))
            residual = float(np.sum((outcome - fitted) ** 2))
            return 1.0 - residual / total

        base_r2 = r_squared(x_base)
        full_r2 = r_squared(x_full)
        return {
            "n": n,
            "controls": ["n_images", "oct_class"],
            "outcome": f"abs({effect_key})",
            "baseline_r2": float(base_r2),
            "full_r2": float(full_r2),
            "incremental_r2": float(full_r2 - base_r2),
            "descriptive_only": True,
        }

    baseline_by_class = {}
    if baseline_records:
        class_keys = next(iter(baseline_records.values()))["attack"]["per_class"].keys()
        for cls in class_keys:
            rows = [r["attack"]["per_class"][cls] for r in baseline_records.values()]
            baseline_by_class[cls] = {
                "class_name": rows[0]["class_name"],
                "auc_mean": float(np.mean([r["auc"] for r in rows])),
                "auc_sd": float(np.std([r["auc"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rows])),
                "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in rows])),
            }

    drift_keys = (
        "deleted_patient_mean_js", "retained_members_mean_js",
        "nonmembers_mean_js", "all_affected_mean_js")
    primary_effect = "mean_delta_full_matched_cross_entropy"
    return {
        "qualification": "pilot/descriptive; matched-class CE is primary; no confirmatory p-values",
        "patient_seed_aggregation": patient_rows,
        "interface_vs_abs_primary_effect": {
            key: rho(key, primary_effect) for key in drift_keys
        },
        "n_images_and_class_adjusted_incremental_r2": {
            key: incremental_r2(key, primary_effect) for key in drift_keys
        },
        "baseline_by_class": baseline_by_class,
        "deterministic_noop": {
            "passed_all": bool(noop_records) and bool(
                all(r["passed"] for r in noop_records)),
            "max_stage1_abs": float(max(
                (r["stage1_max_abs"] for r in noop_records), default=0.0)),
            "max_stage2_abs": float(max(
                (r["stage2_max_abs"] for r in noop_records), default=0.0)),
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout, force=True)
    started = time.time()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)

    if args.affected_shadow < 0 or args.affected_shadow >= args.n_shadow:
        raise ValueError("--affected_shadow must be within [0, n_shadow)")
    if not args.classes:
        raise ValueError("--classes cannot be empty")
    if len(set(args.classes)) != len(args.classes):
        raise ValueError("--classes must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if args.noop_replays < 1 and args.enforce_noop_gate:
        raise ValueError("--enforce_noop_gate requires --noop_replays >= 1")
    if args.noop_tolerance < 0:
        raise ValueError("--noop_tolerance must be nonnegative")
    LOGGER.info("Project module layout: %s", MODULE_DIR)

    cfg = build_config(args)
    config_path = out_dir / "experiment_config.json"
    config_payload = {
        "args": vars(args), "oct_config": asdict(cfg),
        "estimands": {
            "J00": "baseline vectors + baseline membership",
            "J10": "LOO vectors + baseline membership (value pathway)",
            "J01": "baseline vectors + LOO membership (relabel pathway)",
            "J11": "LOO vectors + LOO membership (full patient LOO)",
            "primary": "patient-matched OCT-class cross-entropy: J11 - J00",
            "mechanistic": "patient-matched OCT-class cross-entropy: J10 - J00",
            "secondary": "matched-class balanced accuracy, accuracy, AUC; macro metrics",
        },
        "gate_policy": (
            "A class enters LOO only if its baseline attack passes query-count, AUC, "
            "and balanced-accuracy thresholds for every seed"),
        "primary_sign": (
            "positive matched-class CE change means removal hurts the attack "
            "(the patient was attack-supporting)"),
        "forbidden_inputs": ["scores_class*.npz", "old loo_*.json", "old attack_data.npz"],
    }
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        ignored = {"overwrite", "baseline_only"}
        before = {k: v for k, v in previous.get("args", {}).items() if k not in ignored}
        now = {k: v for k, v in vars(args).items() if k not in ignored}
        if before != now:
            raise RuntimeError(
                f"Output directory already contains a different configuration: {config_path}. "
                "Use a new --output_dir, or use --overwrite only when intentional.")
    json_dump(config_path, config_payload)

    LOGGER.info("Loading a fresh patient-complete OCT subset")
    X, y, groups = load_dataset(cfg)
    if groups is None:
        raise RuntimeError("This experiment requires patient/group identifiers")
    split_path, splits = prepare_split(cfg, X, y, groups, out_dir, args.overwrite)
    split_hash = sha256_file(split_path)
    split_validation = validate_patient_split(splits, groups)
    json_dump(out_dir / "split_validation.json", split_validation)
    LOGGER.info("Fresh split: %s (sha256=%s)", split_path, split_hash)

    affected_split = splits["shadow_models"][args.affected_shadow]
    candidates_path = out_dir / "selected_patients.json"
    if candidates_path.exists() and not args.overwrite:
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["patients"]
    else:
        candidates = choose_patients(
            affected_split["train_idx"], y, groups, args.classes, args.n_patients,
            args.min_patient_images, args.max_patient_images, args.selection_seed)
        json_dump(candidates_path, {
            "selection_rule": (
                "exactly class-balanced random selection within patient image-count bounds; "
                "no old scores or LOO outcomes"),
            "selection_seed": args.selection_seed,
            "split_sha256": split_hash,
            "patients": candidates,
        })
    LOGGER.info("Selected patient IDs: %s", [r["patient_id"] for r in candidates])

    # The target and unaffected shadows are trained exactly once and then frozen.
    LOGGER.info("Training/loading fixed target model")
    target_path = out_dir / "checkpoints" / "target_fixed.pt"
    target_model, target_metrics, _ = train_or_load_stage1(
        target_path, X, y, splits["target_train_idx"], splits["target_test_idx"],
        args.target_seed, args)
    target_queries = make_target_queries(target_model, splits, X, y)
    json_dump(out_dir / "target_query_manifest.json", {
        "split_sha256": split_hash,
        "metrics": target_metrics,
        "n_rows": len(target_queries["x"]),
        "class_counts": {str(c): int((target_queries["classes"] == c).sum()) for c in args.classes},
        "member_counts": {str(m): int((target_queries["membership"] == m).sum()) for m in [0, 1]},
        "raw_indices": target_queries["raw_index"],
    })

    fixed_models: Dict[int, torch.nn.Module] = {}
    for sid, shadow_split in enumerate(splits["shadow_models"]):
        if sid == args.affected_shadow:
            continue
        LOGGER.info("Training/loading fixed shadow %d", sid)
        path = out_dir / "checkpoints" / f"shadow_{sid}_fixed.pt"
        model, _, _ = train_or_load_stage1(
            path, X, y, shadow_split["train_idx"], shadow_split["test_idx"],
            args.fixed_shadow_seed + sid, args)
        fixed_models[sid] = model

    # ------------------------------------------------------------------
    # Phase A: baseline qualification for every seed before any LOO.
    # ------------------------------------------------------------------
    baseline_records: Dict[str, object] = {}
    baseline_contexts: Dict[int, object] = {}
    for seed in args.seeds:
        LOGGER.info("=== Baseline qualification seed %d ===", seed)
        baseline_path = out_dir / "checkpoints" / f"shadow_{args.affected_shadow}_baseline_seed{seed}.pt"
        baseline_model, baseline_shadow_metrics, orders = train_or_load_stage1(
            baseline_path, X, y, affected_split["train_idx"], affected_split["test_idx"],
            seed, args)
        np.savez_compressed(out_dir / f"stage1_order_seed{seed}.npz", raw_index_order=orders)

        models = []
        for sid in range(args.n_shadow):
            models.append(baseline_model if sid == args.affected_shadow else fixed_models[sid])
        interface0 = make_interface(models, splits, X, y)
        baseline_attack, baseline_probabilities = evaluate_attack_condition(
            interface0["x"], interface0["membership"], interface0["classes"],
            target_queries, args.classes, seed, args, return_probabilities=True)
        gate = attack_gate(baseline_attack, args)
        baseline_records[str(seed)] = {
            "shadow_metrics": baseline_shadow_metrics,
            "attack": baseline_attack,
            "attack_gate": gate,
        }
        json_dump(out_dir / f"baseline_seed{seed}.json", baseline_records[str(seed)])
        baseline_contexts[seed] = {
            "shadow_metrics": baseline_shadow_metrics,
            "interface": interface0,
            "attack": baseline_attack,
            "attack_probabilities": baseline_probabilities,
        }
        for cls in args.classes:
            class_gate = gate["per_class"][str(cls)]
            log = LOGGER.info if class_gate["passed"] else LOGGER.warning
            log(
                "Attack gate seed=%d class=%s passed=%s failures=%s",
                seed, CLASS_NAMES.get(cls, str(cls)), class_gate["passed"],
                class_gate["failures"])

    qualified_classes = classes_passing_all_seeds(baseline_records, args.classes)
    skipped_classes = [cls for cls in args.classes if cls not in qualified_classes]
    gate_summary = {
        "policy": "per-class gate must pass for every seed before that class enters LOO",
        "thresholds": {
            "min_queries_per_membership_label": args.gate_min_queries_per_label,
            "min_class_auc": args.gate_min_class_auc,
            "min_class_balanced_accuracy": args.gate_min_class_balanced_accuracy,
        },
        "requested_classes": list(args.classes),
        "qualified_classes": qualified_classes,
        "skipped_classes": skipped_classes,
        "by_seed": {
            seed: record["attack_gate"] for seed, record in baseline_records.items()
        },
    }
    json_dump(out_dir / "attack_gate_summary.json", gate_summary)

    if args.enforce_attack_gate and not qualified_classes:
        failure_summary = {
            "status": "stopped_attack_gate_no_qualified_class",
            "split_sha256": split_hash,
            "attack_gate": gate_summary,
            "elapsed_seconds": time.time() - started,
        }
        json_dump(out_dir / "experiment_summary.json", failure_summary)
        raise RuntimeError(
            "Attack gate stopped LOO: no requested class passed on every seed. "
            f"See {out_dir / 'attack_gate_summary.json'}")

    loo_classes = qualified_classes if args.enforce_attack_gate else list(args.classes)
    if skipped_classes and args.enforce_attack_gate:
        LOGGER.warning(
            "Skipping unqualified classes: %s; continuing qualified classes: %s",
            [CLASS_NAMES[c] for c in skipped_classes],
            [CLASS_NAMES[c] for c in loo_classes])

    # ------------------------------------------------------------------
    # Phase B: complete no-op replay through Stage 1 and Stage 2.
    # ------------------------------------------------------------------
    noop_records: List[Mapping[str, object]] = []
    for seed in args.seeds:
        context = baseline_contexts[seed]
        interface0 = context["interface"]
        baseline_probabilities = context["attack_probabilities"]

        for replay in range(args.noop_replays):
            noop_path = out_dir / "checkpoints" / f"shadow_{args.affected_shadow}_noop_seed{seed}_r{replay}.pt"
            noop_model, _, _ = train_or_load_stage1(
                noop_path, X, y, affected_split["train_idx"], affected_split["test_idx"],
                seed, args)
            mask = interface0["shadow_id"] == args.affected_shadow
            noop_x = get_predictions(noop_model, X[interface0["raw_index"][mask]])
            baseline_x = interface0["x"][mask]
            stage1_abs = np.abs(noop_x - baseline_x)
            noop_interface_x = replace_affected_vectors(
                interface0, noop_model, X, args.affected_shadow)
            _, noop_probabilities = evaluate_attack_condition(
                noop_interface_x, interface0["membership"], interface0["classes"],
                target_queries, args.classes, seed, args, return_probabilities=True)
            stage2_by_class = {}
            for cls in args.classes:
                key = str(cls)
                difference = np.abs(
                    noop_probabilities[key] - baseline_probabilities[key])
                stage2_by_class[key] = {
                    "class_name": CLASS_NAMES.get(cls, str(cls)),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "exact": bool(np.array_equal(
                        noop_probabilities[key], baseline_probabilities[key])),
                }
            stage1_max_abs = float(stage1_abs.max())
            stage2_max_abs = float(max(
                row["max_abs"] for row in stage2_by_class.values()))
            noop_records.append({
                "seed": seed, "replay": replay,
                "tolerance": args.noop_tolerance,
                "stage1_mean_l1": float(stage1_abs.sum(axis=1).mean()),
                "stage1_max_abs": stage1_max_abs,
                "stage1_exact": bool(np.array_equal(noop_x, baseline_x)),
                "stage2_by_class": stage2_by_class,
                "stage2_max_abs": stage2_max_abs,
                "stage2_exact_all_classes": bool(all(
                    row["exact"] for row in stage2_by_class.values())),
                "passed": bool(
                    stage1_max_abs <= args.noop_tolerance
                    and stage2_max_abs <= args.noop_tolerance),
            })
    json_dump(out_dir / "no_op_replays.json", noop_records)

    noop_passed = bool(noop_records) and all(r["passed"] for r in noop_records)
    if args.enforce_noop_gate and not noop_passed:
        failure_summary = {
            "status": "stopped_noop_gate",
            "split_sha256": split_hash,
            "attack_gate": gate_summary,
            "no_op_replays": noop_records,
            "elapsed_seconds": time.time() - started,
        }
        json_dump(out_dir / "experiment_summary.json", failure_summary)
        raise RuntimeError(
            "No-op gate stopped LOO: Stage-1 or Stage-2 replay exceeded "
            f"tolerance {args.noop_tolerance}. See {out_dir / 'no_op_replays.json'}")

    if args.baseline_only:
        baseline_only_summary = {
            "status": "baseline_qualification_complete",
            "split_sha256": split_hash,
            "device": str(DEVICE),
            "attack_gate": gate_summary,
            "loo_classes_if_resumed": loo_classes,
            "no_op_replays": noop_records,
            "elapsed_seconds": time.time() - started,
            "next_step": "rerun the same output directory without --baseline_only",
        }
        json_dump(out_dir / "experiment_summary.json", baseline_only_summary)
        LOGGER.info("Baseline qualification complete. Outputs: %s", out_dir)
        return

    # ------------------------------------------------------------------
    # Phase C: patient LOO only for qualified classes.
    # ------------------------------------------------------------------
    eligible_candidates = [
        patient for patient in candidates
        if int(patient["oct_class"]) in loo_classes
    ]
    all_results: List[Mapping[str, object]] = []
    for seed in args.seeds:
        context = baseline_contexts[seed]
        interface0 = context["interface"]
        baseline_shadow_metrics = context["shadow_metrics"]
        baseline_attack = subset_attack_metrics(context["attack"], loo_classes)

        for patient in eligible_candidates:
            pid = int(patient["patient_id"])
            run_path = out_dir / "runs" / f"seed{seed}_patient{pid}.json"
            if run_path.exists() and not args.overwrite:
                LOGGER.info("[resume] %s", run_path.name)
                all_results.append(json.loads(run_path.read_text(encoding="utf-8")))
                continue
            LOGGER.info("LOO patient=%d class=%s images=%d", pid, patient["class_name"], patient["n_images"])
            removed = list(map(int, patient["raw_indices"]))
            loo_path = out_dir / "checkpoints" / f"shadow_{args.affected_shadow}_seed{seed}_patient{pid}.pt"
            loo_model, loo_shadow_metrics, _ = train_or_load_stage1(
                loo_path, X, y, affected_split["train_idx"], affected_split["test_idx"],
                seed, args, excluded_indices=removed)
            interface1_x = replace_affected_vectors(interface0, loo_model, X, args.affected_shadow)
            membership1 = membership_after_removal(
                interface0, removed, args.affected_shadow)

            affected_mask = interface0["shadow_id"] == args.affected_shadow
            np.savez_compressed(
                out_dir / "runs" / f"seed{seed}_patient{pid}_interface.npz",
                raw_index=interface0["raw_index"][affected_mask],
                oct_class=interface0["classes"][affected_mask],
                baseline_membership=interface0["membership"][affected_mask],
                loo_membership=membership1[affected_mask],
                is_deleted_patient=(groups[interface0["raw_index"][affected_mask]] == pid),
                p_baseline=interface0["x"][affected_mask],
                p_loo=interface1_x[affected_mask],
            )

            conditions = {
                "J00": baseline_attack,
                "J10": evaluate_attack_condition(
                    interface1_x, interface0["membership"], interface0["classes"],
                    target_queries, loo_classes, seed, args),
                "J01": evaluate_attack_condition(
                    interface0["x"], membership1, interface0["classes"],
                    target_queries, loo_classes, seed, args),
                "J11": evaluate_attack_condition(
                    interface1_x, membership1, interface0["classes"],
                    target_queries, loo_classes, seed, args),
            }
            result = {
                "seed": seed,
                "split_sha256": split_hash,
                "affected_shadow": args.affected_shadow,
                "patient": patient,
                "stage1": {
                    "baseline": baseline_shadow_metrics,
                    "loo": loo_shadow_metrics,
                    "delta_heldout_accuracy": float(
                        loo_shadow_metrics["heldout_accuracy"] - baseline_shadow_metrics["heldout_accuracy"]),
                },
                "interface_drift": drift_summary(
                    interface0, interface1_x, pid, groups, args.affected_shadow),
                "conditions": conditions,
                "endpoint_deltas": endpoint_deltas(
                    conditions, int(patient["oct_class"])),
            }
            json_dump(run_path, result)
            all_results.append(result)
            write_flat_summary(out_dir, all_results)

    final = {
        "status": "complete",
        "split_sha256": split_hash,
        "device": str(DEVICE),
        "n_patient_seed_runs": len(all_results),
        "n_selected_patients": len(candidates),
        "n_loo_patients": len(eligible_candidates),
        "qualified_classes": loo_classes,
        "skipped_classes": skipped_classes if args.enforce_attack_gate else [],
        "attack_gate": gate_summary,
        "baseline_by_seed": baseline_records,
        "no_op_replays": noop_records,
        "elapsed_seconds": time.time() - started,
        "primary_estimand": (
            "patient-matched OCT-class cross-entropy J11 - J00 "
            "(full patient LOO)"),
        "mechanistic_estimand": (
            "patient-matched OCT-class cross-entropy J10 - J00 "
            "(continuous value pathway)"),
        "secondary_endpoints": [
            "matched-class balanced accuracy", "matched-class accuracy",
            "matched-class AUC", "macro metrics"],
    }
    analysis = aggregate_analysis(all_results, baseline_records, noop_records)
    final["analysis"] = analysis
    json_dump(out_dir / "analysis_summary.json", analysis)
    json_dump(out_dir / "experiment_summary.json", final)
    write_flat_summary(out_dir, all_results)
    LOGGER.info("Complete. Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
