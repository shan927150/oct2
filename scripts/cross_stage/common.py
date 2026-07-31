"""
common.py — shared utilities for Direction B (cross-stage MIA attribution).

The single most important function here is `reconstruct_provenance`, which maps
every row of the attack-model training set back to the raw shadow-training
sample that produced it. This is possible WITHOUT re-running any model, because
attack.py::_train_shadows builds attack_train_x by concatenating, in a fixed
order, get_predictions(shadow_i, train_X_i) then get_predictions(shadow_i,
test_X_i) for i = 0..n_shadow-1, and get_predictions preserves row order. So the
k-th "in" row of shadow i corresponds exactly to raw index
split["shadow_models"][i]["train_idx"][k].

Nothing here needs a GPU. Loading y/groups needs the OCT dataset on disk only if
you want patient_id attachment and full class cross-validation (recommended).
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


# ---------------------------------------------------------------------------
# split JSON resolution
# ---------------------------------------------------------------------------

def resolve_split_path(output_dir: str, explicit: Optional[str] = None) -> str:
    """Find the split JSON that produced the attack data.

    Preference order:
      1. explicit path if given
      2. exactly one *.json under {output_dir}/splits/
      3. raise, listing candidates, so the user picks.
    """
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"--split_path {explicit} does not exist")
        return explicit

    split_dir = Path(output_dir) / "splits"
    candidates = sorted(glob.glob(str(split_dir / "*.json")))
    if not candidates:
        # also try the conventional Phase-2 location one level up
        alt = sorted(glob.glob(str(Path(output_dir).parent / "*" / "splits" / "*.json")))
        candidates = alt
    if len(candidates) == 1:
        logger.info(f"Using split JSON: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No split JSON found under {split_dir}. Pass --split_path explicitly.")
    raise RuntimeError(
        "Multiple split JSONs found; pass --split_path to disambiguate:\n  "
        + "\n  ".join(candidates))


def load_split(split_path: str) -> Dict:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# attack_data loading
# ---------------------------------------------------------------------------

def load_attack_data(path: str) -> Dict[str, np.ndarray]:
    ad = np.load(path)
    keys = ["attack_train_x", "attack_train_y", "train_classes",
            "attack_test_x", "attack_test_y", "test_classes"]
    out = {}
    for k in keys:
        if k not in ad.files:
            raise KeyError(f"{path} missing key '{k}'. Found: {ad.files}")
        out[k] = ad[k]
    return out


def sha1_of_file(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# provenance reconstruction (the core)
# ---------------------------------------------------------------------------

def reconstruct_provenance(
    split: Dict,
    attack_train_y: np.ndarray,
    train_classes: np.ndarray,
    y_raw: Optional[np.ndarray] = None,
    groups_raw: Optional[np.ndarray] = None,
) -> Tuple[List[dict], dict]:
    """Rebuild the raw->attack row map by replaying the concatenation.

    Returns (rows, report). Each row dict has:
        global_row        int  index into attack_train_x
        shadow_id         int
        role              str  "in" (member) or "out" (nonmember)
        membership        int  1 for in, 0 for out
        raw_index         int  index into the full dataset X/y/groups
        oct_class_attack  int  train_classes[global_row]  (from attack_data)
        oct_class_raw     int  y_raw[raw_index]           (-1 if y_raw is None)
        patient_id        int  groups_raw[raw_index]      (-1 if groups_raw None)
        per_class_local   int  position within the class-c attack-train subset
                               (this is what the 804-pair csv's train_idx_A/B mean)
    """
    shadows = split["shadow_models"]
    n_expected = sum(len(s["train_idx"]) + len(s["test_idx"]) for s in shadows)
    n_actual = len(attack_train_y)

    report = {
        "n_shadows": len(shadows),
        "n_rows_from_split": int(n_expected),
        "n_rows_in_attack_data": int(n_actual),
        "structural_match": bool(n_expected == n_actual),
        "membership_mismatch": 0,
        "class_checked": y_raw is not None,
        "class_mismatch": 0,
        "patient_attached": groups_raw is not None,
    }
    if n_expected != n_actual:
        raise RuntimeError(
            f"Row-count mismatch: split implies {n_expected} attack-train rows "
            f"but attack_data has {n_actual}. The split JSON does not match this "
            f"attack_data.npz. Pass the correct --split_path.")

    rows: List[dict] = []
    g = 0
    for si, s in enumerate(shadows):
        for raw_idx in s["train_idx"]:           # in-block (members)
            rows.append(_mk_row(g, si, "in", 1, int(raw_idx), train_classes,
                                attack_train_y, y_raw, groups_raw, report))
            g += 1
        for raw_idx in s["test_idx"]:            # out-block (nonmembers)
            rows.append(_mk_row(g, si, "out", 0, int(raw_idx), train_classes,
                                attack_train_y, y_raw, groups_raw, report))
            g += 1

    # per-class local index: masking `train_classes == c` preserves global order,
    # so the per-class subset index equals the running count of that class.
    counters: Dict[int, int] = {}
    for r in rows:
        c = int(r["oct_class_attack"])
        r["per_class_local"] = counters.get(c, 0)
        counters[c] = counters.get(c, 0) + 1

    return rows, report


def _mk_row(g, si, role, membership, raw_idx, train_classes, attack_train_y,
            y_raw, groups_raw, report) -> dict:
    if int(attack_train_y[g]) != membership:
        report["membership_mismatch"] += 1
    oct_attack = int(train_classes[g])
    oct_raw = int(y_raw[raw_idx]) if y_raw is not None else -1
    if y_raw is not None and oct_raw != oct_attack:
        report["class_mismatch"] += 1
    pid = _to_python_scalar(groups_raw[raw_idx]) if groups_raw is not None else -1
    return {
        "global_row": int(g),
        "shadow_id": int(si),
        "role": role,
        "membership": int(membership),
        "raw_index": int(raw_idx),
        "oct_class_attack": oct_attack,
        "oct_class_raw": oct_raw,
        "patient_id": pid,
        "per_class_local": -1,  # filled after loop
    }


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalars to ordinary Python values while preserving strings.

    OCT patient/group identifiers may be integers or strings depending on the
    loader version.  Forcing ``int(...)`` can therefore corrupt or crash on the
    real dataset even though synthetic integer fixtures pass.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows_to_arrays(rows: List[dict]) -> Dict[str, np.ndarray]:
    """Column-oriented view for fast lookups."""
    keys = ["global_row", "shadow_id", "membership", "raw_index",
            "oct_class_attack", "oct_class_raw", "patient_id", "per_class_local"]
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out["role"] = np.array([r["role"] for r in rows])
    return out


def build_class_local_lookup(rows: List[dict]) -> Dict[Tuple[int, int], dict]:
    """(oct_class, per_class_local) -> provenance row.

    Use this to resolve the 804-pair csv, whose train_idx_A/B are per-class
    local indices within scores_class{c}.npz / loo_comprehensive_class{c}.json.
    """
    return {(int(r["oct_class_attack"]), int(r["per_class_local"])): r for r in rows}


# ---------------------------------------------------------------------------
# optional dataset loading (only when patient_id / class-validation wanted)
# ---------------------------------------------------------------------------

def try_load_dataset(preset: str = "oct", overrides: Optional[dict] = None):
    """Load (X, y, groups) via the repo's own loader. Requires OCT data on disk.

    Returns (X, y, groups) or (None, None, None) on failure (with a warning),
    so provenance can still proceed structurally.
    """
    try:
        import config as cfgmod
        from data import load_dataset
        cfg = getattr(cfgmod, f"preset_{preset}")()
        for k, v in (overrides or {}).items():
            setattr(cfg, k, v)
        X, y, groups = load_dataset(cfg)
        logger.info(f"Loaded dataset: X={X.shape}, y={y.shape}, "
                    f"groups={'yes' if groups is not None else 'none'}")
        return X, y, groups
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not load dataset ({e}). "
                       f"Provenance will omit patient_id and skip class cross-check.")
        return None, None, None
