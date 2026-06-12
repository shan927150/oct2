"""
数据切分: target 的 member/non-member + shadow train/test。

两种模式:
  1. image-level (balanced / random): 原 [paper] 语义 —— train=test=target_data_size,
     target 与 shadow 不重叠, shadow 之间允许重叠 [paper VI-C]
  2. group-level (patient): 同一 patient 只出现在一个集合, 保证 OCT 实验诚实性 [ours]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy_1d(y):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    return y


def _group_indices_by_class(y: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(c): np.where(y == c)[0] for c in np.unique(y)}


# =====================================================================
#  image-level split [paper VI-C]
# =====================================================================

def make_split_indices(y, target_data_size, n_shadow, random_seed=42,
                       shadow_data_size=None, disjoint_shadow_models=False, balanced=True) -> Dict:
    """图像级切分。balanced=True 类平衡抽样; False=纯随机 ([paper] 做法)。
    target_data_size = train = test，每个模型消耗 2× 该值。
    """
    y = _to_numpy_1d(y)
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)
    n_classes = len(classes)
    if shadow_data_size is None:
        shadow_data_size = target_data_size

    if balanced:
        # ---- 类平衡抽样 ----
        if target_data_size % n_classes != 0 or shadow_data_size % n_classes != 0:
            raise ValueError("target/shadow_data_size 必须能被 n_classes 整除 (balanced 模式)")
        per_t = target_data_size // n_classes          # 每类 target train = target test
        per_s = shadow_data_size // n_classes           # 每类 shadow train = shadow test
        per_s_total = 2 * per_s
        class_to_indices = _group_indices_by_class(y)

        target_train_idx, target_test_idx = [], []
        shadow_pool_by_class = {}
        for c in classes:
            c = int(c)
            idx = rng.permutation(class_to_indices[c])
            need = 2 * per_t + per_s_total
            if len(idx) < need:
                raise ValueError(f"Class {c} has {len(idx)} samples, needs ≥ {need}.")
            target_train_idx.extend(idx[:per_t].tolist())
            target_test_idx.extend(idx[per_t:2 * per_t].tolist())
            shadow_pool_by_class[c] = idx[2 * per_t:]

        target_train_idx = rng.permutation(np.array(target_train_idx)).tolist()
        target_test_idx = rng.permutation(np.array(target_test_idx)).tolist()

        shadow_models = []
        cursor = {int(c): 0 for c in classes}
        for i in range(n_shadow):
            s_train, s_test = [], []
            for c in classes:
                c = int(c)
                pool = shadow_pool_by_class[c]
                if disjoint_shadow_models:   # shadow 之间不重叠
                    start = cursor[c]
                    chosen = pool[start:start + per_s_total]
                    cursor[c] = start + per_s_total
                    if len(chosen) < per_s_total:
                        raise ValueError(f"Shadow {i} not enough samples in class {c}.")
                else:                        # shadow 之间允许重叠 ([paper] 默认)
                    chosen = rng.choice(pool, size=per_s_total, replace=False)
                chosen = rng.permutation(chosen)
                s_train.extend(chosen[:per_s].tolist())
                s_test.extend(chosen[per_s:].tolist())
            shadow_models.append({
                "train_idx": rng.permutation(np.array(s_train)).tolist(),
                "test_idx": rng.permutation(np.array(s_test)).tolist(),
            })
    else:
        # ---- 纯随机抽样 ([paper] 做法) ----
        all_idx = rng.permutation(len(y))
        target_train_idx = all_idx[:target_data_size].tolist()
        target_test_idx = all_idx[target_data_size:2 * target_data_size].tolist()
        shadow_pool = all_idx[2 * target_data_size:]

        shadow_models = []
        cursor = 0
        for i in range(n_shadow):
            if disjoint_shadow_models:
                chunk = shadow_pool[cursor:cursor + 2 * shadow_data_size]
                cursor += 2 * shadow_data_size
                if len(chunk) < 2 * shadow_data_size:
                    raise ValueError(f"Not enough data for {n_shadow} disjoint shadows.")
                chosen = rng.permutation(chunk)
            else:
                if len(shadow_pool) < 2 * shadow_data_size:
                    raise ValueError(f"Shadow pool {len(shadow_pool)} < {2 * shadow_data_size}.")
                chosen = rng.permutation(rng.choice(shadow_pool, size=2 * shadow_data_size, replace=False))
            shadow_models.append({
                "train_idx": chosen[:shadow_data_size].tolist(),
                "test_idx": chosen[shadow_data_size:].tolist(),
            })

    return {
        "meta": {"random_seed": int(random_seed), "n_classes": int(n_classes),
                 "classes": [int(c) for c in classes.tolist()],
                 "target_data_size": int(target_data_size), "shadow_data_size": int(shadow_data_size),
                 "n_shadow": int(n_shadow), "balanced": bool(balanced),
                 "disjoint_shadow_models": bool(disjoint_shadow_models), "group_split": False},
        "target_train_idx": target_train_idx, "target_test_idx": target_test_idx,
        "shadow_models": shadow_models,
    }


# =====================================================================
#  group-level (patient) split [ours]
# =====================================================================

def _log_class_distribution(name, indices, y, classes):
    if len(indices) == 0:
        logger.info(f"  {name}: empty")
        return
    subset_y = y[np.array(indices, dtype=np.int64)]
    logger.info(f"  {name} class dist: {{{', '.join(f'{int(c)}: {int((subset_y == c).sum())}' for c in classes)}}}")


def _build_patient_class_map(y, groups):
    """patient → 图像索引 / patient → 主类别(众数) / class → patients。"""
    y, groups = np.asarray(y), np.asarray(groups)
    patient_to_indices, patient_to_class = {}, {}
    for pid in np.unique(groups):
        pid = int(pid)
        mask = groups == pid
        patient_to_indices[pid] = np.where(mask)[0].tolist()
        vals, counts = np.unique(y[mask], return_counts=True)
        patient_to_class[pid] = int(vals[counts.argmax()])
    class_to_patients = {}
    for pid, cls in patient_to_class.items():
        class_to_patients.setdefault(cls, []).append(pid)
    return patient_to_indices, patient_to_class, class_to_patients


def make_group_split_indices(y, groups, target_data_size, n_shadow,
                             random_seed=42, shadow_data_size=None) -> Dict:
    """分层 patient-level 切分: 同一 patient 只在一个集合，按类分层使各集合类分布均衡。
    target_data_size 为近似值，实际大小取决于 patient 粒度。
    """
    y = _to_numpy_1d(y)
    groups = np.asarray(groups, dtype=np.int64)
    if shadow_data_size is None:
        shadow_data_size = target_data_size

    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)
    n_classes = len(classes)
    patient_to_indices, patient_to_class, class_to_patients = _build_patient_class_map(y, groups)
    total_images = len(y)

    # 按类、按图像配额把 patient 分到 target_train / target_test / shadow_pool
    target_train_patients, target_test_patients, shadow_pool_patients = [], [], []
    for c in classes:
        pats = class_to_patients.get(int(c), [])
        rng.shuffle(pats)
        cls_total = sum(len(patient_to_indices[p]) for p in pats)
        budget = max(1, int(target_data_size * cls_total / total_images))   # train、test 各一份
        count_train = count_test = 0
        phase = "train"
        for pid in pats:
            n_img = len(patient_to_indices[pid])
            if phase == "train":
                target_train_patients.append(pid); count_train += n_img
                if count_train >= budget:
                    phase = "test"
            elif phase == "test":
                target_test_patients.append(pid); count_test += n_img
                if count_test >= budget:
                    phase = "shadow"
            else:
                shadow_pool_patients.append(pid)

    def _expand(patients):
        idx = []
        for pid in patients:
            idx.extend(patient_to_indices[pid])
        return rng.permutation(np.array(idx, dtype=np.int64)).tolist()

    target_train_idx = _expand(target_train_patients)
    target_test_idx = _expand(target_test_patients)
    _log_class_distribution("target_train", target_train_idx, y, classes)
    _log_class_distribution("target_test", target_test_idx, y, classes)
    logger.info(f"[GroupSplit] target_train={len(target_train_idx)}img/{len(target_train_patients)}pat, "
                f"target_test={len(target_test_idx)}img/{len(target_test_patients)}pat, "
                f"shadow_pool={len(shadow_pool_patients)}pat")

    shadow_pool_total = sum(len(patient_to_indices[p]) for p in shadow_pool_patients)
    if shadow_pool_total < 2 * shadow_data_size:
        logger.warning(f"Shadow pool {shadow_pool_total}img < ~{2 * shadow_data_size}; shadows may be smaller.")

    # 每个 shadow 再按类分层抽 patient
    shadow_class_to_patients = {}
    for pid in shadow_pool_patients:
        shadow_class_to_patients.setdefault(patient_to_class[pid], []).append(pid)

    shadow_models = []
    for si in range(n_shadow):
        s_train, s_test = [], []
        for c in classes:
            pats = shadow_class_to_patients.get(int(c), []).copy()
            rng.shuffle(pats)
            cls_total = sum(len(patient_to_indices[p]) for p in pats)
            budget = max(1, int(shadow_data_size * cls_total / max(1, shadow_pool_total)))
            count_train = count_test = 0
            phase = "train"
            for pid in pats:
                n_img = len(patient_to_indices[pid])
                if phase == "train":
                    s_train.extend(patient_to_indices[pid]); count_train += n_img
                    if count_train >= budget:
                        phase = "test"
                elif phase == "test":
                    s_test.extend(patient_to_indices[pid]); count_test += n_img
                    if count_test >= budget:
                        break
        shadow_models.append({
            "train_idx": rng.permutation(np.array(s_train, dtype=np.int64)).tolist(),
            "test_idx": rng.permutation(np.array(s_test, dtype=np.int64)).tolist(),
        })
        logger.info(f"  Shadow-{si}: train={len(s_train)}, test={len(s_test)}")

    return {
        "meta": {"random_seed": int(random_seed), "n_classes": int(n_classes),
                 "classes": [int(c) for c in classes.tolist()],
                 "target_data_size": int(target_data_size), "shadow_data_size": int(shadow_data_size),
                 "n_shadow": int(n_shadow), "balanced": False, "disjoint_shadow_models": False,
                 "group_split": True, "actual_target_train_size": len(target_train_idx),
                 "actual_target_test_size": len(target_test_idx)},
        "target_train_idx": target_train_idx, "target_test_idx": target_test_idx,
        "shadow_models": shadow_models,
    }


# =====================================================================
#  IO + 高层入口
# =====================================================================

def save_splits(splits: Dict, path: str) -> None:
    """把切分索引存为 JSON (供复用，保证可复现)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def load_splits(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_from_indices(X, y, train_idx, test_idx):
    """按索引切出 (train_X, train_y, test_X, test_y)。"""
    X = np.asarray(X)
    y = _to_numpy_1d(y)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def split_data(X, y, cfg, split_path=None, reuse_if_exists=True, shadow_data_size=None,
               disjoint_shadow_models=False, balanced=True, return_splits=False, groups=None):
    """供 data.py 调用的高层入口: groups 非 None 走 patient-level，否则走 image-level。"""
    if split_path is not None and reuse_if_exists and Path(split_path).exists():
        splits = load_splits(split_path)
        logger.info(f"Reusing cached splits from {split_path}")
    else:
        if groups is not None:
            logger.info("Using GROUP-AWARE (patient-level) split")
            splits = make_group_split_indices(y, groups, cfg.target_data_size, cfg.n_shadow,
                                              cfg.random_seed, shadow_data_size)
        else:
            splits = make_split_indices(y, cfg.target_data_size, cfg.n_shadow, cfg.random_seed,
                                        shadow_data_size, disjoint_shadow_models, balanced)
        if split_path is not None:
            save_splits(splits, split_path)
            logger.info(f"Splits saved to {split_path}")

    target_data = build_dataset_from_indices(X, y, splits["target_train_idx"], splits["target_test_idx"])
    shadow_data_list = [build_dataset_from_indices(X, y, s["train_idx"], s["test_idx"])
                        for s in splits["shadow_models"]]
    if return_splits:
        return target_data, shadow_data_list, splits
    return target_data, shadow_data_list
