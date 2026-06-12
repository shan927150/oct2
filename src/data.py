"""
数据加载 + 切分入口。

数据集来源:
  digits/synthetic_purchase : sklearn 合成 [ours, 仿 paper purchase]
  cifar10                   : torchvision [paper VI-A]
  purchase100 / texas100    : processed Purchase/Texas-100 (与 paper VI-A 数据集兼容; 来源 privacytrustlab)
  oct                       : Kermany2018 Retinal OCT (Kaggle) [ours]
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler

from config import Config
from split import split_data as balanced_split_data

logger = logging.getLogger(__name__)


def load_dataset(cfg: Config):
    """按 cfg.dataset 分派到具体 loader，返回 (X, y, groups)；非分组数据集 groups=None。"""
    if cfg.dataset == "digits":
        X, y = _load_digits(); return X, y, None
    if cfg.dataset == "synthetic_purchase":
        X, y = _load_synthetic_purchase(cfg); return X, y, None
    if cfg.dataset == "cifar10":
        flatten = cfg.target_model_type not in ("cnn", "paper_cnn")
        X, y = _load_cifar10(flatten=flatten); return X, y, None
    if cfg.dataset == "purchase100":
        X, y = _load_purchase100(cfg); return X, y, None
    if cfg.dataset == "texas100":
        X, y = _load_texas100(cfg); return X, y, None
    if cfg.dataset == "oct":
        return _load_oct2017(cfg)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def split_data(X, y, cfg: Config, groups=None):
    """切分为 target + shadow；groups 非 None 时按 patient-level。切分索引缓存到 splits/。"""
    split_dir = Path(cfg.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    group_tag = "grp" if groups is not None else "img"
    # cache key 必须包含所有影响切分内容的字段 (改 shadow_data_size 时旧缓存不可复用)
    split_path = split_dir / (
        f"{cfg.dataset}_seed{cfg.random_seed}_n{cfg.n_total_samples}"
        f"_target{cfg.target_data_size}_shadowN{cfg.n_shadow}"
        f"_shadowSize{cfg.get_shadow_data_size()}_{group_tag}.json"
    )
    target_data, shadow_data_list = balanced_split_data(
        X, y, cfg, split_path=str(split_path), reuse_if_exists=True,
        shadow_data_size=cfg.get_shadow_data_size(),
        disjoint_shadow_models=cfg.disjoint_shadow_models,
        balanced=cfg.class_balanced_split, return_splits=False, groups=groups,
    )
    logger.info(f"Target split: train={len(target_data[1])}, test={len(target_data[3])}, "
                f"shadows={len(shadow_data_list)}")
    return target_data, shadow_data_list


# =====================================================================
#  Loaders
# =====================================================================

def _load_digits():
    """sklearn digits，标准化后返回 [ours 冒烟测试]。"""
    data = load_digits()
    X = StandardScaler().fit_transform(data.data.astype(np.float32)).astype(np.float32)
    y = data.target.astype(np.int64)
    logger.info(f"Loaded digits: X={X.shape}, classes={len(np.unique(y))}")
    return X, y


def _load_synthetic_purchase(cfg: Config):
    """合成 purchase 风格二值特征 [ours, 仿 paper VI-A purchase]。"""
    X, y = make_classification(
        n_samples=cfg.n_total_samples, n_features=cfg.n_features,
        n_informative=min(cfg.n_features // 2, cfg.n_classes * 5),
        n_redundant=cfg.n_features // 4, n_classes=cfg.n_classes,
        n_clusters_per_class=1, random_state=cfg.random_seed, flip_y=0.03,
    )
    X = (X > 0).astype(np.float32)
    y = y.astype(np.int64)
    logger.info(f"Loaded synthetic_purchase: X={X.shape}, classes={len(np.unique(y))}, density={X.mean():.3f}")
    return X, y


def _load_cifar10(flatten: bool = False):
    """torchvision CIFAR-10，标准化；flatten 供全连接模型用 [paper VI-A]。"""
    try:
        from torchvision import datasets
    except ImportError as e:
        raise ImportError("torchvision required for CIFAR-10.") from e
    root = "./data"
    train_ds = datasets.CIFAR10(root=root, train=True, download=True)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True)
    X = np.concatenate([train_ds.data, test_ds.data], axis=0).astype(np.float32) / 255.0
    y = np.concatenate([np.array(train_ds.targets), np.array(test_ds.targets)]).astype(np.int64)
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std
    if flatten:
        X = X.reshape(len(X), -1).astype(np.float32)
    logger.info(f"Loaded CIFAR-10: X={X.shape}, classes={len(np.unique(y))}, flatten={flatten}")
    return X, y


def _load_purchase100(cfg: Config):
    """processed Purchase-100 (npz 或文本)，与 paper VI-A 兼容 [privacytrustlab]。"""
    npz_path = Path(cfg.data_dir) / "purchase100.npz"
    txt_path = Path(cfg.data_dir) / "dataset_purchase"
    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        X = data["features"].astype(np.float32)
        y_raw = data["labels"]
        y = (y_raw.argmax(axis=1) if y_raw.ndim == 2 else y_raw).astype(np.int64)
    elif txt_path.exists():
        raw = np.loadtxt(str(txt_path), delimiter=",")
        y = (raw[:, 0] - 1).astype(np.int64)
        X = raw[:, 1:].astype(np.float32)
    else:
        raise FileNotFoundError(
            "Purchase-100 not found. Download from https://github.com/privacytrustlab/datasets "
            f"into '{cfg.data_dir}/'")
    logger.info(f"Loaded Purchase-100: X={X.shape}, classes={len(np.unique(y))}, density={X.mean():.3f}")
    return X, y


def _load_texas100(cfg: Config):
    """processed Texas-100 (npz 或文本)，与 paper VI-A 兼容 [privacytrustlab]。"""
    npz_path = Path(cfg.data_dir) / "texas100.npz"
    txt_path = Path(cfg.data_dir) / "dataset_texas"
    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        X = data["features"].astype(np.float32)
        y_raw = data["labels"]
        y = (y_raw.argmax(axis=1) if y_raw.ndim == 2 else y_raw).astype(np.int64)
    elif txt_path.exists():
        raw = np.loadtxt(str(txt_path), delimiter=",")
        y = (raw[:, 0] - 1).astype(np.int64)
        X = raw[:, 1:].astype(np.float32)
    else:
        raise FileNotFoundError(
            "Texas-100 not found. Download from https://github.com/privacytrustlab/datasets "
            f"into '{cfg.data_dir}/'")
    logger.info(f"Loaded Texas-100: X={X.shape}, classes={len(np.unique(y))}, features={X.shape[1]}")
    return X, y


# =====================================================================
#  OCT 2017 (Kermany) loader [ours]
# =====================================================================

def _load_oct2017(cfg: Config, img_size: int = 128):
    """加载 Kermany2018 Retinal OCT (内存友好: 先扫元数据 → patient-level 子采样 → 只读选中图像)。
    目录: {data_dir}/OCT2017/{train,test,val}/{CNV,DME,DRUSEN,NORMAL}/*.jpeg
    文件名: DISEASE-PATIENTID-IMAGENUMBER.jpeg → patient_id 来自第 2 段。
    返回 X(N,1,H,W) float32 已归一化, y(N,), groups(N,)=patient ID。
    """
    from PIL import Image

    data_root = Path(cfg.data_dir) / "OCT2017"
    if not data_root.exists():
        alt = Path(cfg.data_dir) / "kermany2018" / "OCT2017"
        if alt.exists():
            data_root = alt
        else:
            raise FileNotFoundError(
                f"OCT data not found at {data_root}. Download from "
                "https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

    # Stage 1: 扫元数据 (path, label, patient_id)，不读图像
    records, patient_id_map, next_pid = [], {}, 0
    for ci, cls in enumerate(class_names):
        for split_name in ["train", "test", "val"]:
            folder = data_root / split_name / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpeg")):
                parts = img_path.stem.split("-")
                pid_str = f"{cls}_{parts[1]}" if len(parts) >= 2 else f"{cls}_{img_path.stem}"
                if pid_str not in patient_id_map:
                    patient_id_map[pid_str] = next_pid
                    next_pid += 1
                records.append((str(img_path), ci, patient_id_map[pid_str]))
    if not records:
        raise FileNotFoundError(f"No .jpeg under {data_root}.")
    logger.info(f"OCT scan: {len(records)} images, {next_pid} patients (metadata only)")

    # Stage 2: patient-level 子采样到 n_total_samples (固定 seed 可复现)
    rng = np.random.default_rng(cfg.random_seed)
    records = [records[i] for i in rng.permutation(len(records))]
    if cfg.n_total_samples and cfg.n_total_samples < len(records):
        pid_to_idx = {}
        for i, r in enumerate(records):
            pid_to_idx.setdefault(r[2], []).append(i)
        unique_pats = np.array(list(pid_to_idx.keys()), dtype=np.int64)
        rng.shuffle(unique_pats)
        keep_mask = np.zeros(len(records), dtype=bool)
        count = 0
        for pid in unique_pats:
            n_pid = len(pid_to_idx[int(pid)])
            if count + n_pid > cfg.n_total_samples and count > 0:
                break
            for idx in pid_to_idx[int(pid)]:
                keep_mask[idx] = True
            count += n_pid
        records = [r for r, m in zip(records, keep_mask) if m]
        logger.info(f"Subsampled to {len(records)} images (patient-level)")

    # Stage 3: 只解码选中图像，预分配数组避免碎片
    n_keep = len(records)
    X = np.empty((n_keep, 1, img_size, img_size), dtype=np.float32)
    y = np.empty(n_keep, dtype=np.int64)
    groups = np.empty(n_keep, dtype=np.int64)
    log_every = max(1, n_keep // 10)
    for i, (path, ci, pid) in enumerate(records):
        img = Image.open(path).convert("L").resize((img_size, img_size))
        X[i, 0] = np.asarray(img, dtype=np.float32) / 255.0
        y[i], groups[i] = ci, pid
        if (i + 1) % log_every == 0:
            logger.info(f"  loaded {i + 1}/{n_keep}")

    # Stage 4: 全局归一化 (in-place)
    mean, std = float(X.mean()), float(X.std()) + 1e-8
    X -= mean
    X /= std
    logger.info(f"Loaded OCT: X={X.shape}, classes={len(np.unique(y))}, "
                f"patients={len(np.unique(groups))}, mean={mean:.4f}, std={std:.4f}")
    for ci, cls in enumerate(class_names):
        m = y == ci
        logger.info(f"  {cls}: {int(m.sum())} images, {len(np.unique(groups[m])) if m.any() else 0} patients")
    return X, y, groups
