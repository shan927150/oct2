#!/usr/bin/env python3
"""
Phase 2 主入口: 生成/加载 attack data → 带 checkpoint 重训 attack model →
算 TracInCP (self + cross influence) → 分析高影响样本 → 存结果到 <output_dir>/tracin/。

方法来源: TracInCP [Pruthi 2020 Eq.1 / Sec 4.1]；attack data 流程 [Shokri Fig 3]。
CLI 默认: checkpoint_every=5, top_k=50, equal_weight=True [ours，参 Pruthi Sec 4.2]。

用法:
  python run_tracin.py --preset oct --output_dir ./results/oct_mia_tracin --checkpoint_every 5
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


def _build_shadow_ids(shadow_data_list):
    """复原每个 attack-训练样本来自哪个 shadow (pipeline 按 shadow 顺序追加 member+nonmember)。"""
    ids = []
    for i, (_, trY, _, teY) in enumerate(shadow_data_list):
        ids.extend([i] * (len(trY) + len(teY)))
    return np.array(ids, dtype=np.int32)


def save_attack_data(path, **arrays):
    np.savez_compressed(path, **arrays)
    logger.info(f"Attack data saved → {path}")


def load_attack_data(path):
    logger.info(f"Attack data loaded ← {path}")
    return dict(np.load(path))


def _generate_attack_data(cfg):
    """跑 target + shadows 生成 attack train/test prediction vectors [Shokri Fig 3]。"""
    from data import load_dataset, split_data
    from models import build_model, train_model, get_predictions

    X, y, groups = load_dataset(cfg)
    target_data, shadow_data_list = split_data(X, y, cfg, groups=groups)
    n_classes = len(np.unique(y))
    n_in = X.shape[1]

    # target → attack TEST 集
    train_X, train_y_t, test_X, test_y_t = target_data
    target_model = build_model(cfg.target_model_type, n_in, cfg.target_n_hidden, n_classes)
    target_model, tr_acc, te_acc = train_model(
        target_model, target_data, cfg.target_epochs, cfg.target_lr, cfg.target_batch_size,
        cfg.target_l2, verbose=cfg.verbose, label="Target",
        optimizer_type=cfg.optimizer_type, lr_decay=cfg.lr_decay)
    logger.info(f"Target: train_acc={tr_acc:.4f}, test_acc={te_acc:.4f}")
    attack_test_x = np.concatenate([get_predictions(target_model, train_X),
                                    get_predictions(target_model, test_X)])
    attack_test_y = np.concatenate([np.ones(len(train_X), dtype=np.int64),
                                    np.zeros(len(test_X), dtype=np.int64)])
    test_classes = np.concatenate([train_y_t, test_y_t])

    # shadows → attack TRAIN 集
    all_x, all_y, all_cls = [], [], []
    for i, sd in enumerate(shadow_data_list):
        logger.info(f"Shadow {i+1}/{len(shadow_data_list)}")
        s_trX, s_trY, s_teX, s_teY = sd
        shadow_model = build_model(cfg.get_shadow_model_type(), n_in, cfg.get_shadow_n_hidden(), n_classes)
        shadow_model, _, _ = train_model(
            shadow_model, sd, cfg.get_shadow_epochs(), cfg.get_shadow_lr(), cfg.get_shadow_batch_size(),
            cfg.get_shadow_l2(), verbose=False, label=f"Shadow-{i}",
            optimizer_type=cfg.optimizer_type, lr_decay=cfg.lr_decay)
        all_x += [get_predictions(shadow_model, s_trX), get_predictions(shadow_model, s_teX)]
        all_y += [np.ones(len(s_trX), dtype=np.int64), np.zeros(len(s_teX), dtype=np.int64)]
        all_cls += [s_trY, s_teY]

    return {
        "attack_train_x": np.concatenate(all_x), "attack_train_y": np.concatenate(all_y),
        "train_classes": np.concatenate(all_cls), "train_shadow_ids": _build_shadow_ids(shadow_data_list),
        "attack_test_x": attack_test_x, "attack_test_y": attack_test_y,
        "test_classes": test_classes, "n_classes": np.array(n_classes),
    }


def main():
    parser = argparse.ArgumentParser(description="TracIn Attribution (Phase 2)")
    parser.add_argument("--preset", type=str, default="oct", choices=["oct_smoke", "oct", "oct_large"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--attack_data_path", type=str, default=None)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--top_k", type=int, default=50)
    # 注: store_true + default=True → 该 flag 实际恒为 True (始终等权)，与 Pruthi Sec 4.2 一致
    parser.add_argument("--equal_weight", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from config import preset_oct_smoke, preset_oct, preset_oct_large
    cfg = {"oct_smoke": preset_oct_smoke, "oct": preset_oct, "oct_large": preset_oct_large}[args.preset]()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.random_seed = args.seed

    tracin_dir = Path(cfg.output_dir) / "tracin"
    tracin_dir.mkdir(parents=True, exist_ok=True)
    attack_data_path = args.attack_data_path or str(tracin_dir / "attack_data.npz")
    np.random.seed(cfg.random_seed)

    # Step 1: 生成或加载 attack data (与 attack model 改动无关，可复用)
    if Path(attack_data_path).exists():
        ad = load_attack_data(attack_data_path)
    else:
        ad = _generate_attack_data(cfg)
        save_attack_data(attack_data_path, **ad)
    attack_train_x, attack_train_y = ad["attack_train_x"], ad["attack_train_y"]
    train_classes, train_shadow_ids = ad["train_classes"], ad["train_shadow_ids"]
    attack_test_x, attack_test_y, test_classes = ad["attack_test_x"], ad["attack_test_y"], ad["test_classes"]
    n_classes = int(ad["n_classes"])
    logger.info(f"Attack train={attack_train_x.shape}, test={attack_test_x.shape}, n_classes={n_classes}")

    # Step 2 & 3: per-class 训练 + TracIn
    from models import build_attack_model, predict_attack
    from attribution import (train_attack_with_checkpoints, tracin_cp, tracin_self_influence,
                             top_k_proponents, top_k_opponents, prediction_vector_stats,
                             aggregate_influence_by_group)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    task_classes = args.classes if args.classes is not None else list(range(n_classes))
    n_in = attack_train_x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for c in task_classes:
        logger.info(f"\n{'='*60}\n  TASK CLASS {c} ({CLASS_NAMES.get(c, c)})\n{'='*60}")
        tr_mask, te_mask = train_classes == c, test_classes == c
        c_tr_x, c_tr_y = attack_train_x[tr_mask], attack_train_y[tr_mask]
        c_te_x, c_te_y = attack_test_x[te_mask], attack_test_y[te_mask]
        if len(c_tr_x) < 20 or len(c_te_x) < 10:
            logger.warning(f"  Skipping class {c}: too few samples")
            continue
        c_shadow_ids = train_shadow_ids[tr_mask]

        def make_model():
            return build_attack_model("nn", n_in, cfg.attack_n_hidden).to(device)

        model, checkpoints = train_attack_with_checkpoints(
            make_model(), c_tr_x, c_tr_y, epochs=cfg.attack_epochs, lr=cfg.attack_lr,
            batch_size=cfg.attack_batch_size, l2_ratio=cfg.attack_l2,
            checkpoint_every=args.checkpoint_every, label=f"Attack-c{c}")

        c_pred = predict_attack(model, c_te_x)
        c_acc = accuracy_score(c_te_y, c_pred)
        c_prec = precision_score(c_te_y, c_pred, zero_division=0)
        c_rec = recall_score(c_te_y, c_pred, zero_division=0)
        logger.info(f"  Attack: acc={c_acc:.4f}, prec={c_prec:.4f}, rec={c_rec:.4f}")
        torch.save(checkpoints, str(tracin_dir / f"checkpoints_class{c}.pt"))

        # self-influence [Pruthi Sec 4.1]
        self_inf = tracin_self_influence(checkpoints, c_tr_x, c_tr_y, make_model, equal_weight=args.equal_weight)
        si_m, si_nm = self_inf[c_tr_y == 1], self_inf[c_tr_y == 0]
        logger.info(f"  Self-inf member mean={si_m.mean():.6f}, non-member mean={si_nm.mean():.6f}")

        # cross-influence [Pruthi Eq.1]
        scores = tracin_cp(checkpoints, c_tr_x, c_tr_y, c_te_x, c_te_y, make_model, equal_weight=args.equal_weight)
        logger.info(f"  |inf| by membership: {aggregate_influence_by_group(scores, c_tr_y, 'train')}")
        logger.info(f"  |inf| by shadow ID: {aggregate_influence_by_group(scores, c_shadow_ids, 'train')}")

        # 高影响样本的 prediction-vector 模式 (取首个被正确攻击的 member 测试点)
        correct_member = np.where((c_te_y == 1) & (c_pred == 1))[0]
        if len(correct_member) > 0:
            si = correct_member[0]
            pro = prediction_vector_stats(c_tr_x[top_k_proponents(scores, si, args.top_k)])
            opp = prediction_vector_stats(c_tr_x[top_k_opponents(scores, si, args.top_k)])
            logger.info(f"  test#{si}: proponents entropy={pro['entropy'].mean():.4f}, "
                        f"opponents entropy={opp['entropy'].mean():.4f}")

        with open(tracin_dir / f"tracin_class{c}.json", "w") as f:
            json.dump({
                "class": int(c), "class_name": CLASS_NAMES.get(c, str(c)),
                "n_train": int(len(c_tr_x)), "n_test": int(len(c_te_x)),
                "attack_acc": float(c_acc), "attack_prec": float(c_prec), "attack_rec": float(c_rec),
                "self_influence_member_mean": float(si_m.mean()),
                "self_influence_nonmember_mean": float(si_nm.mean()),
                "member_influence": aggregate_influence_by_group(scores, c_tr_y, "train"),
            }, f, indent=2, default=str)
        np.savez_compressed(str(tracin_dir / f"scores_class{c}.npz"), scores=scores,
                            self_influence=self_inf, train_y=c_tr_y, test_y=c_te_y,
                            train_x=c_tr_x, test_x=c_te_x)
        logger.info(f"  Results saved → {tracin_dir}")

    logger.info(f"\nPhase 2 TracIn complete. Outputs in {tracin_dir}")


if __name__ == "__main__":
    main()
