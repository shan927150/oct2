"""
MI Attack 三阶段 pipeline (Shokri et al. 2017)。

  Stage 1 [paper Fig 1]: 训练 target → 在 target train/test 上收集 prediction vector → attack TEST 集
  Stage 2 [paper Fig 2/3]: 训练 shadow → 收集 prediction vector (in→1, out→0) → attack TRAIN 集
  Stage 3 [paper V-D]:   每个类别一个二分类 attack 模型 (collection of models, one per class)

来源: 结构对齐 [paper] V & Fig 1-3 / 流程参考 [repo] csong27 attack.py。
"""
import logging
import os
import time
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from config import Config
from data import load_dataset, split_data
from models import (
    build_model, build_attack_model, train_model, get_predictions,
    train_attack_model_binary, predict_attack,
)

logger = logging.getLogger(__name__)


class MIAttackPipeline:
    """端到端 MI attack 流程，结果存于 self.results。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.results: Dict = {}
        np.random.seed(cfg.random_seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def run(self) -> Dict:
        """执行 Stage 1→2→3 并返回结果字典。"""
        t_start = time.time()
        logger.info("=" * 70)
        logger.info("MEMBERSHIP INFERENCE ATTACK PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Dataset: {self.cfg.dataset} | shadow={self.cfg.n_shadow} | "
                    f"target_model={self.cfg.target_model_type} | attack_model={self.cfg.attack_model_type}")

        X, y, groups = load_dataset(self.cfg)
        target_data, shadow_data_list = split_data(X, y, self.cfg, groups=groups)
        n_classes = len(np.unique(y))
        n_in = self._infer_model_input_dim(X)

        self.results.update({"dataset": self.cfg.dataset, "n_classes": n_classes,
                             "n_total_samples": len(y)})
        logger.info(f"Data: X={X.shape}, #classes={n_classes}, model_input={n_in}")

        logger.info("\n" + "=" * 20 + " STAGE 1: TARGET " + "=" * 20)
        attack_test_x, attack_test_y, test_classes = self._train_target(target_data, n_in, n_classes)

        logger.info("\n" + "=" * 20 + " STAGE 2: SHADOWS " + "=" * 20)
        attack_train_x, attack_train_y, train_classes = self._train_shadows(shadow_data_list, n_in, n_classes)

        logger.info("\n" + "=" * 20 + " STAGE 3: ATTACK " + "=" * 20)
        self._train_and_evaluate_attack(
            attack_train_x, attack_train_y, train_classes,
            attack_test_x, attack_test_y, test_classes, n_classes,
        )

        self.results["total_time_sec"] = time.time() - t_start
        logger.info(f"\nPipeline finished in {self.results['total_time_sec']:.1f}s")
        return self.results

    def _infer_model_input_dim(self, X) -> int:
        """CNN 取通道数 (X.ndim==4)，全连接取特征维 (X.ndim==2)。"""
        if self.cfg.target_model_type in ("cnn", "paper_cnn"):
            if X.ndim != 4:
                raise ValueError(f"{self.cfg.target_model_type} expects (N,C,H,W), got ndim={X.ndim}")
            return X.shape[1]
        if X.ndim != 2:
            raise ValueError(f"{self.cfg.target_model_type} expects (N,D), got ndim={X.ndim}")
        return X.shape[1]

    def _build_target_or_shadow_model(self, model_type, n_in, n_hidden, n_classes):
        return build_model(model_type, n_in, n_hidden, n_classes)

    def _train_target(self, target_data, n_in, n_classes):
        """Stage 1: 训练 target，train→member(1)/test→nonmember(0) 构成 attack TEST 集 [paper Fig 1]。"""
        train_X, train_y, test_X, test_y = target_data
        model = self._build_target_or_shadow_model(
            self.cfg.target_model_type, n_in, self.cfg.target_n_hidden, n_classes)
        model, train_acc, test_acc = train_model(
            model, target_data, self.cfg.target_epochs, self.cfg.target_lr,
            self.cfg.target_batch_size, self.cfg.target_l2, verbose=self.cfg.verbose,
            label="Target", optimizer_type=self.cfg.optimizer_type, lr_decay=self.cfg.lr_decay)

        self.results.update({"target_train_acc": float(train_acc), "target_test_acc": float(test_acc),
                             "target_gap": float(train_acc - test_acc)})

        member_preds = get_predictions(model, train_X)
        nonmember_preds = get_predictions(model, test_X)
        attack_x = np.concatenate([member_preds, nonmember_preds], axis=0)
        attack_y = np.concatenate([np.ones(len(member_preds), dtype=np.int64),
                                   np.zeros(len(nonmember_preds), dtype=np.int64)])
        classes = np.concatenate([train_y, test_y], axis=0)
        logger.info(f"Target gap={train_acc - test_acc:.4f}; attack TEST X={attack_x.shape}")
        return attack_x, attack_y, classes

    def _train_shadows(self, shadow_data_list, n_in, n_classes):
        """Stage 2: 训练每个 shadow，in→1/out→0 汇成 attack TRAIN 集 [paper Fig 3]。"""
        all_x, all_y, all_cls, shadow_accs = [], [], [], []
        for i, shadow_data in enumerate(shadow_data_list):
            logger.info(f"--- Shadow {i+1}/{len(shadow_data_list)} ---")
            train_X, train_y, test_X, test_y = shadow_data
            model = self._build_target_or_shadow_model(
                self.cfg.get_shadow_model_type(), n_in, self.cfg.get_shadow_n_hidden(), n_classes)
            model, train_acc, test_acc = train_model(
                model, shadow_data, self.cfg.get_shadow_epochs(), self.cfg.get_shadow_lr(),
                self.cfg.get_shadow_batch_size(), self.cfg.get_shadow_l2(), verbose=False,
                label=f"Shadow-{i}", optimizer_type=self.cfg.optimizer_type, lr_decay=self.cfg.lr_decay)
            shadow_accs.append((train_acc, test_acc))

            all_x += [get_predictions(model, train_X), get_predictions(model, test_X)]
            all_y += [np.ones(len(train_X), dtype=np.int64), np.zeros(len(test_X), dtype=np.int64)]
            all_cls += [train_y, test_y]

        attack_x = np.concatenate(all_x, axis=0)
        attack_y = np.concatenate(all_y, axis=0)
        classes = np.concatenate(all_cls, axis=0)
        self.results.update({
            "shadow_train_acc": float(np.mean([a for a, _ in shadow_accs])),
            "shadow_test_acc": float(np.mean([b for _, b in shadow_accs])),
            "shadow_accs": [(float(a), float(b)) for a, b in shadow_accs],
        })
        logger.info(f"Attack TRAIN set from shadows: X={attack_x.shape}")
        return attack_x, attack_y, classes

    def _train_and_evaluate_attack(self, train_x, train_y, train_classes,
                                   test_x, test_y, test_classes, n_classes):
        """Stage 3: 每类训练一个二分类 attack 模型并评估 (precision/recall, baseline=0.5) [paper V-D, VI-D]。"""
        n_in = train_x.shape[1]
        per_class_results = {}
        all_true, all_pred = [], []

        for c in np.unique(train_classes):
            c = int(c)
            c_train_mask = train_classes == c
            c_test_mask = test_classes == c
            if c_train_mask.sum() < 10 or c_test_mask.sum() < 10:   # [ours] 小样本保护，跳过该类
                logger.info(f"Skip class {c}: too few samples")
                continue

            attack_model = build_attack_model(self.cfg.attack_model_type, n_in, self.cfg.attack_n_hidden)
            attack_model = train_attack_model_binary(
                attack_model, train_x[c_train_mask], train_y[c_train_mask],
                self.cfg.attack_epochs, self.cfg.attack_lr, self.cfg.attack_batch_size,
                self.cfg.attack_l2, label=f"Attack-c{c}")

            c_test_y = test_y[c_test_mask]
            c_pred = predict_attack(attack_model, test_x[c_test_mask])
            per_class_results[c] = {
                "accuracy": float(accuracy_score(c_test_y, c_pred)),
                "precision": float(precision_score(c_test_y, c_pred, zero_division=0)),
                "recall": float(recall_score(c_test_y, c_pred, zero_division=0)),
                "n_test": int(len(c_test_y)),
            }
            all_true.append(c_test_y)
            all_pred.append(c_pred)

        if not all_true:
            raise RuntimeError("No class had enough samples for attack.")
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        self.results.update({
            "attack_accuracy": float(accuracy_score(all_true, all_pred)),
            "attack_precision": float(precision_score(all_true, all_pred, zero_division=0)),
            "attack_recall": float(recall_score(all_true, all_pred, zero_division=0)),
            "per_class_results": per_class_results,
            "all_true": all_true, "all_pred": all_pred,
        })

        logger.info("\n" + "=" * 30 + " FINAL ATTACK EVAL " + "=" * 30)
        logger.info(f"acc={self.results['attack_accuracy']:.4f}, prec={self.results['attack_precision']:.4f}, "
                    f"rec={self.results['attack_recall']:.4f} (baseline=0.5, target_gap={self.results['target_gap']:.4f})")
        logger.info("\n" + classification_report(all_true, all_pred,
                    target_names=["Non-member", "Member"], zero_division=0))
