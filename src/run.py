#!/usr/bin/env python3
"""
Phase 1 主入口: 解析 CLI / preset → 跑 MIAttackPipeline → 出 dashboard [Shokri 2017]。

用法:
  python run.py --preset oct
  python run.py --dataset oct --n_shadow 5 --target_epochs 50
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MI Attack (Shokri et al. 2017)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["digits", "synthetic_purchase", "cifar10", "purchase100", "texas100", "oct"])
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_shadow", type=int, default=None)
    parser.add_argument("--target_data_size", type=int, default=None)
    parser.add_argument("--shadow_data_size", type=int, default=None)
    parser.add_argument("--n_total_samples", type=int, default=None)
    parser.add_argument("--target_model_type", type=str, default=None,
                        choices=["nn", "softmax", "cnn", "paper_cnn"])
    parser.add_argument("--target_epochs", type=int, default=None)
    parser.add_argument("--attack_epochs", type=int, default=None)
    parser.add_argument("--attack_model_type", type=str, default=None, choices=["nn", "softmax"])
    parser.add_argument("--optimizer_type", type=str, default=None, choices=["sgd", "adam"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--preset", type=str, default=None,
                        choices=["digits", "purchase_50", "purchase_100", "cifar10", "cifar10_smoke",
                                 "cifar10_delta", "purchase100", "texas100", "oct_smoke", "oct", "oct_large"])
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    # preset → Config
    import config as C
    preset_map = {
        "digits": C.preset_digits, "purchase_50": C.preset_purchase_50, "purchase_100": C.preset_purchase_100,
        "cifar10": C.preset_cifar10, "cifar10_smoke": C.preset_cifar10_smoke, "cifar10_delta": C.preset_cifar10_delta,
        "purchase100": C.preset_purchase100, "texas100": C.preset_texas100,
        "oct_smoke": C.preset_oct_smoke, "oct": C.preset_oct, "oct_large": C.preset_oct_large,
    }
    cfg = preset_map[args.preset]() if args.preset in preset_map else C.Config()

    # CLI 覆盖 (仅非 None 的字段)
    overrides = {
        "dataset": args.dataset, "n_classes": args.n_classes, "n_shadow": args.n_shadow,
        "target_data_size": args.target_data_size, "shadow_data_size": args.shadow_data_size,
        "n_total_samples": args.n_total_samples, "target_model_type": args.target_model_type,
        "target_epochs": args.target_epochs, "attack_epochs": args.attack_epochs,
        "attack_model_type": args.attack_model_type, "optimizer_type": args.optimizer_type,
        "output_dir": args.output_dir, "random_seed": args.seed, "data_dir": args.data_dir,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    logger.info("Config:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    from attack import MIAttackPipeline
    results = MIAttackPipeline(cfg).run()

    if not args.no_viz:
        try:
            from visualize import plot_dashboard
            logger.info(f"Dashboard saved: {plot_dashboard(results, output_dir=cfg.output_dir)}")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    print("\n" + "=" * 50 + "\n  RESULTS SUMMARY\n" + "=" * 50)
    print(f"  Target train/test acc: {results['target_train_acc']:.4f} / {results['target_test_acc']:.4f}")
    print(f"  Overfitting gap:       {results['target_gap']:.4f}")
    print(f"  Attack acc/prec/rec:   {results['attack_accuracy']:.4f} / "
          f"{results['attack_precision']:.4f} / {results['attack_recall']:.4f}")
    print(f"  Advantage over 0.5:   +{results['attack_accuracy'] - 0.5:.4f}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    main()
