"""
配置类 + presets。

参数来源标注约定:
  [paper] = Shokri et al. 2017 (arXiv:1610.05820), Section VI-B / VI-C
  [repo]  = csong27/membership-inference (attack.py argparse / 函数默认值)
  [ours]  = 本项目自定 (OCT / TracIn 适配)

关键对齐说明:
  - target/shadow net 隐层 128 + Tanh, attack net 隐层 64 + ReLU → 均取自 [paper] VI-B/VI-C
    ([repo] 用 n_hidden=50, attack 默认 'softmax'，本项目按论文取值)
  - target_lr=1e-3, lr_decay=1e-7 → [paper] ("learning rate 0.001, decay 1e-07")
    ([repo] 用 0.01)
  - attack_lr=1e-2, attack_epochs=50, attack_l2=1e-6 → 与 [repo] 一致
  - OCT preset 改用 Adam (optimizer_type='adam', lr_decay=0) → [ours]，论文用 Torch7 SGD
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """全局配置；presets 在下方覆盖具体字段。"""
    dataset: str = "synthetic_purchase"

    n_features: int = 600          # [paper] purchase 600 binary features
    n_classes: int = 50
    n_total_samples: int = 40000

    # target_data_size = 训练集大小 = 测试集大小 (paper 语义)，每个模型消耗 2× 该值
    target_data_size: int = 5000   # [paper] purchase/texas/mnist = 10000
    n_shadow: int = 5              # [paper] 随数据集而变 (CIFAR 100 / purchase 20 / texas 10); [repo] CLI=10

    shadow_data_size: Optional[int] = None   # None → 等于 target_data_size ([paper] 默认)
    disjoint_shadow_models: bool = False     # [paper] shadow 训练集允许互相重叠
    class_balanced_split: bool = False       # [paper] 纯随机抽样; True 为本项目可选项 [ours]
    data_dir: str = "./data"

    # ---- Target / Shadow 模型 ----
    target_model_type: str = "nn"   # nn / softmax / cnn / paper_cnn
    target_n_hidden: int = 128      # [paper] VI-B purchase net 隐层 128
    target_epochs: int = 50         # [paper] purchase 200 / CIFAR 100; [repo] CLI=50
    target_lr: float = 1e-3         # [paper] 0.001 ([repo]=0.01)
    target_batch_size: int = 64     # [ours] (paper 未指定; repo CLI=100)
    target_l2: float = 1e-6         # [repo] CLI l2_ratio=1e-6

    optimizer_type: str = "sgd"     # [inferred] 论文未明写优化器，但 lr_decay=1e-7 是 Torch7 optim.sgd 的形式
    lr_decay: float = 1e-7          # [paper] learning rate decay 1e-07

    # shadow: None → 复用 target 设置 ([paper] "shadow 与 target 同样方式训练")
    shadow_model_type: Optional[str] = None
    shadow_n_hidden: Optional[int] = None
    shadow_epochs: Optional[int] = None
    shadow_lr: Optional[float] = None
    shadow_batch_size: Optional[int] = None
    shadow_l2: Optional[float] = None

    # ---- Attack 模型 ----
    attack_model_type: str = "nn"   # [paper] VI-C: 单隐层 + ReLU + softmax ([repo] CLI 默认 'softmax')
    attack_n_hidden: int = 64       # [paper] VI-C 隐层 64
    attack_epochs: int = 50         # [repo] 50
    attack_lr: float = 1e-2         # [repo] 0.01
    attack_batch_size: int = 64     # [ours] (repo CLI=100)
    attack_l2: float = 0.0          # 0 → 与 TracIn 纯 ∇ℓ 一致 (避免 Adam weight_decay 混入梯度) [ours]

    output_dir: str = "./results"
    save_models: bool = False
    random_seed: int = 42           # [ours] ([repo] 用 21312)
    verbose: bool = True

    # shadow getter: 未单独指定时回退到 target 设置
    def get_shadow_model_type(self) -> str:
        return self.shadow_model_type if self.shadow_model_type is not None else self.target_model_type

    def get_shadow_n_hidden(self) -> int:
        return self.shadow_n_hidden if self.shadow_n_hidden is not None else self.target_n_hidden

    def get_shadow_epochs(self) -> int:
        return self.shadow_epochs if self.shadow_epochs is not None else self.target_epochs

    def get_shadow_lr(self) -> float:
        return self.shadow_lr if self.shadow_lr is not None else self.target_lr

    def get_shadow_batch_size(self) -> int:
        return self.shadow_batch_size if self.shadow_batch_size is not None else self.target_batch_size

    def get_shadow_l2(self) -> float:
        return self.shadow_l2 if self.shadow_l2 is not None else self.target_l2

    def get_shadow_data_size(self) -> int:
        return self.shadow_data_size if self.shadow_data_size is not None else self.target_data_size


# =====================================================================
#  Presets
# =====================================================================

def preset_digits() -> Config:
    """sklearn digits 快速冒烟测试 [ours]。"""
    return Config(
        dataset="digits", target_data_size=250, n_shadow=3, shadow_data_size=250,
        class_balanced_split=True, target_model_type="nn", target_n_hidden=64,
        target_epochs=30, target_lr=1e-3, target_batch_size=64,
        optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=32, attack_epochs=30,
        attack_lr=1e-2, attack_batch_size=64, output_dir="./results/digits",
    )


def preset_purchase_50() -> Config:
    """合成 purchase-50 [ours, 仿 paper purchase 设置]。"""
    return Config(
        dataset="synthetic_purchase", n_features=600, n_classes=50, n_total_samples=40000,
        target_data_size=5000, n_shadow=5, class_balanced_split=False,
        target_model_type="nn", target_n_hidden=128, target_epochs=50, target_lr=1e-3,
        target_batch_size=64, optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=64, output_dir="./results/purchase_50",
    )


def preset_purchase_100() -> Config:
    """合成 purchase-100 [ours, 仿 paper purchase-100]。"""
    return Config(
        dataset="synthetic_purchase", n_features=600, n_classes=100, n_total_samples=60000,
        target_data_size=10000, n_shadow=10, class_balanced_split=False,
        target_model_type="nn", target_n_hidden=128, target_epochs=80, target_lr=1e-3,
        target_batch_size=128, optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=128, output_dir="./results/purchase_100",
    )


def preset_cifar10() -> Config:
    """CIFAR-10，paper-aligned: paper_cnn + SGD + lr_decay=1e-7 [paper VI-B; batch/卷积细节为 ours]。"""
    return Config(
        dataset="cifar10", target_data_size=10000, n_shadow=100, shadow_data_size=10000,
        disjoint_shadow_models=False, class_balanced_split=False,
        target_model_type="paper_cnn", target_n_hidden=128, target_epochs=100,
        target_lr=1e-3, target_batch_size=128, target_l2=0.0,
        optimizer_type="sgd", lr_decay=1e-7,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=256, attack_l2=1e-6, output_dir="./results/cifar10",
        random_seed=42, verbose=True,
    )


def preset_cifar10_smoke() -> Config:
    """CIFAR-10 冒烟测试 [ours]。"""
    return Config(
        dataset="cifar10", target_data_size=2000, n_shadow=2, shadow_data_size=2000,
        disjoint_shadow_models=False, class_balanced_split=False,
        target_model_type="paper_cnn", target_n_hidden=128, target_epochs=5,
        target_lr=1e-3, target_batch_size=128, target_l2=0.0,
        optimizer_type="sgd", lr_decay=1e-7,
        attack_model_type="nn", attack_n_hidden=32, attack_epochs=5, attack_lr=1e-3,
        attack_batch_size=128, attack_l2=1e-6, output_dir="./results/cifar10_smoke",
        random_seed=42, verbose=True,
    )


def preset_cifar10_delta() -> Config:
    """CIFAR-10 Delta 版 (20 shadow) [ours]。"""
    return Config(
        dataset="cifar10", target_data_size=10000, n_shadow=20, shadow_data_size=10000,
        disjoint_shadow_models=False, class_balanced_split=False,
        target_model_type="paper_cnn", target_n_hidden=128, target_epochs=100,
        target_lr=1e-3, target_batch_size=128, target_l2=0.0,
        optimizer_type="sgd", lr_decay=1e-7,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=256, attack_l2=1e-6, output_dir="./results/cifar10_delta",
        random_seed=42, verbose=True,
    )


def preset_purchase100() -> Config:
    """真实 Purchase-100 (privacytrustlab) [paper 数据 + repo 流程]。"""
    return Config(
        dataset="purchase100", n_features=600, n_classes=100, target_data_size=10000,
        n_shadow=20, shadow_data_size=10000, disjoint_shadow_models=False,
        class_balanced_split=False, target_model_type="nn", target_n_hidden=128,
        target_epochs=200, target_lr=1e-3, target_batch_size=128, target_l2=1e-6,
        optimizer_type="sgd", lr_decay=1e-7,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=128, attack_l2=1e-6, output_dir="./results/purchase100",
        data_dir="./data", random_seed=42, verbose=True,
    )


def preset_texas100() -> Config:
    """真实 Texas-100 [paper 数据集]。"""
    return Config(
        dataset="texas100", n_features=6170, n_classes=100, target_data_size=10000,
        n_shadow=10, shadow_data_size=10000, disjoint_shadow_models=False,
        class_balanced_split=False, target_model_type="nn", target_n_hidden=128,
        target_epochs=100, target_lr=1e-3, target_batch_size=128, target_l2=1e-6,
        optimizer_type="sgd", lr_decay=1e-7,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=128, attack_l2=1e-6, output_dir="./results/texas100",
        data_dir="./data", random_seed=42, verbose=True,
    )


# ---- OCT presets [ours]: Kermany2018, patient-level split, Adam ----

def preset_oct_smoke() -> Config:
    """OCT 冒烟测试: 小数据、2 shadow、确认 pipeline 跑通。"""
    return Config(
        dataset="oct", n_classes=4, n_total_samples=5000, target_data_size=400,
        n_shadow=2, shadow_data_size=400, class_balanced_split=False,
        target_model_type="cnn", target_n_hidden=128, target_epochs=5, target_lr=1e-3,
        target_batch_size=64, target_l2=1e-5, optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=32, attack_epochs=5, attack_lr=1e-2,
        attack_batch_size=64, attack_l2=0.0, output_dir="./results/oct_smoke",
        data_dir="./data", random_seed=42, verbose=True,
    )


def preset_oct() -> Config:
    """OCT 正式 baseline: patient-level split + SmallCNN，对应主实验。"""
    return Config(
        dataset="oct", n_classes=4, n_total_samples=40000, target_data_size=2000,
        n_shadow=5, shadow_data_size=2000, class_balanced_split=False,
        target_model_type="cnn", target_n_hidden=128, target_epochs=50, target_lr=1e-3,
        target_batch_size=128, target_l2=1e-5, optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=128, attack_l2=0.0, output_dir="./results/oct",  # l2=0 与 TracIn 梯度一致
        data_dir="./data", random_seed=42, verbose=True,
    )


def preset_oct_large() -> Config:
    """OCT 大规模: 更多 shadow / 更大数据，适合 Delta HPC。"""
    return Config(
        dataset="oct", n_classes=4, n_total_samples=80000, target_data_size=10000,
        n_shadow=10, shadow_data_size=10000, class_balanced_split=False,
        target_model_type="cnn", target_n_hidden=256, target_epochs=100, target_lr=1e-3,
        target_batch_size=128, target_l2=1e-5, optimizer_type="adam", lr_decay=0.0,
        attack_model_type="nn", attack_n_hidden=64, attack_epochs=50, attack_lr=1e-2,
        attack_batch_size=256, attack_l2=0.0, output_dir="./results/oct_large",
        data_dir="./data", random_seed=42, verbose=True,
    )
