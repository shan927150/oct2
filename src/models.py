"""
模型定义 + 训练工具。

来源:
  - NNModel (隐层 + Tanh + softmax): [paper] VI-B / [repo] classifier.py 的 'nn'
  - AttackModel (隐层 64 + ReLU + 2 logits): [paper] VI-C
    ([repo] CLI 默认 attack='softmax'，本项目按论文用单隐层 ReLU 网)
  - 损失统一 CrossEntropyLoss + 2 logits: 对齐 [paper] "binary classifier, 2 outputs in/out"
    (由 BCEWithLogitsLoss 改来，避免其 clamp_min_ 不可微点影响 TracIn 梯度)
  - SmallCNN / PaperCNN: [ours] OCT 适配 (AdaptiveAvgPool 支持任意分辨率)
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
#  Target / Shadow 模型
# =====================================================================

class SoftmaxModel(nn.Module):
    """单层 softmax 回归 [repo] 'softmax'。"""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


class NNModel(nn.Module):
    """单隐层 + Tanh 全连接网 [paper VI-B / repo 'nn']。"""
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Adaptive average bins without CUDA adaptive_avg_pool2d_backward atomics.

    Divisible downsampling uses nonoverlapping AvgPool2d (the OCT 128x128
    path). Other sizes use the same floor/ceil bins as AdaptiveAvgPool2d,
    expressed as slice/mean/stack operations. Both paths support double
    backward and forward-mode AD. This layer has no state_dict entries.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = ((output_size, output_size) if isinstance(output_size, int)
                            else tuple(output_size))
        if len(self.output_size) != 2 or any(v < 1 for v in self.output_size):
            raise ValueError("output_size must contain two positive dimensions")

    def forward(self, x):
        height, width = x.shape[-2:]
        out_h, out_w = self.output_size
        if height % out_h == 0 and width % out_w == 0:
            kernel = (height // out_h, width // out_w)
            return nn.functional.avg_pool2d(x, kernel_size=kernel, stride=kernel)
        rows = []
        for i in range(out_h):
            start_h = i * height // out_h
            end_h = ((i + 1) * height + out_h - 1) // out_h
            cells = []
            for j in range(out_w):
                start_w = j * width // out_w
                end_w = ((j + 1) * width + out_w - 1) // out_w
                cells.append(x[..., start_h:end_h, start_w:end_w].mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)


class SmallCNN(nn.Module):
    """通用小型 CNN，AdaptiveAvgPool2d 支持任意分辨率 (CIFAR / OCT) [ours]。"""
    def __init__(self, in_channels, n_hidden, n_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = DeterministicAdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, n_hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class PaperCNN(nn.Module):
    """paper-style CIFAR CNN [paper VI-B]: 2×(conv+pool) + FC128 + Tanh + softmax。
    论文只给了这一结构；conv 通道数/kernel size 及 AdaptiveAvgPool 为本项目选择 [ours]。"""
    def __init__(self, in_channels, n_hidden, n_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# =====================================================================
#  Attack 模型
# =====================================================================

class AttackModel(nn.Module):
    """单隐层 + ReLU + 2 logits [out, in] [paper VI-C]。"""
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class AttackSoftmax(nn.Module):
    """单层 softmax attack 模型 [repo 'softmax']。"""
    def __init__(self, n_in):
        super().__init__()
        self.fc = nn.Linear(n_in, 2)

    def forward(self, x):
        return self.fc(x)


# =====================================================================
#  Builders
# =====================================================================

def build_model(model_type, n_in, n_hidden, n_out):
    """构建 target/shadow 模型。"""
    if model_type == "nn":
        return NNModel(n_in, n_hidden, n_out).to(DEVICE)
    if model_type == "softmax":
        return SoftmaxModel(n_in, n_out).to(DEVICE)
    if model_type == "cnn":
        return SmallCNN(n_in, n_hidden, n_out).to(DEVICE)
    if model_type == "paper_cnn":
        return PaperCNN(n_in, n_hidden, n_out).to(DEVICE)
    raise ValueError(f"Unknown model type: {model_type}")


def build_attack_model(model_type, n_in, n_hidden):
    """构建 attack 模型。"""
    if model_type == "nn":
        return AttackModel(n_in, n_hidden).to(DEVICE)
    if model_type == "softmax":
        return AttackSoftmax(n_in).to(DEVICE)
    raise ValueError(f"Unknown attack model type: {model_type}")


# =====================================================================
#  训练工具
# =====================================================================

def _to_tensor_x(X):
    return torch.tensor(X, dtype=torch.float32)


def _to_tensor_y(y):
    return torch.tensor(y, dtype=torch.long)


def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(_to_tensor_x(X), _to_tensor_y(y))
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)


def _predict_logits(model, X, batch_size=512):
    """批量前向，返回 logits (CPU)。"""
    model.eval()
    loader = DataLoader(_to_tensor_x(X), batch_size=min(batch_size, len(X)), shuffle=False)
    outputs = []
    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            outputs.append(model(xb.to(DEVICE)).detach().cpu())
    return torch.cat(outputs, dim=0)


def _predict_probs(model, X, batch_size=512):
    """softmax 概率 (即 paper 的 prediction vector)。"""
    return torch.softmax(_predict_logits(model, X, batch_size), dim=1).numpy()


def train_model(model, dataset, epochs, lr, batch_size, l2_ratio, verbose=True, label="",
                optimizer_type="adam", lr_decay=0.0):
    """训练分类模型 (target/shadow)。
    optimizer_type: 'sgd' (paper Torch7) 或 'adam'。
    lr_decay: Torch7 风格 lr_t = lr/(1+decay·t) [paper 用 1e-7，影响极小]。
    """
    train_X, train_y, test_X, test_y = dataset
    train_loader = _make_loader(train_X, train_y, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_ratio)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_ratio)

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if optimizer_type == "sgd" and lr_decay > 0:   # Torch7 风格 lr 衰减
                cur = lr / (1.0 + lr_decay * global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
        if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            logger.info(f"[{label}] epoch={epoch+1}/{epochs}, loss={total_loss:.4f}")

    train_acc = accuracy_score(train_y, _predict_probs(model, train_X).argmax(axis=1))
    test_acc = accuracy_score(test_y, _predict_probs(model, test_X).argmax(axis=1))
    logger.info(f"[{label}] train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, gap={train_acc - test_acc:.4f}")
    return model, train_acc, test_acc


def get_predictions(model, X):
    """返回 prediction vector (softmax 概率)，即 attack 模型的输入特征。"""
    return _predict_probs(model, X)


def train_attack_model_binary(model, train_X, train_y, epochs, lr, batch_size, l2_ratio, label=""):
    """训练单个 per-class 二分类 attack 模型 (CrossEntropyLoss + Adam)。"""
    X_tensor = torch.tensor(train_X, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                        batch_size=min(batch_size, len(X_tensor)), shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_ratio)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    logger.info(f"[{label}] attack training finished")
    model.eval()
    return model


def predict_attack(model, X):
    """attack 模型预测 in/out (argmax over 2 logits)。"""
    model.eval()
    loader = DataLoader(torch.tensor(X, dtype=torch.float32),
                        batch_size=min(512, len(X)), shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).numpy()
