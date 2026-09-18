# Route A · 训到收敛再测 Level 3（会议 2026-09-18）

分支 `exp/pathway2-a-stationarity`，基于 `39a7066`（Route A 协议）。本文件在看到任何延长训练结果之前写成；第 4 节的平台规则和第 6 节的判读标准从此固定。

## 1 会议定了什么

| 优先级 | 内容 | 本分支对应 |
|---|---|---|
| 1 | 先只训练，不做别的。把 Stage 1 从 50 epoch 直接翻倍到 100，每个 epoch 都记录，画每个 seed（每种颜色）的曲线，看在哪个 epoch 之后变平 | Step 1：`20_A_convergence_curve.py` + `20_A_convergence_report.py` |
| 2 | 选一个变平的点 E*，在 E* 上重做 Level 3，看预测的参数变化和真实参数变化的相关性有没有变高 | Step 2：`submit_A_step2.sh E*` + `22_A_compare_E.py` |
| 3 | damping 调小。damping 引入 bias，理想情况下越小越接近真实 | Step 2 的 damping 网格加到 0.01（原来最小 0.03） |
| 暂缓 | 式 74（有限 Adam 轨迹导数，学长提到的 MAGIC 类方法）又慢又要存整条轨迹。代码推上 GitHub 给学长看，但不作为主线 | B1 代码已在 `exp/pathway2-b-trajectory@e8818e1`；B1.1 没有提交 |

学长的理由：误差会逐级累积，所以一步一步来。先让式 56 的前提（训练终点是驻点）尽量成立，再看 L3。

## 2 和 B1.1 handoff 的关系

1. handoff 第 2 节列的 B1.1 发布流程（manifest、bundle、installer、提交 Delta）按会议暂停。B1.1 的 10 个未提交文件不在 GitHub 上。
2. 冻结对象不变：原 50 epoch baseline、原 truth、原 split、原 panel、原 seeds 都只读。新的 E* 结果是新锚点，写在 `oct2-pathway2-runs/A/` 下，不覆盖也不冒充原 truth。
3. handoff §3.3 要求 Route A 续训时恢复每个 LOO endpoint 自己的 Adam moments。本方案用同一个 seed 从头训 E* 个 epoch，这个要求自动满足（见第 3 节），不需要额外的 replay 恢复。
4. handoff §3.3 还要求同时报告 ‖b_p‖、‖h‖ 和低梯度 cell 的个数。这些由 `22_A_compare_E.py` 输出。

## 3 从头训 E 个 epoch 就等于在原 50 epoch 上接着训

- `make_epoch_orders(train, E, seed+700000)` 按行依次从同一个生成器抽排列，所以前 50 行和原来的 50 epoch 顺序逐位相同（`test_epoch_orders_are_prefix_stable`）。
- 学习率是常数，没有 scheduler。dropout 的 RNG 流按步消耗。观察器在保存、恢复 RNG 的上下文里运行，只用 `autograd.grad`，不碰 `.grad` 和 Adam state。
- 运行时逐位核对：epoch 25、50 的参数、Adam state、CPU/CUDA RNG，以及 epoch 50 的最终模型和 post-train RNG 指纹。affected shadow 任何一项不等就直接失败。target 和 4 个 fixed shadow 不等时记录 `replay_exact=false` 后继续（加 `--strict_all` 可改为失败）。
- 本地测试逐位验证：延长到第 4 epoch 保存的参数和 Adam state，与直接训练 4 个 epoch 的结果完全相同，RNG 指纹也相同（`test_extension_equals_training_from_scratch_for_E_epochs`）。

所以 Step 2 里所有模型（target、shadow、每个 LOO）都直接用 `--shadow_epochs E*` 从头训练，得到的就是“原训练接着训到 E*”。

## 4 Step 1：训到 100 epoch，每个 epoch 记录

### 4.1 提交

在 Delta 的 Route A worktree（`~/oct2-exp-a`）里，更新到本分支之后：

```bash
cd ~/oct2-exp-a
export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4
export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A
export OCT_DATA_DIR=/u/yli103/oct2/data
bash scripts/cross_stage/submit_A_step1.sh
```

脚本依次做：检查 tracked 文件干净 → Route A 只读 preflight（冻结源码和 baseline 哈希）→ 新建 `convergence_E100_<时间>` 目录 → 提交 GPU 测试（`20_A_tests.slurm`，约 5 分钟）→ `afterok` 之后提交 10 个 array 任务：

| array | 模型 | seed | Hessian 谱 |
|---|---|---|---|
| 0–4 | affected shadow 3 | 42–46 | epoch 50, 60, …, 100 |
| 5 | target | 42007 | 无 |
| 6–9 | fixed shadow 0, 1, 2, 4 | 42100+sid | 无 |

每个任务 1 张 A40、4 小时上限。估计 affected shadow 约 1 小时（其中 Lanczos 约 40 分钟），其余约 10 分钟。

### 4.2 每个 epoch 记录什么（`curve.csv`）

| 列 | 含义 |
|---|---|
| `online_train_ce` | 训练时 minibatch CE 的 epoch 平均（dropout 开）。会上看的就是这条 |
| `eval_objective`、`eval_ce` | 全训练集，dropout 关，CE + wd/2‖θ‖²。式 56 的 H̄ 就是它的 Hessian |
| `eval_grad_norm` | 上面目标的全批梯度范数。式 56 需要它接近 0 |
| `relative_epoch_update` | ‖θ_e − θ_{e−1}‖ / ‖θ_{e−1}‖ |
| `heldout_ce`、`heldout_accuracy`、`generalization_gap_ce` | 非成员上的表现。过拟合加深会改变 attack 信号 |
| `interface_tv_prev_mean`、`interface_tv_from_T0_mean` | Stage 2 看到的概率向量（该模型的 member+nonmember 行）相对上个 epoch、相对 epoch 50 的平均总变差 |

另外 epoch 50、60、…、100 保存模型、Adam state 和 RNG（`checkpoints/`），affected shadow 在这些点上算 H̄ 的 Lanczos Ritz 端点（`spectrum.csv`）。`damping_floor = max(0, −λ_min)` 是让 H̄+γI 在 Krylov 估计上为正所需的最小 γ。epoch 50 的谱会和 v1.1 的 `spectrum_seed*.json` 对照（同一 probe seed 1701/1702、50 步）。

### 4.3 汇总

所有任务 `COMPLETED 0:0` 后：

```bash
module load pytorch-conda/2.8
python3 scripts/cross_stage/20_A_convergence_report.py --root "$HOME/oct2-pathway2-runs/A/convergence_E100_<时间>"
```

输出在 `<root>/report/`：`curves_affected_shadow.png`（每个 seed 一种颜色，6 个面板）、`curves_target_and_fixed_shadows.png`、`spectrum.png`、`window_table.csv`、`plateau_report.json`、`REPORT.md`。可以直接贴到 Slack。

### 4.4 平台规则（预先固定）

- 10-epoch 窗口 (k−10, k]，取 `online_train_ce` 的窗口均值 m_k 和标准误 se_k。
- 相邻窗口 (k, k+10) 算“平”：|m_k − m_{k+10}| ≤ max(0.10·m_k, 0.002, 2·√(se_k² + se_{k+10}²))。也就是变化小于 10%，或者和 minibatch 噪声分不开。
- 某个 seed 从 k 开始平：k ≥ 50，之后每一对都平，且至少有两对（所以 k ≤ 80）。
- 建议 E* = 5 个 affected seed 的起点取最大（每种颜色都平）。有任何一个 seed 在 100 以内没有平，报告写“未建立”，这时先把图给学长看，再决定要不要继续延长。
- 同样的规则也用在 eval objective 上，只作参考。

脚本给出的 E* 是建议。最后用哪一个点，和学长一起看图决定（会上说先只取一个点）。

### 4.5 判读提醒

loss 平了不代表到了驻点。v1.1 在 epoch 50 看到 `eval_grad_norm` 在 0.3 到 2 之间来回跳，每个 epoch 参数还在走 4% 左右。如果延长后 loss 平了、但梯度范数没有下降、λ_min 仍明显为负，要把这一点和曲线一起报告：常数学习率的 Adam 可能停在噪声球里。这种情况下 Step 2 仍然可以按会议做，只是结果不能说成“前提已满足”。备选的学习率衰减方案已在 `plan.json` 的 A-decay 臂登记，不在本轮默认运行。

## 5 Step 2：在一个 E* 上重做 Level 3，并把 damping 调小

### 5.1 提交（E* 从 Step 1 报告里定）

```bash
cd ~/oct2-exp-a
bash scripts/cross_stage/submit_A_step2.sh 70     # 70 只是示例
```

只调用已经验证过的入口，参数和正式实验一致，只有 `--shadow_epochs E*` 不同：

| 作业 | 内容 | 依赖 |
|---|---|---|
| `oct_A_truth_full` | 10 → 05：target、shadow、5 个 baseline、no-op、attack gate、40 个 α=1 LOO，全部训 E* epoch；然后 09 | 无 |
| `oct_A_truth_dose` | 同上，α=0.1 | 无，和上一个并行 |
| `oct_A_score` | 10 → 07：L1/L2/L3 score ladder，γ_A=0.2、γ_S=1，附加 0.8、1.2（和 21950198 相同） | truth_full |
| `oct_A_diagE` | 12 preflight（尺度和 α 线性）+ 12 damping（Lanczos-MINRES，γ ∈ {0.01, 0.03, 0.1, 0.3, 1, 2}） | 前三个都成功 |

冻结 panel 从原树复制，并用 `cmp` 确认字节相同。输出目录 `oct2-pathway2-runs/A/l3_at_E<E*>/` 已存在时拒绝提交。

参考耗时（50 epoch 时实测）：truth 54 分钟，score 114–158 分钟，damping 约 5.6 小时。truth 大致随 epoch 数线性增加。

### 5.2 读结果

四个作业都 `COMPLETED 0:0` 后：

```bash
module load pytorch-conda/2.8
python3 scripts/cross_stage/22_A_compare_E.py --new_root "$HOME/oct2-pathway2-runs/A/l3_at_E70" \
    --baseline_root "$HOME/oct2-calibration-v4"
```

`compare_vs_E50/COMPARE.md` 把 50 epoch 和 E* 放在一起比较：

1. score ladder 的共同行：Spearman、MAE、符号一致率。50 epoch 时 L3 linear 的 Spearman 是 0.049，MAE 0.00473，和零预测 0.00474 基本一样。
2. 预测 Δθ 和真实 Δθ：cosine（50 epoch 时中位数 0.0076）、范数比（8.8e−5）、‖Δθ‖/‖θ0‖（0.70–1.10）、‖Δθ‖ 相对 seed 间距离（0.49–0.79）、R_α（中位数 9.06，1 表示线性）。
3. damping 扫描：每个 γ 的合格 cell 数、cosine、h 投影 Spearman（50 epoch 时 γ=0.03 为 0/40 合格）。
4. 信号大小：‖b_p‖、‖h‖、‖b_p‖<1e−6 的 cell 数、J10−J00 的均值、λ_min、基线梯度范数。

## 6 判读标准（预先写下）

| 看到什么 | 说明 | 下一步 |
|---|---|---|
| L3 Spearman 和 Δθ cosine 明显上升，MAE 低于零预测，R_α 向 1 靠近，同时 ‖b_p‖ 和 J10−J00 没有塌缩 | 收敛确实是 L3 失败的主要原因之一 | 在 E* 上把 damping 调到合格的最小值；再按学长的意思考虑要不要试第二个 E* |
| 小 γ 的合格率上升（λ_min 向 0 靠近），但 cosine 和 Spearman 基本不变 | 前提条件变好了，但预测没有变好 | 和学长讨论：问题不只是收敛，被估量本身可能不对（见下） |
| 基本没有变化，‖Δθ‖ 仍和 seed 间距离同量级，R_α 仍约 9 | 从头重训的 LOO 仍由轨迹分叉主导 | 会上说的“比较麻烦”的情况。两条路：式 74 轨迹路线（B），或者换被估量（下一行） |
| 信号塌缩（‖b_p‖ 或 J10−J00 接近噪声） | 不能说公式变准了，只能说信号分辨不出来 | 如实报告 |

关于“被估量”：文献里，深度网络上的 influence function 近似的不是“从头重训的 LOO”，而是从训练好的 θ̂ 出发、带近端项的局部再优化（Bae 等 2022，proximal Bregman response function）。Koh & Liang 2017 在非凸模型上的验证也是从 θ̂ warm-start 重训。如果第三行成立，可以带这个问题去和学长讨论。本分支不预先实现它。

## 7 本地已完成的验证（CPU，PyTorch 2.8.0）

| 测试 | 结果 |
|---|---|
| `tests/test_A_convergence.py`（7 个） | 通过：顺序前缀稳定；延长结果和从头训练逐位相同（参数、Adam、RNG）；affected 不一致时失败；fixed 不一致时记录；输出目录不能和原结果重叠、不能复用；平台规则包括噪声项；报告拒绝不完整的面板 |
| `tests/run_A_step2_chain.py` | 通过：在合成数据上用 `10_run_calibration.py --dry_run` 打印出的命令依次跑 08 → 05（full、dose01）→ 07 → 12 preflight/damping → 22，确认参数、目录名和文件格式对得上 |
| 原有回归（test_stage1_v11 10 个、test_stage1_diagnostics 9 个、test_formal_analysis 7 个、score ladder math） | 全部通过 |
| 受保护源文件 SHA256（frozen_baseline.json 中 8 个） | 未改动；contract SHA `c9055088…ae67` 不变 |

真实 A40 上的逐位重放由 GPU 测试和 array 任务自己在运行时检查。

## 8 文件

```
scripts/cross_stage/20_A_convergence_curve.py     Step 1 runner（每个模型一个任务）
scripts/cross_stage/20_A_convergence_report.py    Step 1 汇总：曲线、窗口表、平台规则、谱
scripts/cross_stage/20_A_convergence.slurm        Step 1 array
scripts/cross_stage/20_A_tests.slurm              GPU 测试
scripts/cross_stage/submit_A_step1.sh             Step 1 提交
scripts/cross_stage/submit_A_step2.sh             Step 2 提交（参数 E*）
scripts/cross_stage/21_A_diag_at_E.slurm          Step 2：12 preflight + damping
scripts/cross_stage/22_A_compare_E.py             Step 2 读数：50 vs E*
scripts/cross_stage/tests/test_A_convergence.py
scripts/cross_stage/tests/run_A_step2_chain.py
experiments/pathway2/plan_A_meeting_20260918.json
```

参考：MAGIC（Ilyas & Engstrom 2025，arXiv 2504.16430），式 74 类精确轨迹归因；Bae 等，If Influence Functions are the Answer, Then What is the Question?（NeurIPS 2022，arXiv 2209.05364）。
