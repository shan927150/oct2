# 独立 Stage 1 诊断 v1

本版以 GitHub `shan927150/oct2` 的 `cross-stage-calibration-v4` 分支、提交 `2870ad47219722de5bcf6274b043ea0f4f751db6` 为实现基准。准备日期为 2026-09-16。诊断尚未在真实 OCT checkpoint 上运行，不能将下面的设计写成新的实验结果。

## 1. 当前问题与判断边界

full truth 和 dose01 truth 已经完成。21950198 的 full score 在 γA=0.2、γS=1.0 下有 40/40 合格行，但 Stage 1 的参数变化预测仍然很弱。22119842 的 score 可以继续运行。

本轮回答两个问题。

1. Stage 1 的 damping 是否明显改变了实际使用的逆 Hessian 方向。
2. 原来的 50 epoch Adam 训练是否已趋于稳定，是否接近当前 Hessian 对应目标的驻点。

求解收敛只支持“指定 damped 方程得到可靠数值解”，不能证明 damping 没有偏差，也不能证明有限步 Adam 等价于驻点敏感度。当前证据不足以说“所有问题只剩静态公式本身”。

## 2. 已确认的代码实现

`07_cross_stage_score_ladder.py` 的 `make_hvp` 用两次自动微分计算 Hessian-vector product。`conjugate_gradient` 使用没有预条件器的 CG，并在结束时重新计算真实残差，遇到非正曲率会标记失败。没有构造或求逆 503044 × 503044 的稠密矩阵。

令 θ₀ 是原 baseline 参数，N 是 affected shadow 的训练图像数。当前正式设置中 N=2045。

\[
L_0(\theta)=\frac1N\sum_{i=1}^N\ell_i^{\mathrm{eval}}(\theta)
  +\frac{\lambda_{\mathrm{wd}}}{2}\|\theta\|^2,
\qquad \bar H=\nabla^2 L_0(\theta_0),\quad \lambda_{\mathrm{wd}}=10^{-5}.
\]

这里的 eval 表示关闭 dropout。这与训练时包含 dropout 的随机目标有区别。新脚本同时观察两种目标，不将两者混为一谈。

病人 p 的图像集合为 Iₚ。把这些图像的训练权重由 1 减至 1−α，保持原 mini-batch 和原分母。用于诊断的静态近似是

\[
b_p=\frac1N\sum_{i\in I_p}\nabla\ell_i^{\mathrm{eval}}(\theta_0),
\qquad (\bar H+\gamma I)x_p=b_p,
\qquad \widehat{\Delta\theta}_p(\alpha)=\alpha x_p.
\]

正号对应下调训练权重。α=1 是 full，α=0.1 是 dose01。真实变化使用相同 seed 的 checkpoint 差值

\[
\Delta\theta_p^{\mathrm{true}}(\alpha)
=\theta_{p,\alpha}^{\mathrm{retrain}}-\theta_0.
\]

这是沿用现有 score 的“均匀图像平均 CE”局部近似。训练中的逐 batch Adam、dropout 和最后一个 batch 的归一化不会因此变成确定性的全量优化。这里检验的是该近似与原训练程序的差距。

## 3. 固定设计

| 项目 | 设置与理由 |
|---|---|
| 受影响模型 | shadow 3，与原 truth 配对 |
| Stage 1 seeds | 42、43、44、45、46，不新增 seeds |
| 病人 | 原冻结面板 8 人，DME 与 DRUSEN 各 4 人，不按新结果剔除病人 |
| 真实干预 | full α=1 与 dose01 α=0.1，复用已完成 checkpoint |
| 数据和顺序 | 从原 split、patient panel、epoch orders 读取并校验 |
| damping 候选 | γS ∈ {0.7, 1.0, 2.0}，在查看新方向指标前固定 |
| damping 的依据 | 已有谱估计最负边缘约为 −0.621，0.7 探查较小正移位，1.0 是现有基线，2.0 扩大正则化对照 |
| CG | 最多 150 次，停止容差 1e−4，真实残差合格阈值 1e−3 |
| Lanczos | 对未加额外 γ 的 H̄ 做 2 个独立起点，各 50 步，完整重正交化 |
| 计算精度 | 与当前模型一致，默认 float32 |
| 输出 | 每次调用创建全新的 `results/stage1_diagnostics_v1/<mode>_<jobid>` |

0.7 是诊断候选，不保证所有 seed 都合格。有限次 Lanczos 的正最小 Ritz 值不是全局正定性证明。脚本保存极端 Ritz 值、显式 Ritz 残差、半程与全程估计及三对角投影。投影特征值不是全 Hessian 的谱分布。

## 4. Damping 诊断

每个 seed 先估计 H̄ 的谱边缘，所有病人共用该 baseline 算子。每个病人、每个 γ 求解一次，然后将解按 α 缩放，与 full 和 dose01 的真实变化分别比较。默认共 120 次 CG 求解，对应 240 个比较行。

输出指标如下。

| 指标 | 含义与使用方式 |
|---|---|
| λmin、λmax 估计 | 同时报 H̄ 和减去 weight decay 后的 CE Hessian 极端值 |
| γ / λmax | damping 相对最大正曲率的大小，只提供部分信息 |
| γ / abs(λmin) | 当检测到负边缘时，显示移位相对负曲率的大小 |
| ‖(H̄+γI)x−b‖ / ‖b‖ | 实际方程残差，用于数值合格判定 |
| cos(αx, Δθtrue) | 参数变化方向是否一致 |
| ‖αx‖ / ‖Δθtrue‖ | 预测的变化幅度是否合适 |
| ‖αx−Δθtrue‖ / ‖Δθtrue‖ | 相对向量误差 |
| cos(x,b) 与 ‖x−b/γ‖ / ‖x‖ | 检查解是否接近只由梯度决定的各向同性近似 |
| γ‖x‖ / ‖b‖ 与 ‖H̄x‖ / ‖b‖ | 比较方程中 damping 项与曲率项的幅度，二者不是可相加的百分比 |
| cos(0.1Δθfull, Δθdose01) 及其幅度比 | 不依赖 Hessian 预测，直接检查真实响应是否随 α 近似线性 |

为什么不能只看 γ/λmax：在特征方向 qⱼ 上，解的分量为 (qⱼᵀb)/(λⱼ+γ)。即使 γ 远小于 λmax，它仍可能明显改变 b 所在的低曲率方向。

合格行要求两个 Lanczos 起点的端点相对残差不超过 0.005，移位后的最小 Ritz 值超过对应绝对残差，且 CG 没有非正曲率或非有限值，实际相对残差不超过 1e−3。这个组合是数值诊断，不是严格正定证书。

不合格行的原始几何数值只保存在带明确标签的 JSON 中。正式 CSV 的这些字段为空。汇总和图形仅比较在所有候选 γ 上都合格的共同单元。若没有共同单元，汇总为空，不能挑选某个 γ 的好看结果。

结果是探索性诊断。不按 cosine 或 LOO 相关性挑“最佳 γ”。8 个病人和重复 seeds 不构成 40 个独立病人样本。

## 5. Convergence 诊断

原训练程序没有保存完整的逐 epoch loss。新脚本从相同初始化和相同 epoch orders 重放 baseline 50 epochs，每个 epoch 插入保持 RNG 不变的观测。

每个 epoch 记录在线训练 CE、固定全训练集的 eval CE、CE+L2、参数更新范数及相对更新范数。在线 CE 使用实际训练 batch 的损失，它是在一个 epoch 内不同参数位置观察到的值。固定 eval 目标则是在 epoch 结束后的同一组参数上计算，二者需要分别解释。

每 5 个 epochs 计算完整训练集的 eval CE+L2 梯度范数。epoch 0、25、50 另外做 4 次 dropout Monte Carlo 观测，记录训练目标 CE 均值、标准误、平均梯度范数和梯度随机波动。4 次 MC 是初步诊断，均值梯度偏大需要结合 MC 波动解释。

在 epoch 25、50 与原 checkpoint 逐位比较参数、Adam 状态、CPU RNG 和 CUDA RNG。最后还比较最终 baseline 参数与原 post-train RNG 指纹。任何不一致都会使任务失败，不将其称作原实验训练曲线。

收敛解释需要结合 loss 的后期变化、参数更新与梯度尺度。没有设立通用的“梯度范数小于某数即收敛”判据。曲线平稳也不自动证明达到驻点或证明 influence 假设成立。本轮不自行延长训练，不修改原 truth。

## 6. 新文件与结果读取

本补丁仅新增以下 5 个仓库文件。

1. `scripts/cross_stage/11_stage1_diagnostics.py`
2. `scripts/cross_stage/stage1_diagnostic_core.py`
3. `scripts/cross_stage/11_stage1_diagnostics.slurm`
4. `scripts/cross_stage/tests/test_stage1_diagnostics.py`
5. `docs/STAGE1_DIAGNOSTICS_V1_CN.md`

入口复用 05 的数据、模型和随机数配置，以及 07 的 HVP 和 CG。05、07、10 和 `src/models.py` 均未改动。

每个任务生成 `manifest.json`，包含代码与输入文件哈希、实际命令配置、运行状态和输入文件未改变的验证。日志前缀为 `oct_s1diag`。

| 输出 | 内容 |
|---|---|
| `spectrum_seed*.json` | 未加额外 damping 的谱诊断及基线梯度 |
| `damping_seed*.csv/json` | 每 seed 的逐病人逐 damping 对比和原始诊断 |
| `damping_summary.csv` | 相同合格单元上的描述性汇总 |
| `true_dose_scaling_seed*.csv` | full 与 dose01 的真实参数响应比较 |
| `training_curve_seed*.csv` | 逐 epoch 训练曲线和固定目标观测 |
| `replay_checks_seed*.json` | epoch 25、50 的参数、Adam 与 RNG 精确比较 |
| `replay_final_seed*.json` | 最终 checkpoint 与 RNG 复现结果 |
| `all_rows.csv` | 当前任务合并数据 |
| `*_overview.png/pdf` | 直接用于讨论的曲线 |

本版专注 Stage 1，不训练 attack，不重新计算 h，也不输出 hᵀΔθ。该标量投影可在后续复用 attack 端的 h 追加，但不应把本轮向量几何指标误写成跨阶段 CE 预测表现。

## 7. 已执行验证与尚未执行的验证

已在 Python 3.12 与 PyTorch 2.8.0+cpu 上通过 9 项测试。

1. 对称正定小矩阵的 CG 解与直接线性求解一致，α 缩放一致。
2. 不定小矩阵的负边缘检测、移位检查和不合格结果排除有效。
3. HVP 与显式 CE+L2 Hessian 相乘一致。
4. 观测函数保持参数、梯度、模型模式和 RNG 不变。
5. CNN baseline 重放与原 05 的参数、Adam 状态、RNG 逐位一致。
6. 两条完整入口使用真实合成 checkpoint 运行，生成图表和记录，确认输入文件未改变。
7. 零效应时不把未定义的 cosine 写成成功。
8. full/dose 的面板和 α 不匹配时拒绝运行。
9. Delta 测试入口要求 CUDA，不会静默改用 CPU。

第 9 项在本地验证的是入口逻辑，尚未执行 CUDA 测试。真实 OCT 数据加载及现有 CUDA checkpoint 的重放必须在 Delta 验证。本地没有原始图像或 GPU，未生成真实诊断结论。

## 8. Delta 操作顺序

交付 ZIP 内的 `QUICKSTART_CN.md` 给出从 Mac 上传、先同步原代码、再安装新增补丁的复制命令。`sync_delta_code.py` 只暂存指定源码范围，提交后执行普通 push。若分支不符、存在范围外的暂存文件、远端领先或认证失败，它会停止，不执行强制 push、pull、reset 或 checkout。

安装后先提交 `tests`。`tests` 成功后，`damping` 和 `convergence` 可作为两个独立作业并行，也可与 22119842 的 score 并行。三者输出目录不同。无需重跑 full 或 dose01 truth。

下一次判断顺序为谱与残差、共同合格行、方向与幅度、训练曲线和精确重放。若改变 damping 后方向仍差，且训练目标仍显著漂移，再讨论延长训练或 Adam trajectory score。不能仅凭本轮某个失败指标宣布公式被证伪。
