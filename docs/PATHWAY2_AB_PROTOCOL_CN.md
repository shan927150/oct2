# Pathway2 A/B 独立分支与并行实验协议

日期：2026-09-17。此协议取代上一版设计中“先把 A 作为主线前置条件”的排序。

交付状态：E0 和 B0 已实现并有 CPU 回归验证；Delta CUDA 与真实数据验收待运行。A、B1、B2 仍是实验设计。B0 直接调用 05 原训练函数；14_b0_train 只负责原输入加载、一次重放验收和剂量调度。具体操作见 PATHWAY2_E0_B0_RUNBOOK_CN.md。

## 1. 分支与被估量

共同起点为 cross-stage-calibration-v4 的 ba8db97887e18ea1bbf33f67b04d6eba69f47b70。不能从较旧的 main 开始，以免丢失 fixed-mask、严格 CUDA 重放和 v1.1 数值约定。

| 分支 | 定位 | 允许的科学结论 |
| --- | --- | --- |
| exp/pathway2-a-stationarity | 收敛干预的机制诊断 | 在明确改变的续训流程下，驻点误差与静态预测是否一起改善 |
| exp/pathway2-b-trajectory | 原训练流程的主要验证路线 | 原 finite-Adam 算法的局部删除权重导数是否可预测，能外推到多大剂量 |

原 baseline、原 full/dose truth 和原配置保持不变。A 的新锚点另记为 theta_A、P_A、J_A00，禁止覆盖或更名冒充原 theta_0、P_0、J00。

A 作为独立诊断不改变原主实验。如果将“训练到收敛的 shadow”另立为方法适用范围，需要明确缩小 scope、重新检查 attack 可用性和信号尺度；不能将其描述为对原任意训练流程的验证，也不能用新 truth 替换旧 truth。

A 无需等待 B0 失败才有诊断价值。B0 成功而静态导数失败时，A 同样可研究驻点误差；B0 未发现稳定区时，须先区分剂量、有限精度、非光滑点和训练放大，不能自动判定训练不稳定。

## 2. 隔离与冻结要求

Git branch 只隔离代码；worktree、输出路径与输入完整性检查需要一起落实。

- 原工作目录继续使用已有 oct2-calibration-v4，不 checkout 到实验分支。
- A、B 分别使用新 worktree，例如 oct2-exp-a、oct2-exp-b。
- 新结果分别存入独立父目录 oct2-pathway2-runs/A 和 oct2-pathway2-runs/B。
- 每个任务必须有独立 run_id、seed、patient、alpha、window 和 Slurm job/task 子目录。
- 结果路径经 realpath 解析后，不能等于、位于或包含原 baseline 目录；也不能落在另一分支的结果目录。必须检查 symlink。
- 不在原结果目录运行会改写 experiment_config、summary、panel 或 checkpoint 的旧 05/10 入口。
- 原 checkpoint 只读加载；需要新状态时在新输出目录保存。禁止 hardlink 后写入，禁止覆盖旧文件。
- 对代码、配置、split、selected panel、order 文件、checkpoint、原 truth 在运行前后做哈希检查。只做运行前检查不能证明任务没有写坏输入。
- 新 runner 必须接入 preflight 和退出时完整性复核；当前 preflight 本身是只读检查，不是操作系统级写保护。

冻结配置由 experiments/pathway2/frozen_baseline.json 记录，来自实际结果与 v1.1 输入哈希：

| 项目 | 原配置 |
| --- | --- |
| Stage 1 / Stage 2 epochs | 50 / 50 |
| Shadow 3 原训练集 | 2045 images |
| Shadow 模型 | SmallCNN，503044 参数，dropout=0.2 |
| Stage 1 | Adam，lr=0.001，batch=128，coupled wd=1e-5 |
| Stage 2 | MLP 4→64→2，Adam，lr=0.01，batch=128，wd=0 |
| Stage 1 seeds | 42、43、44、45、46 |
| Attack seeds | 5101、5102、5103、5104、5105 |
| 删除 | fixed_mask；患者损失权重 1-alpha；原 batch 与原分母保持不变 |
| 主评价 | patient-matched class CE，J10-J00，membership/route/row identity 固定 |

原训练过程还有初始化、CPU/CUDA 随机流和 Adam moments。配置相同不足以证明这些状态相同，需逐位 no-op/replay 检查。

已知文件哈希覆盖上传审计记录中的输入；图像字节没有上传，不能补造历史图像哈希。本轮额外记录各任务实际加载的 X/y 摘要，并在汇总时要求一致；B0 还必须逐位重放原参数、Adam 状态、RNG 与接口，核对 patient/raw index/split。此数据验收不等于历史原始 JPEG 的字节级快照。preflight 的文件完整性 PASS 也不等于 GPU 或新导数实现验收已完成。

## 3. 固定面板及 1369 的角色

主要小试使用 seeds 42、43 × patients 807（DME，6 images）、2085（DRUSEN，5 images），共 4 cell。

1369（DME，7 images）作为预先注明的低终点梯度观察对象。它不替换任何原患者，也不混入主 4-cell 指标以改变结果。E0/A 对原 8 患者测梯度尺度和饱和比例；1369 的额外重训剂量作为明确的扩展任务，非首轮必需任务。

1369 在五个 seed 的 eval b 范数：

| seed | ||b_1369|| |
| --- | ---: |
| 42 | 1.8455352376204805e-8 |
| 43 | 6.5528330794986525e-9 |
| 44 | 3.4031206793227387e-9 |
| 45 | 6.49515880209473e-12 |
| 46 | 1.6471089491007302e-8 |

这些数来自 reverse_seed*_patient1369.json 的 full.rhs_norm。低终点梯度是已观察事实，“影响只剩噪声”不是已观察事实。

## 4. 饱和与信号验收

必须监测饱和，但不能把“隐式前提成立”与“信号必然消失”画等号：

1. 正则化驻点满足平均数据梯度加 lambda*theta 为零，不要求每个样本梯度都为零。
2. 患者组梯度小，可能来自个体梯度小，也可能来自方向抵消。
3. 小 b 仍需结合曲率放大和 h 的方向理解。H^{-1}b、h^T H^{-1}b 的量级不能只由 ||b|| 推断。
4. 浮点数可表示远小于 1e-8 的数；绝对值小于 machine epsilon 并不自动等于数值噪声。
5. 终点梯度接近零不意味着该患者在训练早期没有影响。B 的轨迹导数包含这些历史作用。

每个 baseline 锚点至少记录：

- eval 和 dropout 目标各自的 ||b_p||，明确算子与 MC 误差。
- 各患者的 CE、正确类概率、逐样本梯度范数，以及 ||sum g_i|| / sum ||g_i|| 的抵消比。
- ||h||、||x||、h^T x；按 class 和 attack seed 保留结果，标明 h 是 implicit 还是 trajectory。
- 真实 ||Delta P||、可获得时的 |Delta J|，与数值/配对重复误差尺度比较。
- 低 CE、低组梯度、梯度无法数值分辨三种计数分别报告。阈值和误差估计方法必须在看新结果前固定。

数值分辨率至少结合固定权重处的高精度梯度对照、求解误差和剂量差分行为。exact no-op=0 只证明重放一致，不能把差分精度阈值也设成 0。

A 中旧 h 只能作为固定探针。报告“微调后 h 的变化”时，必须在新接口上按相同 attack 配置重新训练并求导；否则应标注未测量。不能以 h_old 的恒定范数代替 h_new。

若误差降低的同时真值信号也降到不可分辨区，结论是“信号无法分辨”，不能宣传 score 变准。必须与零预测基线比较。

## 5. 首轮并行矩阵

不预设“E0 一小时内”或“B0 15–20 分钟”。记录实际装载时间、一次 Stage 1 重放时间、HVP/梯度耗时、峰值显存，再调整 Slurm walltime。旧 dose01 的 score 时间不能直接折算为新的 Stage 1 训练时间。

首轮提交一个 CUDA 测试任务，成功后 E0 与两个 B0 seed 任务最多并行占 3 GPU；实际并发服从账户资源限制。B1-smoke 实现并通过短链验收后可使用第 4 张卡，A 暂不自动排队。

| 任务 | 分支 | 首轮工作量 | 可与谁同时跑 |
| --- | --- | --- | --- |
| E0 | B，共享诊断结果供 A 读取 | 4 core cell × 2 剂量；两个 seed 的原 8 患者梯度监测 | B0、B1、A |
| B0-seed42 | B | 1 次 baseline no-op + 2 patients × 4 小剂量 | E0、B0-seed43、B1、A |
| B0-seed43 | B | 同上 | E0、B0-seed42、B1、A |
| B1-smoke | B | seed42/patient807，单个含患者 batch → 1 epoch → 5 epochs | E0、B0、A；后一级依赖前一级通过 |
| A-pilot | A，可选 | seed42/patient807，baseline/full/dose01 × 2 续训方案；另测原 8 患者 b | 与 B 独立；资源不足时排后 |

当前一键提交只排 CUDA 测试、E0、B0 array；后两者用 afterok 依赖测试。A-pilot 是否投入资源结合 E0 决定，E0 投影份额不作为排除静态路线的唯一依据。B1 可以在另一个 worktree 开发，避免改变已排队任务锁定的 B 工作树。

B0 有 16 次扰动训练，加 2 次 baseline no-op，共 18 次 Stage 1 训练。按 seed 切成两个任务，在任务内复用已装载数据和原 baseline。原 target、其他 shadow、Stage-2 attack 不在 B0 重训；B0 结果明确标注 stage1_only_probe，不冒充新的 J10 truth。

B0 array=0-1%2，每个任务跑自己的 8 个剂量/患者组合，然后自动分析；输出包含 array job id 与 seed。所有组合独立输出，汇总必须等待两个任务完成。

E0 不新增训练，但仍需要 GPU 梯度/HVP 运算。A 与 E0/B0 无科学上的强制先后关系；冻结完整性检查与各自实现验收仍是共同前置条件。

## 6. E0：原 checkpoint 残差分解

目的：在不改变原训练对象的情况下测量驻点残差与有限位移 Taylor 余项。

对 L_alpha=L_0-alpha*L_patient，令：

$$
g_0=\nabla L_0(\theta_0),\quad
g_\alpha=\nabla L_\alpha(\theta_\alpha),\quad
d=\theta_\alpha-\theta_0,\quad
b_0=\nabla L_{\rm patient}(\theta_0),\quad
H_0=\nabla^2L_0(\theta_0).
$$

H_0 已包含 wd。定义

$$
\mathcal R_\alpha=
\nabla L_0(\theta_\alpha)-\nabla L_0(\theta_0)-H_0d
-\alpha[b(\theta_\alpha)-b(\theta_0)].
$$

验证代数闭合：

$$
(H_0+\gamma I)d-\alpha b_0
=(g_\alpha-g_0)-\mathcal R_\alpha+\gamma d.
$$

记录绝对误差、各项范数、内积、相互抵消程度。各项范数不构成可相加的因果百分比。极小 RHS 单独标注，避免比值爆炸后作平均。

首先沿用 eval 目标，以便解释现有静态系统；dropout 目标的结果独立标记，不与 eval H 混用。

## 7. B0：原算法的小剂量响应

原 baseline、50 epochs、学习率、初始化、orders、dropout、wd 和原 batch 分母保持不变。从原初始化开始全程施加 alpha=0.03、0.01、0.003、0.001。epochs 从冻结结果配置和原 order 数组读取，不由新脚本另定。

记录 D_alpha=(theta_T(alpha)-theta_T(0))/alpha、prediction 空间差分，以及固定原 h 的投影。原 alpha=0.1/1 是有限剂量参照，不把它们预设成导数真值。

不同小剂量的方向和幅度应在可分辨区间内趋于稳定。局部光滑时，一阶差分会有 O(alpha) 偏差；剂量缩小后若进入浮点误差主导区，不能继续机械缩小。

每个 cell 分别要求连续两对相邻剂量（至少三个剂量点）满足 1-cos<=0.05、参数导数相对变化<=0.05；两端位移都需超过保守 float32 分辨率尺度的 100 倍。该尺度是工程启发式，不是测得的随机噪声或严格误差界。分开的通过区间不合并成一段；总体通过要求所有所请求 cell 各自通过。P 与 patient-matched-class h 分开报告，不由参数空间通过代替它们通过。

若未找到稳定区，先记录“在当前剂量/精度下未解析局部导数”。检查初始状态、随机配对、激活非光滑点与精度后，再讨论敏感性。B0 失败不等于导数不存在，也不自动要求改 baseline。

## 8. B1：有限 Adam 切向传播

固定随机操作，令 w=(theta,m,v,tau)，u=dw/dalpha。传播：

$$
u_{t+1}=D_w V_t\,u_t+\partial_\alpha V_t.
$$

当前完整轨迹为 50×ceil(2045/128)=800 steps；最后一批为 125 images。

前向传播的常驻导数状态相对训练步数 T 不增长；其大小仍随参数维数、Adam 状态与同时传播的方向数增长。4 个 core cell 是两条 seed 轨迹各有两个患者方向，不是同一条轨迹上天然可合并的四个方向。

工程策略：

- 使用函数化、等价的 Adam 更新；不能直接把现有带副作用的 optimizer.step 当作已经可被 jvp 安全转换。
- 当前运行环境固定 PyTorch 2.8。官方文档提示部分算子不支持 forward-mode AD，需在实际 SmallCNN、pooling、dropout 上验收。
- 比较 primal 参数、moments、step counter、RNG 和原实现；仅“算法公式一样”不足以保证 800 步轨迹一致。
- paired finite difference、primal、tangent 必须使用同一 batch 的 dropout realization，并保持下一 batch 的随机流不被额外诊断消耗。
- 保留 coupled wd 与其导数，不能改成 AdamW。
- 检查零梯度/零二阶矩坐标是否产生 NaN。不能为通过验收而改 epsilon 位置、平滑激活或替换模型；若使用稳定等价导数，需单独验证。
- 加不存在病人的方向：初始 tangent 与直接注入均为零，参数与 moments 的 tangent 全程必须有限且逐位为零。它是零方向不变量测试，不能单独检验非零 Jacobian、dropout 流或 coupled-wd 导数；仍需非零方向有限差分、RNG/primal 对照和非零 moments/weight-decay 测试。
- 每一步传递完整的数值 tangent，可释放已完成步骤的 autograd 图以控制内存。不能把 tangent 重新置零；若还要对最终 tangent 做更高阶求导，释放图的语义需另行讨论。

先单 batch，再 1 epoch、5 epochs。含患者样本的 batch 才能检验直接注入项。每个窗口配相同窗口的 truth；将 25→50 的敏感性直接当作 0→50 的敏感性是错误的。

短链可采用 1% 导数相对误差、可分辨相邻剂量 5% 变化作为预注册工程扩展门槛；近零信号用绝对误差。未通过即停止扩展，不通过调剂量或筛 cell 宣布成功。

来源：[PyTorch 2.8 jvp](https://docs.pytorch.org/docs/2.8/generated/torch.func.jvp.html)、[torch.func 限制](https://docs.pytorch.org/docs/2.8/func.ux_limitations.html)。

## 9. B2：闭合 Stage 2 与完整删除验证

先验证 Stage 1 轨迹导数，再通过 prediction Jacobian 与 Stage 2 式 42 得到原有限训练 attack 的超梯度。保持 attack 50 epochs、原 attack seeds、原 target queries 和成员标签。

只实现 Stage 1 轨迹导数、保留 Stage 2 隐式 h 时，结果必须命名为“Stage 1 trajectory + Stage 2 implicit”。PDF 第 9 节的严格有限轨迹形式是式 42 + 式 74。

式 74 是局部导数。通过局部有限差分后，还要单独检验 alpha=0.1/1 的外推误差。只有完整删除的原 J10-J00 预测也有效，才能报告该剂量上的实际归因效果。

## 10. A-pilot：有限预算的配对收敛干预

只在 A 输出空间运行，seed42/patient807 先做机制小试。所有新状态的完整 lineage 指向各自原 baseline/full/dose01 checkpoint。

必须恢复完整 Adam 状态和 RNG。原脚本只给 baseline 的 epoch checkpoints 保存完整 optimizer state；LOO 终点若无额外完整状态，先重放恢复，并验证 LOO 自身的终点权重与 RNG。已有 baseline 重放成功不能代替 LOO 的逐条件验收。

每个原状态通过恢复验收后分两臂：

| 方案 | 额外预算 | 唯一计划干预 |
| --- | --- | --- |
| A-control | 最多 25 extra epochs，lr=1e-3 | 多训练 |
| A-decay | 前 10 extra epochs lr=1e-4，后 15 为 1e-5 | 在相同额外步数下降低学习率 |

读取原配置得到 T0=50；额外 25 是独立诊断预算，不将 baseline 的 epochs 改成 75。额外 orders 与 dropout 配对在两臂间预先固定。

在 +0/+5/+10/+25 记录匹配目标的梯度、objective、参数变化、E0 分解、各患者 b 的尺度/抵消/数值分辨率和信号计数。在 +0 与最后一个可解释锚点，按相同 attack 配置重新计算 h_new，不能复用旧 h 冒充新链的导数。

若梯度误差降低但静态预测仍差，检查局部分支、Taylor 余项和目标匹配；若信号消失，报告不可分辨。R_alpha 不降不构成排除所有静态方法的依据。

若后续做共同锚点局部重优化，另立 A-local，不与 A-paired 的 truth 合并。加入 proximal 项等于改变真实目标，必须在模型训练与 Hessian 两端同时体现。

## 11. 汇总、扩展与停止规则

主指标为幅度误差、相对零预测改善、方向/符号、prediction 与 J 的误差；Spearman 作为补充。低信号 cell 不静默丢弃，单列覆盖率与误差上限。

4-cell 小试只作工程/机制判断，不作患者总体显著性声明。扩展到原 8×5 面板后，仍按患者、Stage 1 seed、attack seed 的交叉结构评价；不能将所有重复测量视为独立患者。

并行是计算调度选择，不改变验收顺序。B1 完整轨迹依赖短链验收；B2 依赖 Stage 1 导数验收。A 的正负结果均不能改写原 truth 或阻断 B 的有效局部验证。

后续实现推进 B1 短链；根据 E0 安排 A 的完整状态恢复诊断，B2 接续通过验收的 Stage-1 轨迹导数。E0 代数闭合是必要检查，不是独立的 HVP 正确性证明；JVP 对照和有限差分梯度检查各有作用，单个差分步长跨越 ReLU/max-pool 分支不能宣布整个有限差分方法无效。
