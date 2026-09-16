# Stage 1 诊断 v1.1

本版替代 v1 的执行方案，入口为 `12_stage1_diagnostics_v11.py`。旧版 11 号文件保留为公共函数和回归测试依赖，不要再按旧版说明提交 damping。准备日期为 2026-09-16。以下是设计与实现说明，不是新的 OCT 实验结果。

## 当前应做什么

先安装本版并运行 `preflight`。它只读取已完成的 full、dose01 checkpoint 和配对记录，在 CPU 上计算 40 个 patient–seed 单元的幅度、方向和基线 seed 距离，不读取 OCT 图像、不训练、不计算 Hessian。

确认预检完整后，提交新入口的 GPU tests。测试通过后可并行运行 `convergence` 和 `damping`。已提交的 score 22119842 可以继续。代码同步不会改写正在运行的训练文件。

## 与 v1 相比的变化

| 项目 | v1.1 |
|---|---|
| 预检 | 增加只读取 checkpoint 的 CPU 模式 |
| γS 网格 | 预先固定为 0.03、0.1、0.3、0.7、1.0、2.0 |
| Stage 1 求解 | 每位病人共享一次从 b 出发的 Lanczos 基，分别检查 50 与 100 步的移位最小残差近似 |
| 数值资格 | 两种深度均通过真实残差，并检查解和 h 投影随深度的稳定性 |
| 不定系统 | 允许作为探索性线性系统诊断，单独标记负曲率，不当作极小点影响函数已经合格 |
| 主评价 | hᵀΔθhat 与 hᵀΔθtrue 的相关性、MAE、幅度和符号 |
| h 来源 | 按原 attack seeds 重建 baseline attack，核对 J00，保存各 seed、各类的 h 矩阵 |
| 反向检验 | 同时报无约束最优 γ、非负约束最优 γ，以及对应绝对和相对残差 |
| 训练目标 | 保留逐 epoch 原训练重放，dropout MC 默认从 4 提至 16，增加均值梯度误差尺度估计 |
| 计算成本 | 记录实际 Stage 1 HVP 次数，不用 CG 上限冒充实测成本 |

## 1. 固定数据与干预

原 affected shadow 为 3，Stage 1 seeds 为 42–46，attack seeds 为 5101–5105。仍使用原 8 位病人，DME 与 DRUSEN 各 4 位，不按本轮数值表现删病人。full 为 α=1，dose01 为 α=0.1。

沿用原 `fixed_mask`，保留 batch 张量、分母和随机数序列，对病人图像使用 1−α 的损失权重。数据、面板、epoch orders、baseline 参数、随机数指纹和 J00 均有配对检查。训练 epochs 与 epoch checkpoint 位置从 `experiment_config.json` 读取，当前正式配置应为 50 和 [25,50]，实际读取值会写入 `manifest.json`。

## 2. 方程与目标

\[
L_{eval}(\theta)=\frac1N\sum_i\ell_i^{eval}(\theta)
+\frac{10^{-5}}2\|\theta\|^2,
\quad \bar H=\nabla^2L_{eval}(\theta_0),
\quad b_p=\frac1N\sum_{i\in I_p}\nabla\ell_i^{eval}(\theta_0).
\]

\[
(\bar H+\gamma I)x_p=b_p,
\qquad \widehat{\Delta\theta}_p(\alpha)=\alpha x_p,
\qquad \Delta\theta^{true}_p(\alpha)=\theta_{p,\alpha}-\theta_0.
\]

正号对应下调病人权重。b 不含 weight decay。H̄ 含原 Adam 的 L2 项。HVP 仍复用 07 的 eval 模式双重自动微分，因此本版没有把 Hessian 改成 dropout 期望目标的 Hessian。

沿用现有 score 的图像平均 CE 局部近似，不将逐 batch Adam 的有限训练过程等同于确定性全量优化。

h 是 attack 查询 CE 经预测向量接口对 Stage 1 参数的局部导数。Stage 2 damping 固定为 γA=0.2，与已通过数值资格的原 full score 一致。h 按每个 Stage 1 seed、类别和 attack seed 分别计算。主表先在固定的 5 个 attack seeds 上取均值，另保存逐 attack-seed 行。

## 3. 预检读数的边界

输出包含

\[
R_\alpha=\frac{\|\Delta\theta_{0.1}\|}{0.1\|\Delta\theta_1\|},
\quad \cos(\Delta\theta_{0.1},\Delta\theta_1).
\]

Rα 近 1 只说明幅度与线性缩放相容，还需看方向。近 10 表示两种剂量的变化幅度相近，不能识别“纯混沌”。

跨 baseline seeds 的原始参数距离是尺度参考。不同初始化和网络参数对称性会影响这个距离，不能把它作为轨迹漂移的因果分解。没有按这些比值自动剔除病人或改变 damping 网格。

## 4. 共享 Krylov 与数值检查

利用 Kₘ(H̄+γI,b)=Kₘ(H̄,b)。从 b 出发构造完整重正交化的 Lanczos 基，在小型扩展投影矩阵上解最小二乘。这个实现使用 MINRES 的投影最小残差形式，可以处理不定投影，不要求原 CG 的正定前提。它不是稠密求逆，也没有调用原附件中未经修正的 `shifted_solutions`。

小矩阵在线性代数 CPU float64 下求解，系数明确转到基向量所在设备再重建 x。HVP 与模型保持原精度，正式模型默认 float32。

每个 γ 在 50 和 100 步都重新计算

\[
r=\frac{\|\bar Hx+\gamma x-b\|}{\|b\|}.
\]

两次真实残差都须 ≤1e−3，解的相对变化须 ≤0.01，各 attack seeds 上 hᵀx 组成向量的相对变化须 ≤0.01，基的最大正交性误差须 ≤1e−3。attack 求解还要通过原 07 的资格检查。零 RHS 的方向和相对量标记未定义，不写成成功。

闭式或投影残差只作诊断，不能代替真实残差。有限步检查不是正定证书，也不是病态系统的严格前向误差界。不定分支的数值合格不代表驻点假设已成立。若低 γ 没有合格行，报告未解决，不依据真实 LOO 相关性调整 γ。

默认只运行预先固定的两种深度。如果残差或深度稳定性不合格，之后可以仅依据数值诊断增加步数，使用新输出目录。不会自动根据预测质量挑参。

## 5. 谱与反向检验

保留两个独立随机起点的全局谱边缘探针，并增加病人 RHS 加权的 Ritz 节点和权重。`mass_estimates` 是有限求积近似，不给未经证明的 CDF 上下界。

`signed_damping_projection = γ bᵀx / ‖b‖²` 是带符号标量。在不定 H̄ 下不一定处于 [0,1]，不能当成百分比，接近 1 也不能单独证明 x≈b/γ。仍同时报告 cos(x,b)、‖x−b/γ‖/‖x‖ 和向量项幅度。

对每个真实 d，额外一次 HVP 得到 H̄d，然后计算

\[
\gamma_* =\frac{d^T(\alpha b-\bar Hd)}{\|d\|^2},
\qquad \gamma_*^+=\max(0,\gamma_*).
\]

分别报告两个最优值的绝对残差、相对 RHS 残差，以及以 ‖H̄d‖+|γ|‖d‖+‖αb‖ 为分母的尺度化残差。后者帮助判断 RHS 极小时的比值膨胀。

反向残差大只能说明当前 eval 算子、RHS 与标量 damping 不能解释这个有限剂量变化，不能单独判断是训练未收敛、目标不匹配、非线性还是轨迹敏感性。

## 6. h 配对、缓存和汇总

每个 seed 一次性重建原 baseline attack，γS 扫描中不再训练 attack。每个病人对应的 full/dose J00 逐 attack-seed 都要与重建值匹配，容差 1e−6。原 `score_ladder_A0.2_S1/ladder_rows.csv` 存在时，还核对 hᵀΔθtrue 与原 L2，允许 float32 点积与 float64 汇总之间的误差，绝对容差 1e−5、相对容差 1e−4。

保存 `h_seed*.pt` 和来源校验记录以便追溯。本版不自动读取别的运行的缓存，也不会误用旧 attack seeds 的 h。

`projection_summary.csv` 只在所有预定 γ 都合格的共同单元上比较，报告 patient–seed 和 patient mean 两个层级，并分别给出 DME、DRUSEN。patient mean 还要求病人具备全部所选 Stage 1 seeds，不对缺失 seeds 取均值冒充完整面板。若没有共同单元，汇总为空。

逐 γ 的散点图明确标记实际合格数量，各 γ 覆盖率可能不同。不要直接凭图选择最佳 γ。所有描述性相关性都没有显著性宣称，40 个重复单元不等于 40 个独立病人。

## 7. Convergence 与 dropout

继续重放原 baseline 50 epochs，在 epoch 25、50 比较参数、Adam、CPU/CUDA RNG，最终比较 baseline 参数与 post-train RNG 指纹。失败就停止，不能把不匹配曲线当作原训练轨迹。

每 epoch 记录在线训练 CE、固定 eval CE+L2 和参数更新，每 5 epochs 记录全数据 eval 梯度。epoch 0、25、50 做 16 次 dropout MC，记录 CE 误差与均值梯度的随机误差尺度估计。增加 MC 只减少抽样误差，不能保证足够判断驻点。

eval 负曲率与 eval 梯度只针对 eval 目标。训练时优化的 dropout 期望目标与之不同。本版没有延长训练，也没有实现 Adam trajectory score。

## 8. 默认输出与验证状态

| 模式 | 默认规模 | 主要文件 |
|---|---:|---|
| preflight | 40 行，加 10 个 baseline seed 距离 | checkpoint_ratios.csv、baseline_seed_distances.csv |
| convergence | 255 个 epoch 观测点 | training_curve_seed*.csv、replay_checks_seed*.json、曲线图 |
| damping | 480 个汇总行、2400 个逐 attack-seed 行 | projection_summary.csv、damping_seed*.csv/json、reverse_seed*.json、rhs_spectrum_seed*.json、h_seed*.pt、散点图 |

所有模式创建新输出目录，保存代码及输入哈希，结束时核对原输入未改变。失败写入 manifest 状态，不将部分结果伪装成完整结果。

本地已执行 CPU 回归验证，包含原 9 项测试与新版 10 项测试。覆盖真实/投影残差区别、SPD 与不定系统、近奇异情况、零 RHS、约束反向检验、三种入口以及真实合成 checkpoint 的 attack 重建。GPU 测试和真实 OCT 运行仍需在 Delta 执行。完整运行命令见包根目录 `QUICKSTART_CN.md`。
