# 跨阶段 LDS 梯子（v4）：按图像随机删 50%，重跑 L1 / L2 / L3

日期：2026-10-12。来源：导师 2026-10-12 的 Slack 回复（"用更新后的配置重跑 L1、L2、L3，看 TDA 方法对跨阶段影响有多有效"）。
父提交 `bffa3b8`（image LDS v3 reviewed）。本目录只新增文件，不改动任何已有实现，也不重训 Stage 1。

## 1. 问题

v3 已经证明：在按图像随机删 50% 的标准 LDS 下，Stage 1 的阻尼 IF 能排序 Stage 1 查询损失的变化（γ=0.03 时 CE 0.405）。
这一轮把目标换成 Stage 2 攻击损失 J，问同一套归因方法能否排序**跨阶段**影响，并用梯子定位误差出现在链条的哪一环。

## 2. 真值（事先固定）

- 子集：沿用 v3 的 32 个子集（每个保留 1,022 / 2,045 张 shadow3 训练图像），训练 seed 42–49，共 256 个子集模型 + 8 个全数据模型。全部直接读 v3 运行的 `cells/*/final.pt` 与 `interface.npz`，哈希在 prepare 时固定。
- Attack 训练表：Shokri 拓扑不变。shadow 0/1/2/4 与 target 用 calibration v4.1 的冻结旧模型（用户 2026-10-12 选择），只把 shadow3 的 4,343 行换成稳定配方模型的预测。成员标签 M₀ 不变：被删图像仍标为 member（value pathway，Eq. 83）。
- Attack：legacy 配置（MLP 4→64→2、Adam、50 epoch、batch 128、float32），每类一个 attack。attack seed 取 legacy 交叉面板的前 K=3 个，对所有训练 seed、所有类别相同（strict crossed）。
- 主目标：
  $$J=\sum_{c\in\mathcal Q}\omega_c J_{Q,c},\qquad \omega_c=\frac{|Q_c|}{\sum_{c'\in\mathcal Q}|Q_{c'}|}$$
  即合格类别的 target query 按图像等权的攻击 CE（logits 的 log-softmax CE）。
- 真值：$\Delta_j=\frac1{SK}\sum_{s,k}\big[J(P_{j,s},M_0;k)-J(P_{\text{full},s},M_0;k)\big]$。只测 Δvalue，不测 Δfull（用户 2026-10-12 选择）。
- 类别门槛（smoke 时在 8 个全数据模型 × K 个 attack seed 上判定，不看任何子集结果）：某类在全部 24 次 attack 上都满足 AUC ≥ 0.55 且 balanced accuracy ≥ 0.53 才进入 $\mathcal Q$。$\mathcal Q$ 为空则停止。

## 3. 梯子（每层都对同一组 32 个子集算 LDS）

| 层 | 用了多少真值 | 预测量（对子集 j，先逐 (s,k) 计算再平均） | 检验的环节 |
|---|---|---|---|
| L1 | 真实 $P_{j,s}$ | $\sum_{r\in\text{shadow3}} v_{s,k,r}^\top(P_{j,s,r}-P_{\text{full},s,r})$ | Stage 2 隐式微分 + 线性化（Eq. 11–21） |
| L2-retrain | 真实 $\Delta\theta$ | $J(\hat P,M_0;k)-J_{00}$，$\hat P=\Pi(P_0+J_p\Delta\theta)$，精确重训 attack | 只剩 $p(\theta)$ 的线性化 |
| L2-lin | 真实 $\Delta\theta$ | $h_{s,k}^\top(\theta_{j,s}-\theta_{\text{full},s})$ | 再加上把 $\Delta P$ 压成标量内积（Eq. 45–49） |
| L3 | 无 | $-\sum_{i\notin S_j}\tau_{s,i}(\gamma)$，$\tau_{s,i}=-\bar h_s^\top(H_s+\gamma I)^{-1}g_{s,i}$ | 完整公式（再加 Stage 1 阻尼 IF，Eq. 50–61） |
| proxy | 无 | $-\sum_{i\notin S_j}(-\bar h_s^\top g_{s,i})$ | 无 Hessian 对照（frozen_h 的图像版） |

- $v_{s,k,r}=\omega_{c_r}\,dJ_{Q,c_r}/dp_r$ 由 attack 的稠密 Hessian（fp64，约 450 维）隐式求解；$h_{s,k}=\sum_r(\partial p_r/\partial\theta)^\top v_{s,k,r}$。$\bar h_s=\frac1K\sum_k h_{s,k}$；由线性，L3 对 $\bar h_s$ 解一次等于逐 k 求解后平均。
- $g_{s,i}=\nabla\mathrm{CE}_i/N$，$N=2045$；$\Pi$ 把 JVP 结果截到 ≥1e-6 再归一化，报告截断比例。
- 删除者写法与 matched 写法只差统一的正比例和常数（$N_j$ 恒为 1,022），排序相同。
- Attack 阻尼 $\gamma_A$ 网格 {1, 0.3, 0.1, 0.03, 0.01}，L1 对每个值都报告；L2-lin / L3 / proxy 只用 $\gamma_A^*$ = 在全部 (s, k, c∈𝒬) 上 $\lambda_{\min}(H_A)+\gamma_A>0$ 且求解残差 ≤1e-10 的最小值（smoke 时由全数据模型确定，不看子集结果）。
- Stage 1 阻尼 γ 网格 {1, 0.3, 0.1, 0.03, 0.01, 0.003}，multi-shift Lanczos 一次求出；每个 γ 独立判定数值合格（真实残差 ≤1e-3、Krylov 正曲率、随机探针），不合格的 γ 不出分数。γ 不按 LDS 选；主报告取"全部 score seed 都合格的最小 γ"。
- Score seed：42–49 全部 8 个（每个 seed 一个 $\bar h_s$ 和一次 multi-shift 求解）。

## 4. 统计

- 统计单位是子集（m=32），单侧 5% 置换门槛约 0.29。每层报告 Spearman、2,000 次子集 bootstrap CI、置换 p、斜率与相对误差。
- 相邻层的差值用成对子集 bootstrap：L1 − L2-retrain、L2-retrain − L2-lin、L2-lin − L3(γ*)、L3(γ*) − proxy。
- 真值稳定性：训练 seed 不重叠的 4 对 4 可重复性；attack seed 两两之间的子集排序相关；子集 × seed × attack seed 的方差分量。
- 次要：逐类别的真值与 L1 / L2-retrain / L2-lin；Stage 1 查询损失的 v3 结果只作参照，不重算。

## 5. 作业与门槛

`smoke`（1）→ `base`（8，每个训练 seed）→ `truth`（256 个子集单元打包）与 `score`（8）并行 → `collect`。

smoke 必须全部通过：
1. 设计与 v3 运行的 `designs.json` 逐字节一致；legacy shadow3 行顺序等于 v3 接口行顺序。
2. 用 v3 全数据 `final.pt` 重算的 shadow3 预测与 v3 `interface.npz` 一致（≤1e-10）。
3. 同一 attack 训练两次逐位一致；本包的稠密伴随与 legacy `attack_implicit_v` 一致（相对 ≤1e-10）。
4. $h$ 的链式求导与有限差分一致；JVP 与有限差分一致。
5. 类别门槛与 $\gamma_A^*$ 写入 `smoke/frozen.json`，之后所有作业只读。

base 重训的 attack 必须与 smoke 的 J₀₀ 逐位一致；每个 truth 包重算该 seed 的 J₀₀，与 base 的差 ≤1e-6，Δ 用包内（同一 GPU）的 J₀₀。

## 6. 范围

- 静态阻尼 IF，在 raw 全数据端点计算；Stage 1 与 attack 都不是驻点（报告梯度范数）。轨迹版公式（Eq. 42 / 74）不在本包。
- 删一半图像会让 shadow3 约 1,023 个 member 行看起来像 non-member，这正是 value pathway 的定义；Δfull（改标签）不在本轮。
- L1、L2 用了重训后的真值，是定位诊断，不是可部署的归因。
