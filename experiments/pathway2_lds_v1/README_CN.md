# continue2 公式核验与病人级 LDS：隔离实验 v1

日期：2026-09-29。此目录只新增实验代码，不修改 stable、polish、原 A/B/C 的实现或输出。

## 1. 本轮回答什么

可以并行开展局部公式核验与小规模 LDS。暂缓扩大旧 damping 网格、相同配置 polish 和全 40-cell 冷启动重训。旧结果保留；不需要删除、覆盖或取消其他作业。

两条证据链分别报告：

1. **公式的条件与实现**：受控两阶段驻点模型、含完整 Adam 状态的两阶段轨迹、真实 OCT 的接口链式求导有限差分，以及可选的局部驻定目标响应。
2. **预测能力**：在冻结的随机病人子集上，公式分数能否预测冷启动训练后的输出；用 LDS、幅度误差及 L1/L2/L3 定位。

数学证明依赖 continue_pathway-2 的前提。有限实验只能核验实现、检验近似，不能证明当前非驻定 CNN 满足全部前提。高 LDS 不能单独证明 Eq.56/79 正确，也不能推翻已经观察到的单病人非线性。

## 2. 对所给建议的修正

- 病人聚合、重复训练平均、跨子集相关是合理方向。TRAK 的 LDS 定义为子集输出与保留项分数之和的 Spearman；Appendix C.4 使用 100 子集、每子集 5 次训练，再计算均值输出。它不是对 LOO 必然有效的替代，也没有保证 IF 在 LDS 上表现良好。
- 同 seed、同初始化、同 batch 顺序的真实删除响应已出现 R≈9；这部分是算法对干预的敏感性，不能全部归为未配对的随机噪声。现有 stable 结果只证明尾段平稳及 no-op 可复现，尚未证明 seed 方差已经充分降低。
- Spearman 对单调尺度变换不敏感，并不自动消除 basin/trajectory 分叉。中心极限定理也不保证多病人效应变成可加函数。50% 删除比局部微扰更远，可能更难。
- LDS 高只支持指定子集分布、输出与算法上的排序预测。必须同时看校准斜率、未经拟合的误差、跨 seed 稳定性。单调非线性响应也能取得 LDS=1，因此不能仅据此认定可加性成立。
- L2-lin 使用真实子集参数位移，L1 使用真实接口输出；二者是重训后的诊断，不是可提前部署的逐病人 attribution。不能把单病人真实位移直接相加当作子集真值。
- 用测试子集标签拟合 datamodel 再报告测试分数会泄漏。这里实现按子集做外层留一、内层选 ridge 正则的交叉拟合；m=8 远少于病人数，只是数据有限的对照，不称“oracle”或“上界”。
- frozen_h 是单 checkpoint 梯度 proxy，不是完整 TracIn；没有把它贴上原论文 TracIn 的标签。
- 热启动有限步算法不天然等同于驻点 IF。冷/热启动分数差同时混合优化预算、随机性与目标差异，不能直接命名为纯 estimand gap。
- 本项目已核对的 E100 γ=1 参数 cosine 结果是 36/40 正，不使用此前未经支持的“38/40、5.8σ”。患者和 seed 的依赖也不能忽略。

## 3. 公式、符号与分母

原始训练集有 N=2045 张图，病人 p 的全部训练图像构成 I_p。令

\[
F(\theta)=N^{-1}\sum_i\ell_i(\theta)+\lambda\|\theta\|^2/2,\quad
b_p=N^{-1}\sum_{i\in I_p}\nabla\ell_i(\theta_0),\quad H=\nabla^2F(\theta_0).
\]

病人损失权重增加 ε 时，原驻点假设下的导数为 −H^{-1}b_p；删除/降权的方向相反。对输出 Y，令 h=∇Y，解 (H+γI)w=h，定义保留项分数

\[
\tau_p=-w^Tb_p.
\]

因此，对权重变化 δw_p，预测 ΔY=Σ_p δw_p τ_p。单病人降权 α 的预测为 −ατ_p=+αw^Tb_p。代码先以解析模型检查正负号。

代码区分两个目标，禁止混报：

| 模式 | 训练操作与目标 | 对齐的 δw_p |
| --- | --- | --- |
| `subset`，默认 | 从全训练顺序中过滤未保留病人，重新分 batch；保留图像均值 CE + 原 λL2；140 epochs 的 warmup/cosine 按实际 step 总数重算 | (N/N_S)1{p∈S}−1 |
| `fixed_mask` | 原 batch 行与分母不变；未保留病人权重为 1−α；L2 不变 | −α(1−1{p∈S}) |

每个子集保留的病人数相同，但病人图像数量不同，N_S 并不恒定。因此默认 subset 同时报告：

- 文献形式的 canonical LDS：corr(Σ_{p∈S}τ_p, Y(S))；
- 与实际归一化一致的一阶外推：corr(Σ_p δw_pτ_p, ΔY(S))，并报告斜率和未经校准的误差。

这些是病人 cluster、固定聚合输出上的 LDS 扩展，不宣称逐字复现 TRAK 的图像级平均 LDS。固定 mask 删掉约半数图像时，CE 相对 L2 变小；它不是标准 subset 目标。

## 4. 第一轮：8 × 3 的 Stage-1 筛查

固定 shadow3、原数据/划分、stable 配方；从已完成的 raw baseline 42/43/44 计算分数。dropout=0、Adam、lr=1e-3、10% warmup + cosine、wd=1e-5、batch128、140 epochs。EMA 随训练核保存，但此轮主分析只用 raw，不能把 raw/EMA 算成独立重复。

病人子集：从 shadow3 全体训练病人中均匀抽取 round(P×0.5) 人，m=8 个不同子集；不选择“效果好”的子集，不配互补对子，不在看结果后改变分层规则。各子集使用共同 seeds42/43/44。固定 subset_seed=20260929。

主输出：原 shadow3 heldout 病人的 CE，先病人内平均、再类别内病人平均、最后四类等权平均。所有 query 行与权重冻结。选择一个聚合输出可以只解一个 h 的伴随系统，而不用为每个训练病人分别求逆。

作业：smoke → 3 个评分任务 → 1 个完整 no-op → 24 个子集训练任务。smoke 含受控公式测试、真实 CNN h 的有限差分、实际子集两轮精确 replay。评分使用 FP64、γ=1、最多80次 CG、12步 Lanczos、真实残差阈值1e-3。Lanczos 正值只是数值诊断，不能证明整个 Hessian 正定。求解不合格时保留诊断并使作业失败，阻止后续批量训练；不要根据 LDS 挑 damping。

完整 no-op 比较新调用得到的 raw、EMA、Adam 状态与旧 stable seed42 baseline，必须逐元素相同。评分、子集和 no-op 均检查数据/代码/配置哈希，不覆盖旧输出。

收集的读数：

| 读数 | 作用 |
| --- | --- |
| seed 间子集排序相关、subset×seed 残差、删去一个 seed 的敏感性 | 判断 k=3 的均值是否足够稳定；不是已证明的统计上界 |
| canonical LDS 和归一化匹配的 IF 预测 | 评价保留项分数、实际干预的一阶外推 |
| L1=Σ v·ΔP_true，L2=h·Δθ_true，L3=Σδwτ | 定位接口线性化与参数响应近似；此处 L3 仅是 Stage-1 因子 |
| frozen_h、随机置换零假设 | proxy 和无关联对照 |
| nested ridge、仅类别病人/图像数量的 nested ridge | 检查更普通的线性/组成特征是否已经解释结果 |
| 校准斜率、RMSE、relative error | 防止“排序可以但幅度失真”被称为公式成功 |

Bootstrap 对整个子集行同步重采样，保留三个 seed 列；CI 条件于这个 seed panel。置换检验只适用于预先固定的公式/proxy 预测，不给使用真值的 L1/L2 或拟合后的 ridge 套相同显著性解释。m=8 是探索轮，不能凭显著/不显著做最终论文判断。

## 5. 可并行的局部公式实验：改变目标并明确标记

已有 raw 端点非驻定，直接 warm-start 几步并不能补上 Eq.52。可选实验使用 seed42、patients807/2085，构造

\[
\widetilde F_{\epsilon,p}(\theta)
=F(\theta)-g_0^T(\theta-\theta_0)
+\tfrac\gamma2\|\theta-\theta_0\|^2
+\epsilon N^{-1}\sum_{i\in I_p}\ell_i(\theta),\quad g_0=\nabla F(\theta_0).
\]

于是 ∇F̃_{0,p}(θ0)=0，Hessian=H+γI。只要局部非奇异且跟踪同一局部解分支，就得到 dY/dε=−hᵀ(H+γI)⁻¹b_p，直接对应当前计算的 damped 因子。

代码从相同 θ0 对 ε=±0.01、±0.001 独立重优化，共8个短任务放在一个 GPU 作业。FP64全量目标、LBFGS最多40次 outer；约300次函数评估的软预算，单次 line search 可超出该计数。新的 θ0 每次重置；不是原 polish 继续跑。

只有终点残差 ||∇F̃||/(|ε| ||b_p||)≤1e-3 的正负成对结果才标为 qualified。报告中央差分、公式值、相对误差、参数位移和残差。较小 ε 并不保证更好，必须检查数值误差/优化精度。未合格不是公式被证伪。

这条线的成功 claim 仅是“受控、改目标的局部近似实现获得支持”；不能包装成原冷启动 LOO 成功。γ=1 与线性校正都改变了被估量，尤其线性校正不只是普通 proximal 正则。

## 6. 下一轮与完整跨阶段公式

先看筛查结果，不自动扩到96次训练：

- seed 排序不稳：先增加相同子集的独立 seed，调查 subset×seed 交互；不要仅加更多子集稀释问题。
- seed 响应稳定但公式无区分度：继续看 L1/L2 分解；不要用 ridge 好作为 L3 好的证据。
- 公式有区分度且不是单纯类别/样本数组成效应：冻结所有选择，再用独立子集做确认。建议确认集32子集×3seeds，探索子集不用于最终验收；最终需要更稳定估计时再考虑64–100子集与5seeds。
- ρ=0.9/0.99 的较小删除尺度作为单独实验分布；不和ρ=0.5混成一个相关。

跨阶段确认必须重新生成 stable baseline 的 attack 与 v/h，不能复用旧配方的 h。保持原 Shokri 风格 shadow→预测概率→按类别 attack 的拓扑、target 与 query 固定。对每个子集模型只替换 shadow3 的4343条预测行，其余 shadow 行、原 membership、类别路由、行组成均冻结。被删病人的原 member 标签在 value-only 干预中仍保持原标签；真实 membership 重标记属于 Eq.84–87 的另一项。

攻击端使用成对相同的初始化、顺序与预算；最小先2个 attack seeds，比较 attack 噪声与子集间信号，不够则增加。重新训练 J00/J10，最终输出 Δvalue=J10−J00。若攻击非驻定，用 continue2 Eq.42 的完整有限 Adam 轨迹或明确标记的近似，不能假定 implicit attack Hessian 自动准确。

完整阶梯固定为 L1=vᵀΔP_true、L2=hᵀΔθ_true、L3=Σδw_pτ_p，其中 v/h 来自同一 stable reference 和同一 attack seed，先逐 seed 配对再平均。公式必须先于确认真值冻结。保留 full/relabel 对照供另一个研究问题使用，不混入 value-pathway LDS。

**交付范围：**本包已经实现并可运行 Stage-1 LDS、L1/L2/L3 的 Stage-1 因子诊断、受控两阶段公式测试和可选真实 OCT 局部响应。完整跨阶段 attack J、完整轨迹 TracIn、全140轮CNN Adam反传没有在本包实现，不会将 Stage-1 结果冒充完整 Eq.79 的实证验证。上面是后续确认轮的冻结设计。

## 7. 参考与证据定位

- 用户提供 `continue_pathway-2.pdf`：§2前提；Eq.48接口链式法则；Eq.52–61驻点/删除符号；Eq.70–74完整Adam状态；Eq.75–79完整跨阶段公式；Eq.83–87连续值与重标记拆分。
- Park et al., TRAK, ICML2023，Definition2.2/2.3、AppendixC.4：<https://proceedings.mlr.press/v202/park23c/park23c.pdf>。
- Ilyas et al., Datamodels，2022：<https://arxiv.org/abs/2202.00622>。拟合 datamodel 需要与评估子集隔离，本轮 ridge 不是文献的大规模 oracle。
- Koh & Liang，IF，ICML2017，§2：<https://proceedings.mlr.press/v70/koh17a/koh17a.pdf>。局部驻点导数不能直接等同于任意冷启动训练结果。
- Shokri et al., MIA，2017：用户PDF `1610.05820v2.pdf`，<https://arxiv.org/abs/1610.05820>。原攻击框架与本项目新增的病人级连续归因假说分开表述。
- Pruthi et al., TracIn，2020：用户PDF `2002.08484v3-2.pdf`，<https://arxiv.org/abs/2002.08484>。完整checkpoint累计不等同于单点 frozen_h。
- Xia et al., LESS，2024：用户PDF `2402.04333v1.pdf`。优化器感知 proxy 不等同于对完整 Adam 状态的精确求导，不作为本公式已被证明的依据。

## 8. 成本与状态

旧 stable 单个完整训练约8–9分钟，24次按此粗算约3.2–3.6 GPU·h；subset步数更少、FP64评分另计，不能沿用未经核验的“24–48 GPU·h”。新作业有真实数据加载与数值审计，应以 smoke/score elapsed 校正预算。每个score和可选local作业上限3小时，是调度上限，不是预计耗时。

默认 concurrency=1。可选local在score完成后可与子集线同时跑，合计可能占2张GPU；不会抢占/取消其他作业。提高concurrency到2或4不会改变随机设计，但会增加同一时刻的资源占用。

本地已测受控数学、训练核、评分、no-op、两种分母、局部响应、收集统计、Slurm依赖与防重复提交。实际OCT数据、CUDA数值与运行耗时需要Delta smoke验证。本地没有提交GPU作业、没有推送GitHub。
