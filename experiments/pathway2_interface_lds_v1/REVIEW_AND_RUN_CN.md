# 接口概率 LDS：实验解释、修订和运行说明

日期：2026-10-22。修订分支：`exp/pathway2-interface-lds-v1r1`。

## 1. 导师的话是什么意思

导师认为，在多个随机种子下观察到 0.2–0.3 的相关性已经是值得继续研究的信号。他并没有说这个数值证明了方法足够准确，也没有要求把相关性提高到很高才有意义。

他建议先做一个更直接的消融：暂时不经过 membership attack，直接比较 CNN 四类概率变化的预测与真实重训变化。对于固定图像，假设完整模型输出 `[0.10, 0.60, 0.20, 0.10]`，子集模型输出 `[0.15, 0.50, 0.25, 0.10]`，真实变化就是 `[+0.05, -0.10, +0.05, 0]`。IF 要预测的是这四个变化，而不是四个原始概率或最终类别。

这样可以独立检验 Stage 1 的近似在接口处是否有预测力。“预计得到类似相关性”是导师的待检验预期，不是已经成立的结果。逐输出相关性的平均与 attack 加权标量的相关性是两个不同指标，数值相似不能证明全部跨阶段误差来自 Stage 1。

这里的 counterfactual 是“如果用该子集训练模型，这张固定图像的输出会怎样变化”。即使某张训练图像被删除，它仍然可以作为固定查询输入。此实验无需修改 attack 的 membership 标签，也不训练新 attack。

## 2. 保留的实验设计

| 项目 | 固定做法 |
|---|---|
| 数据和干预 | v3 的 32 个图像子集，每个约保留一半 shadow3 训练图像 |
| 接口 | 4343 行 × 4 类；其中 2045 个训练行、2298 个 nonmember 行 |
| 真值 | 已有子集模型与对应完整模型的概率差；256 个子集模型对应 32 子集 × 8 Stage-1 seeds |
| 主预测 | 与 v6 L3 同一模型、Hessian、梯度和阻尼定义的 IF |
| 主 γ | 0.03，结果出来后不替换；其他继承自 v6 的 γ 为补充 |
| 主统计量 | 每个 nonmember 行和类别，跨 32 子集算 Spearman，再在有效输出上取平均 |
| 推断单位 | 同一次置换或 bootstrap 对所有输出使用相同的子集索引 |
| 对照 | 去掉逆 Hessian 的 proxy；真实参数位移的线性化；logit 步进为补充 |
| 训练行 | 全部子集指标，以及该行被删除/被保留时的分组描述指标 |
| v6 衔接 | h̄、τ、L1、L3 数值复现；在 L3 中替换单个行组的概率变化 |

读取已有真实 Δp 不用 GPU，也不用重新训练那 256 个模型。IF 的 HVP、线性系统求解和 JVP 仍有计算成本。默认预算为 24 GPU 小时，是预算参数，不是已测得运行时间；默认单个 solve 的 Slurm 上限为 12 小时。smoke 会估计成本，先尝试全部 seeds，预算不够时固定前三个，否则停止。估计计入求解器的投影和正交化耗时及 25% 余量，也扣除本次 smoke 耗时，但后续更大子空间可能使实际成本继续增加。

这次复用已有子集，属于机制诊断。若要确认“小 γ 在新子集上更好”，仍应另设新子集实验。此主检验不含 attack，因此不需要 attack seed 交叉拟合。将来正式比较跨阶段分数与 attack 真值时，仍应保持两者 attack seeds 不相交。

## 3. IF 的符号与归一化

令完整训练集大小为 N，子集 j 保留 K_j 个训练样本。定义

$$g_i=\nabla_\theta CE_i/N,\quad b_j=\sum_{i\text{ 被删除}}g_i,\quad u=\sum_i g_i.$$

在完整和子集目标都采用各自训练集的平均 CE、相同正则项时，目标梯度差为

$$\Delta\nabla L_j=(N/K_j-1)u-(N/K_j)b_j.$$

相应的 matched IF 预测为

$$\widehat{\Delta\theta}_j=(H+\gamma I)^{-1}\big[(N/K_j)b_j-(N/K_j-1)u\big],\qquad
\widehat{\Delta p}_{j,r}=J_{p,r}\widehat{\Delta\theta}_j.$$

原代码已经对 IF 实现了这个修正，保留了它。此次给 proxy 增加相同修正，避免行组函数比较采用不同归一化。不要直接删掉 u 项：即使正则化总目标驻点成立，CE 梯度之和也未必为零。

如果所有子集的 K_j 相同，固定输出上的 matched 预测只是删除和预测的正比例缩放加固定平移，所以逐输出的跨子集 Spearman 不变。但是幅度、变动成员的行组平均以及与 v6 的标量对接都需要 matched 形式。

代码先储存 logit JVP，再使用

$$J_p d\theta=p\odot\big[J_zd\theta-\langle p,J_zd\theta\rangle\big].$$

四个分量之和应为零。该线性化是概率变化的切向量，不能对它逐类裁剪或重新 softmax 后仍称为同一个主 IF 预测。额外的 logit 步进单列为补充。

## 4. 已修订的问题

1. **预测与真值的模型 seed 面板对齐。** 原来预测降到三个 seeds 时，主真值仍平均八个。现在主检验两侧使用同一固定面板，八 seed 真值另作补充。行组替换及其 attack 真值也使用对应 Stage-1 面板。
2. **不再把 split-half 称为严格噪声上限。** 八 seeds 枚举全部 35 种不同的 4 对 4 拆分。奇数面板使用平衡的互斥半组，并正确记录外推倍数。报告 Spearman 可重复性、Pearson 条件性衰减参考及其有效输出数。负可靠性不裁剪成零后冒充可达上限。
3. **退化统计量不产生假显著性。** 没有有效输出时，统计量、p 和 CI 为未定义。拒绝非有限或形状不匹配的输入。置换比较考虑浮点误差，计入相同统计量的排列。
4. **bootstrap 固定输出集合。** 原来每次重采样用 nanmean，可能悄悄丢掉不同输出。现在如果重采样使原有有效输出变成常数，整次抽样记为未定义并报告次数；有效比例不足 95% 时不报 CI。提供的区间条件于有效抽样。额外的“自身删除效应”排名不再按图像独立 bootstrap，因为图像共享患者及子集。
5. **数值门槛参与结果资格。** 零右端项合法；辅助 γ 的奇异投影不使其他 γ 一起崩溃。每个右端项必须达到该 γ 预设残差阈值，且合格 v6 参考下的 τ 和 matched L3 对接必须通过，否则该 γ 标为不合格。主 γ 不合格时不换成另一个 γ。
6. **运行和输入检查。** 增加数组布局、概率有效性、γ 顺序检查。限制 pilot 子集数。修复 Slurm 时间可能出现 `01:60:00`。计时纳入实际求解开销，预算仍明确标为估计。使用新 schema 和新输出前缀，避免混入原版中间结果。
7. **收紧解释。** v6 历史桥接使用原 attack seeds，不作交叉拟合显著性宣称。行组替换的 Spearman 改善不能相加为误差占比。真实 Δθ 线性化表现差，只提示局部线性化可能不足，不能单独证明 Hessian 是唯一原因。

联合置换保留四类概率和所有行之间的依赖。它检验的是子集配对可交换的零假设，不能把 2298×4 个输出当成同样数量的独立实验。Monte Carlo p 使用 `(超过或等于观测值的次数 + 1)/(置换次数 + 1)`，本次默认 2000 次。非主 γ、分组和对照的 p 仅作补充，不据此另挑“成功”的主结果。

置换实现参考：[Phipson and Smyth (2010), Permutation P-values Should Never Be Zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)。

## 5. 如何理解结果

| 观察 | 可以支持的解释 |
|---|---|
| IF 主相关性稳定为正 | 对当前固定接口和种子面板，Stage 1 的概率变化排序有信号 |
| IF 与 v6 L3 数值接近 | 与导师预期一致，但不是两个统计量相等或误差已被唯一定位的证明 |
| IF 排序不错，幅度斜率或范数比差 | 可能能排相对顺序，却不能准确预测变化大小 |
| 真实 Δθ 线性化接近真实 Δp，而 IF 明显差 | 提示应进一步检查 IF 的参数位移近似、阻尼和训练轨迹 |
| 真实 Δθ 线性化也差 | 输出函数对有限参数位移的非线性是需要检查的因素 |
| 替换某组行后 L3 接近 L1 | 在当前固定伴随权重下，修正该组能改善标量排名 |
| split-half 真值可重复性低 | 当前有限种子真值本身的排序不稳定，不能简单把低相关全归为 IF 错误 |

Spearman 不衡量概率校准或变化幅度。这里的 CI 条件于固定图像、患者和训练 seeds，仅覆盖子集重采样；它不是新患者、新训练种子的总体置信区间。

## 6. 在 Delta 安装与运行

下载修订 bundle 到本机后，上传到 Delta 的 home。下面的 bundle 以 v6 r1 提交 `a29971490eeaa5cfc26322bc12b86241d15a6249` 为前提。不要把 bundle 当成 ZIP 解压。

在已有 `~/oct2` 仓库中导入到独立 worktree，保留原实验目录：

```bash
cd ~/oct2
git bundle verify ~/oct2_interface_lds_v1.bundle
git fetch ~/oct2_interface_lds_v1.bundle exp/pathway2-interface-lds-v1r1
git worktree add ~/oct2-interface-lds-v1r1 -b exp/pathway2-interface-lds-v1r1 FETCH_HEAD
cd ~/oct2-interface-lds-v1r1
git status --short
```

若这条分支或目录已存在，先检查是否就是同一个版本，不要强制覆盖。运行入口要求干净 worktree，并核验 v3/v4/v5/v6 及审计输入。正式运行目录必须是新的，不能沿用原版 interface run 的 config 或 solve 缓存。

下列命令会准备新目录并提交 smoke、solve、collect，沿用脚本已设的 v6 current 路径、账户和默认分区：

```bash
bash experiments/pathway2_interface_lds_v1/start_delta.sh
```

如果需要指定已完成的 v6 r1 目录或预算，使用：

```bash
bash experiments/pathway2_interface_lds_v1/start_delta.sh \
  /实际的已完成v6目录 \
  --primary-gamma 0.03 --budget-gpu-hours 24 --solve-hours 12
```

`/实际的已完成v6目录` 是占位符，必须替换后再运行。默认继承 v6 的 γ 网格，不能临时添加不在该网格中的 0.001。若将来研究新 γ，应另立实验。

查看进度：

```bash
out=$(cat /u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_current.txt)
python3 experiments/pathway2_interface_lds_v1/run.py status --out "$out"
```

仅重提未完成任务：

```bash
python3 experiments/pathway2_interface_lds_v1/run.py resubmit --out "$out"
```

重提 solve 后，原自动 collect 依赖旧作业，可能不会自动续上。待全部 seeds 完成，再在 CPU 计算节点运行 `collect`。若旧 collect 仍在排队或运行，先用 Slurm 状态确认，避免重复收集。

```bash
python3 experiments/pathway2_interface_lds_v1/run.py collect --out "$out"
```

返回的 `*.report.tar.gz` 包含中文报告、summary、表格、配置和日志。大体积的 `interface.npz` 留在 Delta。不要把部分结果当成主检验通过；`collect --allow-partial` 会明确标记缺失并禁止主阈值判定。

## 7. 验证与范围

本次验证环境为 Python 3.12、PyTorch 2.5.1 CPU。19 项单元测试和 1 项完整 CPU 端到端测试通过，合计 20 项；端到端测试约 22 分钟。另通过 Python 语法、Shell 语法及 Git diff 空白检查。

合成测试使用两个 Stage-1 seeds、八个子集，测试 γ 为 1、0.3、0.1，主测试 γ 为 0.3；这是数值实现测试的参数，不改变正式实验主 γ=0.03。实测核验误差为：

| 核验 | 合成测试实测误差 |
|---|---|
| smoke 的 v̄ VJP 对 h̄ | 相对误差 0 |
| smoke 的 τ 对 v6 风格参考 | 各 γ 最大相对 L2 误差 2.08×10⁻⁸ |
| seed42 的前向/反向恒等式 | 最大相对误差 1.08×10⁻¹⁴ |
| L1 的最终聚合复现 | 最大绝对误差 5.55×10⁻¹⁷ |
| L3 的最终复现 | 最大相对误差 9.30×10⁻¹³ |

因此应分别报告恒等式误差和迭代求解误差，不能把全部核验统一写成 10⁻¹² 到 10⁻¹⁶。这里的 τ 误差通过了测试要求的 10⁻⁶ 一致性检查；生产运行仍使用配置中冻结的数值门槛。

单元测试命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover \
  -s experiments/pathway2_interface_lds_v1 -p test_unit_if.py -v
```

完整 CPU 合成流程命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover \
  -s experiments/pathway2_interface_lds_v1 -p test_gpu_path_cpu_if.py -v
```

完整测试覆盖构造 v3/v6 风格输入、原 CNN 和复用数值内核、smoke、所有 seed 的 solve、collect、h̄/τ/L1/L3 对接、缩小 seed 面板后的分析及报告生成。测试数据是合成的小数据面板，不是正式 OCT 结果，也不能替代 Delta 上的 CUDA smoke。

本次修订未提交 GPU 正式作业，未产生新的 OCT 实验结果。
