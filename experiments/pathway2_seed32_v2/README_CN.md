# Stage 1：32 seed、删除者 LDS、等大小子集、γ 扫描与逐查询 LDS（隔离实验 v2）

日期 2026-10-06。父提交 `c4880c9`（`exp/pathway2-lds-formula-v1`）。本目录只新增文件，不改 stable、LDS v1 或任何 legacy 代码；`design.py`、`analysis_stats.py`、`stable_core.py` 按字节复用，prepare 会核对它们与 LDS pilot 实际运行时的哈希一致。

对应会议纪要（2026-10-05）的任务 1–5：

| 任务 | 本包里的实现 | 输出 |
|---|---|---|
| 1 32 个 seed | 旧 8 个子集 × seed 42–73（42–44 直接复用旧模型），加 32 个全数据 baseline | `fig_task1_pilot8_*.png`：SE、可靠性（split-half + Spearman–Brown + 方差分量）、LDS 随 k |
| 2 删除者写法 | 预测 $\sum_{p\notin S_j}(-\tau_p)$；保留 matched（$\delta w=\frac{N}{N_j}m-1$）；等价性写进单测并在每次 collect 时复核 | `summary.json → task2_3.*.deleter_equivalence` |
| 3 子集大小控制 | 每个 LDS 都同时报告：图像数基线（事先固定符号 $-N_j$）、类别组成交叉拟合 ridge、扣除图像数后的 Spearman 和偏秩相关；新设计 `eq32`：32 个子集，按类别分层保留病人数，总图像数 ±2%、各类图像数 ±10%（拒绝抽样，约束内均匀） | `table_lds.csv`、`fig_task3_size.png` |
| 4 γ 扫描 | γ ∈ {1, 0.3, 0.1, 0.03}，3 个 score seed × 3 个目标。multi-shift Lanczos：$(H+\gamma I)x=h$ 对所有 γ 共享同一个 Krylov 空间，一次运行得到全部 γ 的解（与 CG 迭代等价，单测核验）；每个 γ 单独过门槛 | `fig_task4_gamma.png`、`table_gamma.csv` |
| 5 逐查询 LDS | 163 名查询病人逐个算 LDS 后平均（TRAK 写法），用梯度内积 proxy；置换零分布对全部查询使用同一个子集置换 | `fig_task5_per_query.png`、`table_per_query.csv` |

三个目标都算：`oneminusp`（1−p_y）、`brier`、`ce`，均为类别等权 → 病人等权宏平均（与 LDS v1 相同的冻结权重）。主终点 raw，EMA 只作稳健性对照。

## 关键门槛（任何一个不过，作业失败，不会静默继续）

1. **smoke**：用本包代码重训 LDS v1 的 `subset000_seed42`，raw/EMA 必须与旧模型逐位一致；随后在同一进程重训 seed42 全数据，raw/EMA/Adam 状态必须与冻结 stable baseline 逐位一致（同时证明一个作业里串行多次训练是确定性的）；在 GPU 上跑 multi-shift 与稠密解、CG 的对照；实测 fp64 HVP 耗时。
2. **复用（harvest）**：从旧 `final.pt` 重算 fp64 概率，与旧 `interface.npz` 比对（LDS v1 ≤1e-10，stable baseline 的 fp32 概率 ≤1e-4）。
3. **proxy**：逐查询 proxy 按宏权重重组必须等于宏目标 proxy（相对误差 ≤1e-8）；seed 42–44 的 CE proxy 必须与 LDS v1 的 `frozen_h_upweight` 一致（≤1e-6）。
4. **score**：γ=1 的 CE 分数必须复现 LDS v1 的 CG 分数（相对 L2 ≤1e-3 且 Spearman ≥0.999）；每个 γ：真实残差 ≤1e-3、求解 Krylov 空间里 $\theta_{\min}+\gamma>0$、随机起点 Lanczos 探针的最小 Ritz 值加 γ 大于其残差。γ=1 不合格则作业失败；更小的 γ 不合格就在报告里写"不可解"，**不按 LDS 挑 γ**。
5. 每个单元写在临时目录，完成后原子改名；重复提交会跳过已完成单元，超时的包可以直接补交。

## 在 Delta 上运行

```bash
# 1) 取代码（新 worktree，不动正在用的目录）
cd /u/yli103/oct2   # 你的主仓库
git fetch origin
git worktree add /u/yli103/oct2-exp-seed32-v2 exp/pathway2-seed32-lds-v2
CODE=/u/yli103/oct2-exp-seed32-v2/experiments/pathway2_seed32_v2
module load pytorch-conda/2.8

# 2) CPU 测试（登录节点可跑，约 2 分钟）
python3 "$CODE/test_experiment.py" && python3 "$CODE/test_gpu_path_cpu.py"

# 3) prepare（只读校验旧目录，冻结设计、seed、γ；不提交作业）
OUT=/u/yli103/oct2-pathway2-runs/A/seed32_lds_v2_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --out "$OUT"
#   可选：--eq-seeds 4（等大小子集只跑 4 个 seed，省一半）；--concurrency 16
#   小删除设计（每个子集只删 10 人）需要导师确认：--smalldel --mentor-note "..."（不能含尖括号）

# 4) 提交。建议先不提交 γ 扫描，等 smoke 给出 HVP 耗时再定 walltime
python3 "$CODE/run.py" submit --out "$OUT" --defer-score
cat "$OUT/smoke/result.json"        # timing.score_task_upper_bound_hours
python3 "$CODE/run.py" resubmit --out "$OUT" --only score --score-time 04:00:00

# 5) 查进度、补交超时或失败的包
python3 "$CODE/run.py" status --out "$OUT"
python3 "$CODE/run.py" resubmit --out "$OUT"              # 只补未完成的
python3 "$CODE/run.py" queue-report --out "$OUT"          # 排队时间 = Start − Submit
python3 "$CODE/run.py" queue-report --since 2026-09-28    # 会议要求：你自 9/28 以来所有作业的排队统计

# 6) 汇总。可以在部分完成时先看（只用全部子集都完成的 seed）
python3 "$CODE/run.py" collect --out "$OUT" --allow-partial
python3 "$CODE/run.py" collect --out "$OUT"
#   生成 $OUT/report*/（summary.json、REPORT_CN.md、表、图）和 $OUT.report*.tar.gz
```

pack 按 seed 先后排列：seed 42、43…依次整组完成，所以中途 `collect --allow-partial` 就能看到 k 较小时的曲线。

## 计算量（A40 实测：全数据约 7 分钟，半数据约 3 分钟）

| 部分 | 单元 | 估计 GPU 时 |
|---|---|---|
| 全数据 baseline（新 29 个） | 15 个 pack | ≈3.6 |
| 旧 8 子集 × 29 个新 seed | 47 个 pack | ≈12.8 |
| eq32 × 8 seed（`--eq-seeds 4` 减半） | 52 个 pack | ≈14 |
| proxy（32 seed，fp64） | 8 个任务 | ≈1–2 |
| γ 扫描（3 seed × 3 目标，每个至多 160 次 fp64 HVP） | 9 个任务 | smoke 给出上界，预计每个 2–3 小时 |

每个 pack 约 20 分钟，walltime 30 分钟以利用 backfill；数组并发默认 `%8`。

## 预期（按会议要求先写明）

- SE 一定随 k 下降。CE 的单 seed 可靠性约 0.47，32 seed 约 0.97；1−p_y 单 seed 约 0.75，几乎没有提升空间。
- 按衰减公式，32 seed 最多把 pilot8 的 CE LDS 从 0.286 抬到约 0.33。pilot8 只有 8 个子集，单侧 5% 门槛约 0.62，置信区间宽主要来自子集数，加 seed 解决不了；eq32 的门槛约 0.30。
- pilot8 上只用图像数就有约 0.667 的排序相关；扣除图像数后 IF 只剩约 0.119。eq32 把图像数差异压到 ±2%，这一混杂应基本消失。
- L2 用真实 Δθ 已经失败（−0.262），调小 γ 大概率救不了 50% 子集上的 LDS。γ 扫描的作用是说清楚 Hessian 有没有用：看 IF 与梯度内积的相关是否随 γ 变小而下降。

## 范围

- IF 在各 score seed 的 raw 全数据端点计算，端点不是驻点；这是阻尼端点近似，不是 Eq.56 的证明。
- 任务 1 的 IF 曲线固定用 3 个 score seed 的平均 τ，只让真值的 seed 数变化；proxy 曲线用与真值同一组 k 个 seed。
- 逐查询 LDS 只用梯度内积，没有逐查询的 Hessian 逆（163 个右端项的求解代价太高）。
- 不涉及 Stage 2 / 跨阶段 J、late-deletion、轨迹导数。

## 文件

`run.py` 控制器 · `core2.py` GPU 侧（训练/复用单元、proxy、γ 扫描、smoke）· `solver.py` multi-shift Lanczos 与门槛 · `targets.py` 三个目标及其导数 · `designs.py` 子集分布 · `stats2.py` 统计 · `analysis.py` 汇总 · `figures.py` 作图 · `report_md.py` 中文报告 · `formula_tests.py` 共用数值核验 · `test_experiment.py`（单元 + 合成端到端）· `test_gpu_path_cpu.py`（GPU 代码路径在 CPU 上的集成测试）· `job.slurm`。
