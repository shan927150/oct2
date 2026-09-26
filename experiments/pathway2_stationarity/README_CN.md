# Route A：同目标驻点与患者扰动审计

## 依据与本次问题

`continue_pathway-2.pdf` 的式 (52)–(56) 从驻点条件推导
`d theta*/d epsilon = -H^{-1}g_i`。H、样本梯度和驻点必须属于同一个目标。
式 (70)–(74) 的 Adam 路径还需要对参数、动量、二阶矩和步数状态求导。

现有 frozen loss audit（2026-09-26，seed 42/44，control/decay，E120/E140）
重复 eval 完全一致，而固定端点 dropout loss 有抽样波动；不同端点的期望 loss
也有变化。因此参数变化和测量波动可以同时存在。仅凭这些结果不能确认收敛，
也不能把在线 loss 的总方差定量拆成两部分。

已有 E140 dropout MC 计算了全梯度，本次新增的是**同一随机样本中的患者贡献**，
以及扰动后的梯度、它们的协方差和 MC 精度。不是再跑一次 loss 曲线。

## 固定设计

| 项目 | 默认 pilot | 可选 matrix |
|---|---|---|
| baseline seed | 42、44 | 42、44 |
| arm / epoch | decay / E140 | control、decay × E120、E140 |
| checkpoint 数 | 2 | 8 |
| patient | 807（6 张）、2085（5 张） | 相同 |
| 删除剂量 alpha | 0.1、1.0 | 相同 |
| dropout 抽样 | 每端点 128 次 | 每端点 128 次 |
| 数值 batch | 64，遍历全 2045 张训练图 | 相同 |
| L2 | lambda=1e-5，包含所有参数 | 相同 |

保留训练行、原始 N、类别和 membership 标签。这里只读 baseline，不训练 attack。
E135 没有保存 checkpoint；代码仅接受 E120/E140。
默认 pilot 只代表两 seed、两患者，不外推为全 40-cell 结论。

## 公式与输出含义

对 eval 目标和 dropout 期望目标分别定义：

```
F(theta) = (1/N) sum_i CE_i(theta) + lambda/2 * ||theta||^2
g        = grad F(theta)
b_p      = (1/N) sum_{i in patient p} grad CE_i(theta)
F_alpha  = F - alpha * (1/N) sum_{i in patient p} CE_i
grad F_alpha(theta) = g - alpha * b_p
rho_alpha = ||g|| / (alpha * ||b_p||)
```

删除对应负的 loss 权重扰动；若驻点与 Hessian 条件成立，预测位移是
`+alpha H^{-1} b_p`。本次不求 H，也不输出“预测 vs 真实”cosine。
输出的 `cosine_g_b`、`cosine_dropout_eval` 均有独立含义，不得混用。

- `b_p` 不含 L2；分母不是患者图数，也不是 N-n_p。
- 同一 draw 中 g 和所有 b 共用 logits/dropout masks。
- 输出 `||mean g||`，不是 `mean ||g||`；保留向量供后续独立检查。
- `g_vector_se` 是 `sqrt(trace(sample covariance)/R)`，表示均值向量的
  RMS 抽样误差估计；不是置信半径、p-value 或“几 sigma”。
- `g-alpha b` 的 SE 包含 g/b 协方差，不能按独立样本误差相加。
- 16/32/64/128 次前缀用于观察精度变化，彼此不是独立实验。
- 小 b 若尚未超出 MC 误差，rho 不可靠。`norm²-SE²` 可以为负；不强行截成零。
- rho 小只能提供原始梯度尺度上的证据；它不是驻点、Hessian 可逆性、
  局部性或影响函数有效性的充分条件。Hessian 条件数仍然重要。
- **不得把 dropout 的 g/b 与旧 eval Hessian 混算。**
- dropout 期望 loss 是目标层面的诊断，不等于完整 Adam 训练轨迹导数。

## 隔离方式

以 `ca332634a5f50860cea1eb9c1839755232b9434b` 为 base 创建独立 worktree。
所有新增文件仅位于 `experiments/pathway2_stationarity/`。
安装不 checkout/reset/merge 旧分支，不改旧脚本、基线 checkpoint 或旧 run 文件。
共享 Git object/ref 数据库只新增本实验的 commit/branch/worktree。

每次结果写到新 `objective_stationarity_*` 目录；拒绝覆盖既有目录或输入树。
prepare 锁定配置、代码和 checkpoint 的 SHA256，并与已完成 loss audit 的
checkpoint SHA256 对照；计算前后复核。加载数据后核验完整数组 fingerprint。
内存 model state、RNG、模式及已有 p.grad 均须保持不变。

这提供文件/分支隔离，不消除共享 GPU 队列资源竞争；数组 `%1` 限制并发为 1 GPU。

## 安装后运行（Delta 登录节点）

```bash
CODE=/u/yli103/oct2-exp-a-objective-stationarity/experiments/pathway2_stationarity
bash "$CODE/launch.sh" pilot
```

这条命令才会申请 GPU：先 smoke，再 afterok 数组。安装脚本本身不提交作业。
保存终端打印的 OUT、SMOKE、ARRAY。默认两 cell；无需运行 matrix 来完成 pilot。
不使用 `rg`。

更细的分步控制：

```bash
CODE=/u/yli103/oct2-exp-a-objective-stationarity/experiments/pathway2_stationarity
OUT=/u/yli103/oct2-pathway2-runs/A/objective_stationarity_pilot_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --profile pilot --reps 128 --out "$OUT"
python3 "$CODE/run.py" submit --out "$OUT"
```

prepare 不需要 PyTorch，也不申请 GPU。支持 `--smallpanel` / `--loss-audit` 明确指定
输入目录；默认绑定本次已完成的两个目录。输出必须是新目录。
脚本使用 Delta `pytorch-conda/2.8`，账号从 smallpanel 配置读取，gpuA40x4，1 GPU，48G。
30 分钟 smoke / 每 cell 1 小时时限是调度上限，不是运行时间预测。

smoke 强制运行数值测试，再在真实 seed42 端点执行：重复 eval、原实现梯度对照、
历史 eval CE 复核、eval/dropout 下两患者两剂量的直接 weighted-loss 梯度恒等式、
输入/model/RNG/模式/p.grad 不变检查。任何失败会阻断数组；先看 logs，不跳过检查。

```bash
# OUT 替换成 launch 实际打印的完整路径；跨终端需要重新赋值
SMOKE=$(cat "$OUT/smoke_job.txt")
ARRAY=$(cat "$OUT/run_job.txt")
sacct -X -j "$SMOKE,$ARRAY" --format=JobID,JobName,State,ExitCode,Elapsed
```

submit 有独占 ledger，重复调用同一 OUT 不会再次提交。若提交中途失败，保留的
`submission.jsonl` / `smoke_job.txt` 可追踪已有作业；先检查这些作业，再新建 attempt。
launch 每次都会创建新 attempt；不要重复调用它来“查看状态”。

## 收集与下载

待所有作业 COMPLETED、ExitCode=0:0：

```bash
python3 "$CODE/run.py" collect --out "$OUT"
```

打印 `$OUT.tar.gz`，包含配置、日志、检查和结果；大向量 `.npz` 留在 Delta。
主表 `report/stationarity_summary.csv` 和说明 `report/REPORT.md`。
在本地电脑终端运行（主机名沿用你平时 SSH 的完整地址）：

```bash
scp yli103@你的Delta登录地址:/u/yli103/oct2-pathway2-runs/A/实际OUT目录名.tar.gz .
```

## 结果驱动的后续步骤

1. MC 相对误差大：新 attempt 增加 reps 到 256；不把均值小解释为已收敛。
2. 残差明显大于删除扰动：再设计独立的同目标 polish，并重新审计。
3. 残差足够小且精度足够：下一步局部 fixed-mask 剂量扫描，同时检查 Hessian
   与患者梯度目标一致。冷启动完整 Adam LOO 是另一种被估量，不能静默替换。

本包实现和提交范围到本次审计为止；不自动触发 polish、LOO 或 damping。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 "$CODE/test_audit.py"
# Delta smoke 强制加 --require-torch，不能跳过 PyTorch 数值测试
```

覆盖独立 softmax 解析梯度、中心有限差分、原始 N 与 L2、同 mask 剂量恒等式、
RNG/model/p.grad 保持、协方差 SE、零均值高不确定性、输入树保护、依赖调度与防重复。
具体本地验证结果见分发包的 VALIDATION_CN.md；真实 Delta 检查以 smoke 为准。
