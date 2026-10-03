# Late-deletion v2，条件晚期降权 pilot

新分支 `exp/pathway2-late-deletion-v2`，仅新增本目录，基于 c4880c9 的已审计 LDS/stable 代码。正式运行复用旧数据划分、epoch 顺序及 stable 配方，dropout=0、Adam lr=0.001/wd=1e-5/batch128、10% step warmup + cosine、EMA=0.99、140 epochs。

## 固定的首轮设计

- seed42、patient807。prepare 可显式改为既有 seed43/44 或 patient2085；每次必须新输出目录。没有自动扩大矩阵。
- T0={0,60,100,130}；alpha={0.01,0.1,1}。alpha=0 完整状态 replay 在每个 T0 的任务内执行。
- 从 T0 起患者权重变为 1-alpha，保留原 rows、batch 分母、全局 step 和日程。患者在 T0 之前已参与训练，因此这是条件续训响应，不是从未见过该患者的完整 LOO。
- 每条分支同时保存 T0+10 和 E140 状态/查询预测，绘制固定终点与固定 10-epoch 窗口曲线。T0=130 两者重合；固定窗口仍有学习率差异。
- raw/EMA 参数几何，固定 heldout query 的 probability TV、CE、Brier、accuracy、1-p_true。查询为 patient/class macro，CE 从 FP64 logits 计算，概率不裁剪。query 向量几何按 sqrt(weight) 加权。
- 本包不计算静态 H^-1b，不实现 finite-unrolling tangent 或 pulse 实验，不训练 attack。保存的 checkpoint 可供后续独立公式/机制对照；不能把本次曲线解释为混沌确认或 Eq.56 证明。

## 正确性门

1. 新鲜 smoke：真实 CNN、4 epochs；每个 T0={0,2,4} 完整状态 replay，与继承 stable 核逐位一致、分支逆序/重复一致、无患者方向为零、T0=T 为零。
2. 140-epoch baseline 的 raw、EMA、Adam state 必须逐位复现旧 stable baseline。失败保留诊断输出，不放行分支。
3. 每个 T0 对完整 raw/EMA/Adam moments/step/RNG 等做 alpha=0 replay，并检查源快照内容和文件 hash 不变。
4. 每个完成标记绑定 config SHA 和输出 SHA。后续任务读取并验证工件，而不只依赖 Slurm 退出码。

Adam restore 对优化器 state 深拷贝，修复原包的张量别名污染。所有 snapshot 在 CPU 上落盘，参数按名称和注册顺序对齐。查询读取不改变训练随机流。原上传包不被覆盖。

## 提交阶段

`submit` 只提交 fresh smoke -> baseline -> T0=130 pilot，使用新 job IDs 的 afterok。`submit-rest` 在上述工件合格后提交 T0=100/60/0，默认 `%1`，不依赖已经过期的历史 job ID。可在 prepare 时预设 concurrency=2。

所有 receipt 独立写入并 fsync。提交失败或结果未知时停止，禁止盲目自动重试；`debug` 可打包现有日志。源码/配置冻结后勿手工编辑正在运行的分支，修复应另建输出和提交版本。

## 资源与输出

- 单任务 1 GPU、8 CPU、48G。smoke 上限1h、baseline/pilot各2h、其余每个T0上限4h，均为限额，不是耗时预测。
- 核心训练量为 baseline140 + 4×(140+80+40+10)=1220 epoch-equivalents，加 smoke 和查询前向。无 CG/HVP 全数据求解。实际GPU·h以作业结果为准。
- baseline 保存每个 epoch 的 raw/EMA 状态供距离轨迹计算，另存关键 epoch 的完整恢复状态，预计每 seed 约1GB量级。每个分支只存两个端点及查询数组。
- `report/geometry_all.csv`、`targets_all.csv`、`trajectory_all.csv`、`summary.json`、六格 PNG/PDF。单 seed/patient 不给虚假的置信区间。
- tar 用 Python 标准库生成，保留 csv/json/npz/日志/图，排除 `.pt`。完成标记中仍列出 Delta 端 checkpoint hash，因此下载的轻量包不充当可恢复训练快照。
- 参数 ULP norm 是分辨率诊断，不是严格误差界。零位移 R/cos 为 NA；接近分辨率的非零比值需要谨慎解释。

安装包里的 `START_HERE_CN.md` 提供 Mac、Delta 与 GitHub 命令。
