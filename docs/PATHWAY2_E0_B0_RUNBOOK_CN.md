# E0/B0 修正版操作说明（2026-09-17）

本轮只运行 E0 和 B0。B1/B2 与 A 尚未实现，也没有在本环境提交 Delta 作业。基线始终来自 ba8db97 的原结果，50 epochs 从原配置读取；没有新增“训到收敛”的主实验。

## 这次修了什么

- 原冻结清单不动，另列 B 分支源码清单，解决 05 的参数筛选改动与 preflight 哈希冲突。原目录仍按原哈希验收。
- 05 加 `--loo_patients` 后保留旧配置的默认恢复兼容；10 的 early/late 窗口不再被 patient 筛选分支跳过。训练函数、初始化、batch/随机流和优化器代码不变。
- B0 使用原 05.train_classifier_from_orders。每 seed 先完整重放一次，验证原 checkpoint 参数、各保存 epoch 的 Adam/RNG 和接口逐位相同，再训练 4 剂量 × 2 patients。共 16 次扰动 + 2 次 no-op；不重复训练 target、其他 shadow 或 attack。
- 小剂量文件标注 `stage1_probe_complete`，用于 Stage-1 导数诊断，不能当作已经重跑的 J10/J11 truth。
- 允许原五个 seeds 的目录与新单个 seed 的目录比较，但验证所请求 seed、配置、实际 split/order、接口行与 baseline 一致。
- h 按 Stage-1 seed、病人类别与 attack seed 匹配，验证原 baseline/split 与求解资格；取消跨类别平均。
- 判据按 cell 和连续区间报告。一个 cell 或两个分开的通过区间不能代表整个面板通过。
- E0/B0 在异常退出时也复核实际输入哈希；Slurm EXIT trap 再核对冻结原目录。检测输入变化必须以非零退出码失败。
- 输出含 job id，拒绝覆盖已有 run。任务锁定提交时的代码 commit。日志目录在 sbatch 前建立。

这些是文件完整性与流程防护，不是操作系统级只读挂载。历史 JPEG 字节哈希不在已有资料中；本轮记录实际加载的 X/y 摘要，要求 E0/B0 一致，并用原训练逐位重放验证使用的数据与训练行为。

## 安装、发布、提交

使用修正版 ZIP 内的 `install_reviewed.py`，兼容未建分支、已装 e3fd4b1 协议版、已装 93bb06b 原上传版。它仅对两个实验分支执行创建或快进，拒绝未知分叉和脏工作树；不 checkout、reset、merge 原 baseline 或 main。

Mac：

```bash
scp ~/Downloads/oct2_pathway2_e0_b0_reviewed_20260917.zip yli103@login.delta.ncsa.illinois.edu:~/
```

Delta：

```bash
oct2_setup_dir=$(mktemp -d "$HOME/oct2-e0b0-reviewed.XXXXXX")
unzip -q "$HOME/oct2_pathway2_e0_b0_reviewed_20260917.zip" -d "$oct2_setup_dir"
python3 "$oct2_setup_dir/oct2_pathway2_e0_b0_reviewed_20260917/install_reviewed.py" \
  --repo "$HOME/oct2-calibration-v4" --worktree-parent "$HOME" --push

cd "$HOME/oct2-exp-b"
export OCT_BASELINE_ROOT="$HOME/oct2-calibration-v4"
export OCT_RUNS_ROOT="$HOME/oct2-pathway2-runs/B"
export OCT_DATA_DIR=/u/yli103/oct2/data
bash scripts/cross_stage/submit_e0_b0.sh
```

`--push` 用 Delta 上已有 GitHub 凭据发布两个实验分支；无凭据会明确失败，本地安装仍保留。确认安装成功后可用 `git push origin exp/pathway2-a-stationarity exp/pathway2-b-trajectory` 发布；不使用 force。此交付不声称远程已经发布。

提交会打印 `GPU_TEST`、`E0`、`B0_ARRAY`，并写入 `$OCT_RUNS_ROOT/submission_<GPU_TEST>.txt`。测试使用 PyTorch module `pytorch-conda/2.8` 与实际 CUDA；测试失败时后续依赖任务不会运行。

| 任务 | GPU | 工作量 |
| --- | ---: | --- |
| CUDA tests | 1，先执行 | E0/B0 新回归 + 原 v1.1 回归 |
| E0 | 1 | seeds 42/43 × patients 807/2085 × alpha 1/.1；原 8 患者信号监测 |
| B0 array task 0 | 1 | seed 42；一次 no-op + 8 个扰动 |
| B0 array task 1 | 1 | seed 43；一次 no-op + 8 个扰动 |

测试之后最多同时 3 GPU。walltime 是首跑资源上限，不是耗时预测；以实际记录调整后续预算。B0 结束后自动分析，不需要另起 GPU 作业。

默认 h 路径为原目录下 `results/stage1_diagnostics_v11/damping_22122145/h_seed{42,43}.pt`。若该目录已搬迁，提交前设置 `OCT_H_DIR` 指向真实原缓存目录。缺失会停止，不能伪造或用其他类别探针替代。

## 查看、验收和下载

用提交输出的实际数字替换下列占位符：

```bash
squeue -u "$USER"
sacct -j <GPU_TEST>,<E0>,<B0_ARRAY> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

cd "$HOME/oct2-exp-b"
python3 -B scripts/cross_stage/collect_e0_b0.py \
  --submission "$HOME/oct2-pathway2-runs/B/submission_<GPU_TEST>.txt" \
  --output "$HOME/oct2_e0_b0_review.tar.gz"
```

要求作业 COMPLETED/0:0。收集器检查任务后哈希、E0 闭合与 8-cell/16-monitor 覆盖、B0 的逐位重放、每 seed 的 12 行六档剂量（含原 1/.1）覆盖、commit 和实际加载数据一致性；满足后才打包。`resolved_band=false` 是可以接受的科学结果，不算执行失败。

默认包包含 JSON/CSV/日志，不包含 `.pt`/`.npz` 大文件；这些仍保留在 Delta。如需下载原始新 checkpoint，可在收集命令追加 `--include-checkpoints`，并换一个新输出文件名。

Mac 下载：

```bash
scp yli103@login.delta.ncsa.illinois.edu:~/oct2_e0_b0_review.tar.gz ~/Downloads/
```

## 如何读结果与推进 B1

E0 的 `share_stationarity`、`share_remainder` 是 gamma=0 时沿残差的带符号投影；可为负或大于 1。它们不是独立因果比例，也不能仅凭哪项更大就宣布唯一病因。闭合主要检查实现一致性，不能独立证明 HVP 正确。

B0 每个 cell 的预注册工程门槛：连续两对相邻剂量至少三个点，`1-cos<=0.05`、导数相对变化 `<=0.05`，两端位移均超过保守 float32 分辨率尺度的 100 倍。尺度是启发式，不是测得的随机噪声。各 cell 的通过区间分别给出；全部 cell 各自通过才令本任务总体 `resolved_band=true`。P 与 h 投影另列诊断，参数空间稳定不能代替它们稳定。任何通过都不证明 alpha=1 可线性外推。

B1 下一轮先实现短链：含病人样本的一个 batch → 1 epoch → 5 epochs，每段配对应窗口的 truth。验收包括 primal 参数/m/v/step/RNG 对照、非零方向 finite-difference 步长阶梯、coupled-wd/非零 moments 情形、零矩坐标有限性，以及不存在病人的全程有限且逐位零 tangent。零方向只能检查零注入不变量，不能独立查全 Jacobian、dropout 与 wd 错误。

JVP 和有限差分互补；ReLU/max-pool 存在分支变化，不能凭一个步长差分失败就否定全部有限差分验证。全轨迹导数应保留原 800 steps，传播完整数值 tangent 并释放已完成步骤的图。

B1 开发可在独立 worktree 中进行，避免改变正在排队或运行的 B 工作树：

```bash
git -C "$HOME/oct2-exp-b" worktree add -b exp/pathway2-b1-smoke "$HOME/oct2-b1-smoke" HEAD
```

该工作树用于开发；本轮提交脚本仍只从 `exp/pathway2-b-trajectory` 运行。A 保持可选机制诊断，B2 必须同时接入 Stage-2 式 (42) 才是严格有限轨迹版。
