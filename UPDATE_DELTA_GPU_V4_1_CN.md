# v4.1：修复 Delta GPU 复现失败并同步 GitHub

日期：2026-09-10。基础提交：`0feb6c4884d5b89c564ae07aa9815f9370ab76fe`。
分支继续使用 `cross-stage-calibration-v4`。

**需要把代码修复同步到 GitHub，再让其他工作目录拉取同一提交。**
你提供的 `21930537` 作业在数学测试 E 失败；日志没有显示已启动正式 OCT 实验。

**报错原因与修复**

日志中的 `adaptive_avg_pool2d_backward_cuda` 警告与后面的逐位相等断言有关：旧版 `warn_only=True` 允许非确定性 CUDA 反向传播继续运行。PyTorch 2.8 官方将 CUDA AdaptiveAvgPool2d 的反向传播列为不支持严格确定性的操作：[PyTorch 文档](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html)。

另外，旧版 fixed_mask 使用逐样本 CE 求和再除以 batch 大小，filter_rechunk 使用 CE 的内置 mean。这两种浮点计算可能不逐位相同。仅凭这份日志不能分离两者各自贡献，所以两处都修复：

- `SmallCNN` 保留 adaptive pooling 的分箱与输出尺寸；可整除时使用无重叠 AvgPool2d，其他尺寸使用切片求均值。128×128 OCT 输入和 16×16 合成测试都支持；池化没有可训练参数。
- 两种 Stage 1 训练模式共享相同的 loss reduction。fixed_mask 删除仍保持原 batch 分母和 dropout 配对。
- 启用严格确定性，包含从 checkpoint 恢复时的前向计算。Slurm 在 Python 启动前配置 CUBLAS，并要求 CUDA 可用。
- 保留严格的 `torch.equal` 与 no-op gate；添加同模式重复训练、128×128 训练以及池化梯度/HVP/JVP 检查。
- 05/07 拒绝混用旧训练数值版本的 checkpoint/truth。默认结果根目录改为 `results/cross_stage_calibration_v4_1`。

`cray-mpich` 的模块提示不是这次 Python 断言失败的直接原因；程序已经进入训练测试。当前修复不调整实验假设、seed 面板或 CE/AUC 指标。

**本地验证范围**

PyTorch 2.8.0+cpu、Python 3.12：8 项数学检查、3 项池化/复现/兼容性检查、7 项统计分析检查通过。full、dose=0.25、late-window 三组 `08 → 05 → 07 → 09/06` 合成链通过。还检查了新目录恢复、旧 truth 拒绝，以及 GPU 作业不能静默落到 CPU。

本地没有 CUDA。必须在 Delta 上通过下述 tests 和 smoke 才开始真实 OCT 训练；CPU 通过不能替代 GPU 验证，也不代表 score 已有预测力。包内 `validation/` 保存这些本地证据。

**1. 更新已有的 Delta 目录**

如果修复提交已经在 GitHub，在 Delta 执行：

```bash
cd ~/oct2-calibration-v4
git status --short
git switch cross-stage-calibration-v4
git pull --ff-only origin cross-stage-calibration-v4
git log -1 --oneline
```

`git status --short` 若列出了你改过的源文件，先保留这些改动并检查差异；不要用 reset/覆盖来消除它们。数据符号链接等未跟踪文件不需要提交。

如果修复尚未推送，使用附带的增量 bundle。在 Mac 下载 `cross_stage_gpu_fix_v4_1.zip` 后执行：

```bash
cd ~/Downloads
unzip cross_stage_gpu_fix_v4_1.zip
cd cross_stage_gpu_fix_v4_1
shasum -a 256 -c SHA256SUMS
scp cross-stage-calibration-v4_1.bundle yli103@login.delta.ncsa.illinois.edu:~/
ssh yli103@login.delta.ncsa.illinois.edu
```

在 Delta 执行：

```bash
cd ~/oct2-calibration-v4
git status --short
git switch cross-stage-calibration-v4
git bundle verify ~/cross-stage-calibration-v4_1.bundle
git fetch ~/cross-stage-calibration-v4_1.bundle cross-stage-calibration-v4
git merge --ff-only FETCH_HEAD
git log -1 --oneline
git push -u origin cross-stage-calibration-v4
```

bundle 已包含提交，所以不需要重新 `git add`/`git commit`。它是基于 `0feb6c4` 的增量包，适用于你已安装的 v4 仓库。`git merge --ff-only` 会在历史分叉时停下；不使用强推。

这样 Delta 当前代码和 GitHub 就使用同一修复。其他 Mac/Delta checkout 再执行 `git pull --ff-only origin cross-stage-calibration-v4`。若推送遇到身份认证问题，代码已经合并在 Delta，本地 tests 仍可继续，远端同步需使用你正常的 GitHub SSH/PAT 登录方式。

**2. 重跑 Delta GPU 检查**

```bash
cd ~/oct2-calibration-v4
mkdir -p logs
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm tests
```

用新返回的 job ID 替换 `JOB_ID`：

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 70 logs/oct_cal_v4-JOB_ID.out
tail -n 70 logs/oct_cal_v4-JOB_ID.err
```

通过条件：Slurm `COMPLETED` / `0:0`；数学测试 `all checks passed`；复现与统计测试均为 `OK`；输出有 `training_device=cuda` 和 `128x128 exact training replay passed on cuda`。

unittest 的正常 `OK` 也写入 stderr，所以 `.err` 非空不等于失败。

随后运行合成链，检查 GPU 上 attack、Hessian、JVP、配对与汇总能够一起工作：

```bash
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm smoke
```

确认日志为 `SYNTHETIC CHAIN PASSED` 且作业成功。短合成训练的 Hessian/score gate 失败可以是预期结果；脚本会核对这些 gate 的处理方式，不能据此判断真实 OCT score 质量。

**3. 开始真实 OCT 校准**

每一步完成并检查结果后再提交下一步：

```bash
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm preflight
```

检查 `results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json`：`panel_complete: true`、`shortfall: {}`。

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm baseline
```

检查 `results/cross_stage_calibration_v4_1/shadow*_full/` 下的 `attack_gate_summary.json` 和 `no_op_replays.json`。通过后：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm truth
```

truth 完成后：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm score
```

详细的 5×5 crossed seeds、CE/AUC 方差、score coverage 和 damping 解读见 [START_DELTA_GITHUB_V4_CN.md](START_DELTA_GITHUB_V4_CN.md)。如果曾另外跑过旧版 baseline/truth，保留作记录，在 v4.1 新目录重新生成；不要复制旧 checkpoint 到新目录。
