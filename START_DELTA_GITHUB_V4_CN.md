# OCT 跨阶段实验 v4.1：上传与启动

更新：2026-09-10（v4.1 GPU 复现修复）。基于交付的 v3 分支 `e87f42d`，新分支为 `cross-stage-calibration-v4`。

先跑小面板的真实 OCT 校准，再决定正式实验规模。默认是 **DME 4 人 + DRUSEN 4 人、1 个受影响 shadow、5 个 Stage 1 seed × 5 个共用 attack seed**。所有训练保持 50 epochs。这里的 5×5 是每位病人的配对设计，不是 25 位独立病人。

已安装 v4 的用户先看 [UPDATE_DELTA_GPU_V4_1_CN.md](UPDATE_DELTA_GPU_V4_1_CN.md)。本文件的旧 v4 zip/bundle 只安装基础版本，新安装也需要先应用 v4.1 更新再执行第 4 节。后续启动步骤适用于更新后的代码；默认结果目录已改为 `results/cross_stage_calibration_v4_1`，旧结果保留。

## 1. 这个包是什么

- `cross-stage-calibration-v4.bundle`：完整 Git 历史与新分支，可以直接 clone，不要求先安装 v1/v2/v3 patch。
- `code/`：v4 的相关源文件和说明，便于直接查看。
- `validation/`：本地数值测试与合成链结果。只证明实现机制和簿记通过，不能证明真实 OCT score 有预测力。
- `SHA256SUMS`：文件校验。

新代码用 `src/` 目录中的模块。推荐在 Delta 建立 `~/oct2-calibration-v4`，连接原来的 OCT 数据目录。原先的 `~/oct2` 继续保留。

v4 还统一了 CE 定义：直接计算 logits 的稳定 log-softmax CE，与公式的 J_Q 一致。旧版 sklearn 对 float32 概率的裁剪可能压低极端 CE（回归例子：30 变为约 15.94）。因此 05/07 会拒绝混用旧 CE 目录，需在新的 v4 目录生成 truth；NPZ 额外保存各 seed/query 的 `*_log_probs`。完整合成链使用测试专用的 16×16 输入，生产 OCT loader 仍为 128×128。

## 2. 在你的 Mac 上传

下载 `cross_stage_calibration_v4.zip` 到 Downloads，然后打开 Mac 的 Terminal：

```bash
cd ~/Downloads
unzip cross_stage_calibration_v4.zip
cd cross_stage_calibration_v4
shasum -a 256 -c SHA256SUMS
```

下面把 `YOUR_NCSA_USERNAME` 改成你平时登录 Delta 的用户名。它不一定与 UIUC NetID 相同：

```bash
scp cross-stage-calibration-v4.bundle YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu:~/
ssh YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu
```

按提示输入 NCSA 密码并完成 Duo。Delta 官方推荐这个登录地址和 SSH 登录流程：[Delta Login Methods](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)。

## 3. 在 Delta 准备新目录

登录 Delta 后执行：

```bash
git clone --branch cross-stage-calibration-v4 \
  ~/cross-stage-calibration-v4.bundle ~/oct2-calibration-v4
cd ~/oct2-calibration-v4
git log -1 --oneline
git status --short
ln -s ~/oct2/data data
mkdir -p logs
ls data/OCT2017
accounts
```

如果原来的数据在 `~/oct2/data/kermany2018/OCT2017`，用 `ls data/kermany2018/OCT2017` 检查即可；loader 支持这两种布局。如果数据目录在别处，把 `ln -s` 的第一个路径改成实际的 data 目录。不要重复创建已存在的符号链接。

`accounts` 输出中确认可用的 GPU Project。你旧脚本用的是 `bgjy-delta-gpu`；下面命令保留这个名字，如果实际分配不同，只替换 `--account` 后的值。

脚本申请 1 张 A40、8 CPU、64 GB 内存、最长 12 小时，并加载你之前使用的 `pytorch-conda/2.8`。通过 `sbatch` 提交计算任务；可以用命令行覆盖脚本的资源设置。NCSA 官方文档也列出了 `gpuA40x4` 队列和 `accounts` 命令：[Running Jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)。

## 4. 先验证环境，再开始真实 OCT 校准

每次提交后会显示 `Submitted batch job 数字`。后续用这个数字查看日志和状态。

### 第一步：数值与簿记测试

```bash
cd ~/oct2-calibration-v4
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm tests
squeue -u "$USER"
```

用实际 job ID 替换 `JOB_ID`：

```bash
tail -n 60 logs/oct_cal_v4-JOB_ID.out
tail -n 60 logs/oct_cal_v4-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

应看到数学测试 `all checks passed`、复现测试和分析测试均为 `OK`，复现测试日志包含 `training_device=cuda` 与 `128x128 exact training replay passed on cuda`，以及 Slurm `COMPLETED` / `0:0`。Python unittest 的正常结果也会写到 `.err`，因此不能仅凭 `.err` 非空判断失败。

随后做一次短合成链检查（不读取 OCT）：

```bash
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm smoke
```

确认 `SYNTHETIC CHAIN PASSED` 后继续。

### 第二步：08 eligibility preflight，冻结病人面板

```bash
sbatch --account=bgjy-delta-gpu --time=00:30:00 \
  scripts/cross_stage/10_calibration.slurm preflight
```

完成后查看：

```bash
cat results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json
```

确认 `panel_complete: true`、`shortfall: {}`，并记下 `proposed_affected_shadows`。脚本根据合格人数选择 shadow，后续自动消费同一个面板，不用手动猜 shadow 编号。面板只根据 split、病人图像数和固定 selection seed 选择，不读取 score 或删除效果。

如果面板不足，先看各类人数和图像数分层。选择新规则时建立新的 `--root` 并记录原因；不要在旧冻结面板上静默补人。

### 第三步：baseline 与 no-op 检查

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm baseline
```

默认 Stage 1 seeds 是 `42 43 44 45 46`，attack seeds 是 `5101 5102 5103 5104 5105`；每个 Stage 1 seed 使用完全相同的 attack seed 列表。

目录名由实际受影响 shadow 决定，例如 `shadow1_full`。下面用通配符查看检查结果：

```bash
cat results/cross_stage_calibration_v4_1/shadow*_full/attack_gate_summary.json
cat results/cross_stage_calibration_v4_1/shadow*_full/no_op_replays.json
```

默认要求两类在每个 Stage 1 seed 上都通过：每个 membership label 至少 50 个 target queries，基于 K 次 attack 指标均值的 AUC ≥ 0.55、balanced accuracy ≥ 0.53；no-op 的 Stage 1 向量和全部 K 个 attack 的概率必须复现。失败会停下，不能通过删除失败类别来缩小原先指定的校准面板。

**baseline_only 只能说明 attack 可用、复现一致，不能估计病人删除的 CE/AUC 方差。**

### 第四步：真实病人删除，并计算 CE/AUC 方差

上一步成功后提交：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm truth
```

这一步复用完成的 baseline checkpoint，真实重训每个病人的删除模型，并评估 J00/J10/J01/J11。完成后自动运行 09。

主要输出：

| 文件 | 看什么 |
|---|---|
| `shadow*_full/experiment_summary.json` | `status: complete`、40 个 patient × Stage 1 seed runs、主要 estimand |
| `shadow*_full/runs/seed*_patient*.json` | 每个条件下每个 attack seed 的 CE、AUC、Brier |
| `shadow*_full/runs/*_attack_probs.npz` | 全部 attack seeds 的概率与 log-probability、query raw index 和 patient ID，供后续 query bootstrap |
| `shadow*_full/seed_stats/paired_seed_effects.csv` | 先在相同 seed 下做差的原始结果 |
| `shadow*_full/seed_stats/seed_variance_components.json` | 每位病人的 Stage 1、attack、交互方差，以及 grand mean 的 SE |

### 第五步：实现的 score 与 oracle ladder

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm score
```

默认 γ_A=0.001、γ_S=0.01 是**校准的起始数值设置**，不能预先当作已经通过的设置。07 保存谱、CG 真残差、attack 解的条件数和每层 coverage。

结果位于 `shadow*_full/score_ladder_A0.001_S0.01/`：

- `ladder_rows.csv`：每位病人 × Stage 1 seed，已对 attack seeds 平均。
- `ladder_attack_seed_rows.csv`：每位病人 × Stage 1 seed × attack seed，score 与 truth 逐 seed 对齐。
- `ladder_summary.json`：每类的相关、MAE、符号一致率、可靠行数；病人均值只使用完整 Stage 1 seed 面板。

先看哪些层通过数值检查，再看效应量和预测误差。如果通过率很低，应依据谱和真残差调整 damping 或 CG 迭代次数，并保留原始结果。不要依据相关性选择 γ。示例：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm score \
  --damping_shadow 0.1 --damping_attack 0.01 --cg_iters 150
```

这个示例会写入另一个按 damping 命名的目录；它不是对真实 OCT 合适 damping 的保证。相同 damping 下如果改变迭代次数，原来的 score 目录参数不兼容会报错；用 07 的 `--out_dir` 指定新的诊断目录。通过数值检查仍不等于有限步 Adam 训练满足隐式公式的驻点近似。

## 5. 当前实验究竟测什么

令 (P_0,P_1) 为删除前后受影响 shadow 的预测向量，(M_0,M_1) 为原始及重标注 membership。target queries、其他 shadow 和 target model 固定。

| 条件 | 向量 | membership |
|---|---|---|
| J00 | P0 | M0 |
| J10 | P1 | M0 |
| J01 | P0 | M1 |
| J11 | P1 | M1 |

**H1（主要问题）**：continue_pathway2 Eq.75–79 的 patient-level 连续 score 能否预测 matched-class attack CE 的 (J_{10}-J_{00})？

\[
\widehat{\Delta J}_{\mathrm{value},p}
=\frac{\alpha}{n_s}\sum_{i\in p}
h_s^\top(H_s+\gamma_s I)^{-1}g_{si},\qquad
h_s=\sum_j J_{p_{sj}}^\top[-B_{sj}^\top(H_A+\gamma_A I)^{-1}q].
\]

代码中的 (H_s) 包括 Stage 1 的 L2 项，α 是**移除比例**，该病人的 loss 权重为 (1-\alpha)。CE Δ>0 表示删除后 attack 的 CE 增加。AUC Δ<0 表示删除后排序能力降低；两者不是同一个量。不能要求 CE 的可微 score 在 AUC 上也数值校准。

**H2（次要问题）**：将预测向量变化与真实 membership 重标注结合的 hybrid 能否预测 (J_{11}-J_{00})，并优于只做 (J_{01}-J_{00}) 的 relabel-only baseline？



| score / 层 | 预测对象 | 所需数值 gate |
|---|---|---|
| L1 lin：真实 ΔP | value | attack solve |
| L2 lin：真实 Δθ | value | attack solve |
| L2 retrain：真实 Δθ 的 JVP，再重训 attack | value | 不依赖 Hessian 逆 |
| L3 lin：Eq.79 | value | attack solve + score CG |
| L3 retrain / hybrid：估计 Δθ 的 JVP，再重训 attack | value / full | Δθ CG |
| frozen_h | value 的对照 | attack solve |
| frozen_self | value 的对照 | 不依赖 Hessian 逆 |

这些都是数值 gate；baseline 的 attack 能力检查是另一个层面的前提。正的 Lanczos 最小 Ritz 值只能表示当前探测未发现负曲率，不构成全空间 SPD 的证明。

**为什么增加 seed？** 对固定病人和 shadow，crossed 设计使用

\[
\Delta_{rk}=\mu+a_r+b_k+e_{rk},\quad
\operatorname{Var}(\bar\Delta)=\sigma_S^2/R+\sigma_A^2/K+\sigma_{SA}^2/(RK).
\]

09 用 balanced two-way ANOVA 的矩估计分解；一个 seed cell 只有一次训练，因此交互项包含 cell residual，不能再拆开。小样本可能出现负方差分量估计：保留原始值，并另外输出截为 0 的描述性分量和 SE。这里不是精确置信区间，也不是病人总体或 target-query 总体的不确定性。缺 seed 时标记 `incomplete_grid`，K=1 只标 `combined_only_K1`。

如果 Stage 1 项主导，优先增加 R；attack 项主导，优先增加 K；病人之间差异和排名仍不稳定，则增加病人面板。正式分析需要在固定类别内按病人聚类，并保持 crossed seeds 在重采样时共用，不能把 8×5×5 行当成 200 个独立样本。

## 6. Dose 与前后半程实验

先完成 full 校准。之后用同一个冻结面板运行，例如：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm truth --condition dose025
```

该任务完成后：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm score --condition dose025
```

同样支持 `dose01`、`dose05`，α=0 直接用 baseline，α=1 是 full。检查小 α 时 score/真实 Δ 的尺度和方向是否一致，而不仅检查代码的 ×α 恒等式。

时间窗口：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm truth --condition early
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm truth --condition late
```

默认 early 是 epochs `[0,25)` 删除，late 是 `[25,50)` 删除。patient 仍在另半程被训练，因此只评价 J00/J10。full 删除也先取 **value** 效应，定义时间非加性残差：

\[
R_{\rm time}=\Delta_{\rm full\ horizon,value}-\Delta_{\rm early,value}-\Delta_{\rm late,value}.
\]

不要把这里的 full-horizon value 与四格设计的 full effect（J11−J00）混同。静态 Eq.79 没有窗口因子，window 的 L3 与 frozen score 保持 N/A。当前仍未实现 Adam trajectory score（Eq.42+74）；保存 midpoint/final Adam 状态也不等于完整 trajectory 导数已经实现。

## 7. 什么时候扩大到正式实验

完成校准后，先根据 CE/AUC 方差分量决定 R/K，检查每层 coverage、原始/投影 JVP 的差异和 damping 稳定性，再冻结正式参数。可考虑每类约 30 位病人、至少 3 个受影响 shadow，最终人数要服从 08 的真实 eligibility。

新建正式实验 root，例如：

```bash
sbatch --account=bgjy-delta-gpu \
  scripts/cross_stage/10_calibration.slurm preflight \
  --root results/cross_stage_formal_v4 --n_affected_shadows 3 --patients_per_class 10
```

后续 baseline/truth/score 都传同一 `--root`、`--n_affected_shadows 3 --patients_per_class 10`，分别以 `--panel_index 0`、`1`、`2` 运行。每个 shadow 有不同病人，不能把 3 个 shadow 当成很多独立模型。若用于 confirmatory 验证，应额外冻结新的 Stage 1/attack seed 列表；建议预留与校准不重叠的病人面板，或在正式统计中排除已用于调参的校准病人。runner 支持 `--split_seed` 和 `--selection_seed`；默认仍为 42005/42006。换 root 本身不会得到新 split，换 split 也不自动保证与校准病人不重叠。

## 8. 上传 GitHub

训练可以直接通过 bundle 开始。GitHub 用于保存代码版本。你可以在 Mac 上操作，避免依赖 Delta 上是否已配置 GitHub 登录。

在 Mac 新 Terminal：

```bash
git clone --branch cross-stage-calibration-v4 \
  ~/Downloads/cross_stage_calibration_v4/cross-stage-calibration-v4.bundle \
  ~/oct2-calibration-v4-code
cd ~/oct2-calibration-v4-code
git remote set-url origin https://github.com/shan927150/oct2.git
git log -1 --oneline
git push -u origin cross-stage-calibration-v4
```

如果没有 GitHub 登录，已安装 `gh` 时先执行 `gh auth login`，选择 GitHub.com、HTTPS、浏览器登录，并允许配置 Git 凭证；然后重新 push。GitHub 官方说明：[缓存 Git 凭证](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)。无需把 token 填进仓库地址或聊天。

push 后打开仓库，会看到新分支 `cross-stage-calibration-v4`，可以创建 pull request。这里不需要 force push。[GitHub push 文档](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository)。

Delta 上如需以后从 GitHub 更新：

```bash
cd ~/oct2-calibration-v4
git remote set-url origin https://github.com/shan927150/oct2.git
git fetch origin
git branch --set-upstream-to=origin/cross-stage-calibration-v4 cross-stage-calibration-v4
git pull --ff-only
```

本次交付没有替你登录 Delta、提交 GPU 作业或确认远端 push 成功。代码已经在 bundle 内。

## 9. 跑完以后发哪些结果来继续判断

在 Delta 打包结果（排除模型 checkpoint 和图片）：

```bash
cd ~/oct2-calibration-v4
tar --exclude='*/checkpoints' --exclude='*/data' \
  -czf ~/oct_calibration_v4_results.tar.gz results/cross_stage_calibration_v4 logs
```

回到 Mac Terminal 下载：

```bash
scp YOUR_NCSA_USERNAME@login.delta.ncsa.illinois.edu:~/oct_calibration_v4_results.tar.gz ~/Downloads/
```

把这个结果包发来，就可以用真实数值判断：score 在哪一层失真、数值 gate 的覆盖率、CE/AUC 方差有多大、该增加哪种 seed，以及是否需要转向 Adam trajectory score。

作业失败时保留对应 `.out/.err` 和 `experiment_summary.json`；同参数重复提交 truth 会利用已完成的 checkpoint/病人结果。不要用 `--overwrite` 处理不明原因的失败，也不要在同一输出目录同时启动两份相同作业。
