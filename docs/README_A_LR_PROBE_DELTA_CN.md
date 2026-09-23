# Route A LR probe 修订版提交说明

范围：五个 affected shadow3 baseline，各从 E100 的模型、Adam、RNG 状态开始。
control 再训 40 epochs，lr=1e-3；decay 前 20 epochs lr=1e-4，后 20 epochs lr=1e-5。
只做收敛诊断，不训练 LOO、attack，不扫描 damping，不选择 E*。
新的衰减配方按项目约定确认后再提交。准备时填写真实确认记录，不要虚构学长批准。

修订内容：完整的十任务与连续 epoch 检查、输入/数据/随机数配对检查、修正 dropout 解释、独立 clone 安装、手动测试门及重复提交保护。
原 `~/oct2-exp-a` 保持 `3231a195`，原 `~/oct2-calibration-v4` 与 E95/E100 输出只读。

## 1. Mac 本地终端上传

下载本次修订的 `oct2_A_lr_probe.zip` 到 Downloads。不要误用之前下载的旧 ZIP。

```bash
scp ~/Downloads/oct2_A_lr_probe.zip \
  yli103@login.delta.ncsa.illinois.edu:/u/yli103/oct2_A_lr_probe_v2.zip
```

若你平时使用不同的 Delta SSH alias，用该 alias 替换主机。

## 2. Delta 上安装到独立 clone

```bash
unzip /u/yli103/oct2_A_lr_probe_v2.zip -d /u/yli103

bash /u/yli103/oct2_A_lr_probe_v2/install_A_lr_probe_isolated.sh \
  /u/yli103/oct2-exp-a \
  /u/yli103/oct2-exp-a-lrprobe
```

安装器校验 SHA256，拒绝覆盖已有目标目录，且不会更改源仓库的 HEAD 或分支。
目标是独立 clone，保留原分支名以满足 preflight，但不共享 Git refs 或对象硬链接。
没有 GitHub push，没有 GPU 提交。若源目录已经被升级，请停止并先核对现状，不要 reset。

## 3. 准备新 attempt，只提交 GPU 测试

```bash
cd /u/yli103/oct2-exp-a-lrprobe
module load pytorch-conda/2.8
export OCT_BASELINE_ROOT=/u/yli103/oct2-calibration-v4
export OCT_RUNS_ROOT=/u/yli103/oct2-pathway2-runs/A
export OCT_DATA_DIR=/u/yli103/oct2/data
export OCT_A_STEP1_ROOT=/u/yli103/oct2-pathway2-runs/A/convergence_E100_20260918T111502

read -r -p '填写真实确认记录（确认人、日期、baseline-only decay probe 范围）: ' OCT_A_LR_NOTE
export OCT_A_LR_OUT="$(bash scripts/cross_stage/submit_A_lr_probe.sh prepare "$OCT_A_LR_NOTE")"
test -f "$OCT_A_LR_OUT/probe_config.json"
printf "export OCT_A_LR_OUT='%s'\n" "$OCT_A_LR_OUT" > ~/routeA_lr_probe_root.sh

bash scripts/cross_stage/submit_A_lr_probe.sh test "$OCT_A_LR_OUT"
bash scripts/cross_stage/submit_A_lr_probe.sh status "$OCT_A_LR_OUT"
```

第一次 prepare 创建新结果目录；再次准备会创建另一个独立 attempt，因此不要为查询状态重复 prepare。
后续重新登录用 `source ~/routeA_lr_probe_root.sh` 恢复目录。
test 只提交一个 1-GPU、30 分钟上限的测试任务，不自动提交 array。
GPU 测试覆盖 CUDA 可用性、合成小数据上的模型/Adam/RNG 恢复配对，以及报告和提交保护。
合成测试不等于已经在真实 OCT 上重跑了 140 epochs。

## 4. 测试成功后，单独提交十任务 array

必须先看到测试 `COMPLETED 0:0`，并且 status 显示 `gpu_tests_passed: true`。
若 sacct 暂时尚未更新，等待记录更新即可，不要重新提交 test。

```bash
cd /u/yli103/oct2-exp-a-lrprobe
source ~/routeA_lr_probe_root.sh

bash scripts/cross_stage/submit_A_lr_probe.sh run "$OCT_A_LR_OUT" --max-parallel 2
bash scripts/cross_stage/submit_A_lr_probe.sh status "$OCT_A_LR_OUT"
```

array 是 0–9，共十个任务，一次最多两个。每任务 1 GPU、48 GB、2 小时上限。
0/1=seed42 control/decay，2/3=seed43，4/5=seed44，6/7=seed45，8/9=seed46。
两个 truth-dose 若仍在跑，加上这里最多两个任务，相关任务合计最多四个 GPU；其他作业仍需计入。
`--max-parallel` 支持 1..4。代码与输入锁定后不要改此 clone 或移动 attempt。
run/test 重复调用会拒绝，不会重提。若提交返回异常，先查已有作业与锁文件，不要直接删锁。

## 5. 全部十个任务成功后生成报告并下载

```bash
cd /u/yli103/oct2-exp-a-lrprobe
source ~/routeA_lr_probe_root.sh
module load pytorch-conda/2.8

bash scripts/cross_stage/submit_A_lr_probe.sh status "$OCT_A_LR_OUT"
python3 scripts/cross_stage/23_A_lr_report.py --root "$OCT_A_LR_OUT"
```

报告要求五个 seed 的 control/decay 全部 complete，连续 epoch 100–140 和配对证据完整。
输出在 `$OCT_A_LR_OUT/report`，包括 PNG、PDF、REPORT.md、probe_summary.json。
`--allow_incomplete` 仅用于诊断，会标记 incomplete，不能用它替代验收。

回到 Mac 本地终端，可下载轻量诊断文件，省略模型 checkpoint：

```bash
OCT_REMOTE_PROBE="$(ssh yli103@login.delta.ncsa.illinois.edu 'source ~/routeA_lr_probe_root.sh; printf "%s" "$OCT_A_LR_OUT"')"
test -n "$OCT_REMOTE_PROBE"
mkdir -p ~/Downloads/oct2_A_lr_probe_results
rsync -av --exclude='checkpoints/' --exclude='epoch_orders.npz' \
  "yli103@login.delta.ncsa.illinois.edu:${OCT_REMOTE_PROBE}/" \
  ~/Downloads/oct2_A_lr_probe_results/
```

## 判读边界

- 梯度目标 1e-3 是探索性目标，不是严格驻点证书。小步长导致参数移动变小也不构成证明。
- 末十个 epoch 的完整梯度窗口用于摘要；不会用更早的 epoch 补缺失值。
- eval 与 dropout-MC 梯度都大，不能推断目标不匹配。需结合 MC 均值误差；该误差不是范数置信区间。
- 本 probe 无法单独证明 L3 改善。后续 LOO 需让 baseline/full/dose01 各自从自己的 E100 状态使用相同配方，另行冻结实验。
- 当前交付环境无 PyTorch/GPU，实际训练配对测试必须由 Delta 的 GPU test 通过后再运行正式 array。
