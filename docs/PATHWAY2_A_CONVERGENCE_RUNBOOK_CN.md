# Route A：先看收敛，再分阶段重跑 Level 3（2026-09-18 修订版）

分支：`exp/pathway2-a-stationarity`。原 50-epoch baseline、truth、split、patient panel、初始化、epoch orders、dropout/RNG、Stage 2 配置全部只读。新结果只写入 `oct2-pathway2-runs/A/`，不能替代原正式 truth。

本修订版解决两个执行问题：

1. Step 1 只跑会上要求的 affected shadow 五个 seeds（42–46），不再默认跑 target、四个 fixed shadows 或 Hessian spectrum。
2. Step 2 不再一次提交一条很长的依赖链。每条命令最多提交一个 Slurm job；每一步结束后先看 `status` 和结果文件，再手动决定是否继续。A2 damping 必须在 A1 结果经过人工审核并留下 approval record 后才会解锁。

## 1. Step 1 做什么

五条曲线把 affected shadow 3 按原常数学习率 Adam 训练从 50 epoch 延长到 `E_max=100`。每个 seed 是一个 array task，array 同时最多运行四个任务（`%4`）。

默认记录：

| 指标 | 频率 | 含义 |
|---|---:|---|
| `online_train_ce`、参数更新 | 每个 epoch | dropout 开启时的真实训练轨迹 |
| eval CE/objective、held-out CE/accuracy、interface drift | 每个 epoch | dropout 关闭的描述性曲线 |
| `eval_grad_norm` | 每 5 epoch，并强制记录 0、50、100 | 检查式 56 的驻点前提 |
| dropout-MC 梯度诊断（16 reps） | 50、75、100 | 区分 eval-mode 梯度与 dropout 期望梯度 |
| checkpoint（模型、Adam、RNG） | 50、55、…、100 | 供人工选择一个真实保存点 E* |

默认不计算 Hessian spectrum。Step 1 的目标是先回答“曲线是否趋平”和“梯度/更新是否同时变小”，不是先做新的 damping 搜索。

### 1.1 精确续训检查

- epoch order 的前 50 行必须与原运行逐位相同。
- 原来保存的 epoch 25/50 checkpoint 会核对模型参数、Adam state、CPU/CUDA RNG。
- 原 epoch-50 final model 与 post-train RNG 也必须精确相同。
- 观察器在保存/恢复 RNG 的上下文内运行，不改变 dropout sequence、`.grad` 或 Adam state。
- 每个任务保存代码、原输入和数据数组指纹，并在结束前再次核对。任一检查失败，该任务直接标记 `failed`。

### 1.2 提交 Step 1

```bash
cd ~/oct2-exp-a
export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4
export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A
export OCT_DATA_DIR=/u/yli103/oct2/data

bash scripts/cross_stage/submit_stage1_extended_convergence.sh \
  | tee ~/submit_pathway2_A_step1.txt
```

这个长文件名是上一版 handoff 中已经发出的兼容入口；它只转到正式实现
`submit_A_step1.sh`，两者行为完全相同。

它只会提交：

1. 一个 GPU 回归测试 job；
2. 测试成功后，一个 `0-4%4` convergence array。

查看 job ID 和状态：

```bash
cat ~/submit_pathway2_A_step1.txt
sacct -X -j <GPU_TEST_ID>,<ARRAY_ID> \
  --format=JobID%18,JobName%16,State,ExitCode,Elapsed,NodeList
```

当 array 的五个 task 全部是 `COMPLETED 0:0` 后，使用 submission 文件中显示的 `OUT=...`：

```bash
module load pytorch-conda/2.8
export OCT_A_OUT=<复制 submission.txt 中的 OUT 路径>
python3 scripts/cross_stage/20_A_convergence_report.py --root "$OCT_A_OUT"
```

主要输出：

- `report/curves_affected_shadow.png`
- `report/window_table.csv`
- `report/plateau_report.json`
- `report/REPORT.md`

报告只给出 10-epoch window mean、变化、slope 和 range，并明确写 `selected_E_star=null`。代码不会自动宣布 plateau，也不会自动提交 Step 2。

### 1.3 人工冻结 E*

看完五种颜色的 loss、gradient、relative update、held-out CE 和 interface drift 后，选择一个已经保存的 checkpoint（默认只能选 55、60、…、100）。例如团队决定使用 70：

```bash
python3 scripts/cross_stage/20_A_freeze_epoch.py \
  --root "$OCT_A_OUT" \
  --epoch 70 \
  --confirmed_by "advisor meeting 2026-09-18" \
  --note "reviewed all five curves; use epoch 70 for one sensitivity rerun"
```

这条命令不会帮你选择 epoch。它只验证五个 seed 完整、checkpoint 存在、来源一致，然后生成不可覆盖的：

```text
$OCT_A_OUT/frozen_E_star.json
```

## 2. Step 2 为什么拆开

正式 `full truth` 内部仍保持一个完整运行单元，因为随意拆分并事后拼接 seeds/patients 会改变现有配置和完整性检查。但它会持续写出 `runs/seed*_patient*.json`，所以 `status` 能显示已经完成多少个 patient-seed cells。

外层拆成以下人工关卡：

| 顺序 | 命令 | 内容 | 自动提交下一步？ |
|---:|---|---|---|
| 0 | `prepare` | 锁定 E*、commit、panel、baseline hashes | 否 |
| 1 | `test` | 小型 GPU replay/continuation smoke test | 否 |
| 2 | `truth-full` | E* 的 full truth；每条 Stage-1 路径做 T0 replay check | 否 |
| 3 | `score` | 只算原阻尼 `γ_A=0.2, γ_S=1` | 否 |
| 4 | `compare-primary` | A1：50 vs E* 的 matched-cell 主比较 | 否 |
| 5 | `truth-dose` | E* 的 dose01 truth | 否 |
| 6 | `preflight` | full/dose 参数尺度、方向和剂量线性检查 | 否 |
| 7 | `approve-a2` | 人工记录已经看过 A1；不提交 GPU | 否 |
| 8 | `damping` | A2：`0.01,0.03,0.1,0.3,1,2` | 否 |
| 9 | `compare-full` | matched-cell 完整报告 | 否 |

任何步骤失败都不会触发下游。一个 attempt 的输出不可覆盖；若正式运行失败并需要重试，再执行一次 `prepare`，它会创建带新 UTC 时间戳的新 attempt。

## 3. Step 2 命令（一次只执行一段）

先保持三个环境变量：

```bash
cd ~/oct2-exp-a
export OCT_BASELINE_ROOT=$HOME/oct2-calibration-v4
export OCT_RUNS_ROOT=$HOME/oct2-pathway2-runs/A
export OCT_DATA_DIR=/u/yli103/oct2/data
```

### 3.1 Prepare

```bash
bash scripts/cross_stage/submit_A_step2.sh prepare \
  "$OCT_A_OUT/frozen_E_star.json"
```

命令会打印一条精确的：

```bash
export OCT_A_STEP2_ROOT='...'
```

把这一行复制执行。之后所有命令都可省略路径参数。

### 3.2 先跑小型 GPU replay test

```bash
bash scripts/cross_stage/submit_A_step2.sh test
bash scripts/cross_stage/submit_A_step2.sh status
```

只有 `A0_step2_gpu_smoke` 显示 `complete`，`truth-full` 才会解锁。该测试用生成数据验证 target、fixed shadow、baseline、no-op 和每个 LOO endpoint 的 replay 与继续训练逻辑，不代表 OCT 科学结果。

### 3.3 Full truth

```bash
bash scripts/cross_stage/submit_A_step2.sh truth-full
bash scripts/cross_stage/submit_A_step2.sh status
```

运行中可以反复执行 `status`。它同时显示 Slurm 状态和：

- `experiment_summary.json` 状态；
- replay certificate 状态；
- 已写出的 `patient_seed_results_written` 数量（正式 full 完整值应为 40）。

正式 replay certificate 对所有路径执行原 epoch 50 检查：

- target：1 条；
- fixed shadows：4 条；
- affected baseline + no-op：10 条；
- LOO：5 seeds × 8 patients = 40 条。

每条路径必须重现原参数和 post-train RNG。baseline/no-op 还直接核对原 checkpoint 内的 Adam state 和 RNG；LOO 原 checkpoint 没保存 optimizer state，因此通过同一初始化、order、dropout sequence 做 deterministic replay，记录重新生成的 Adam-state fingerprint，然后用同一个内存中的 optimizer 继续到 E*。在 E*，五个 baseline 和五个 no-op 还必须与 Step 1 冻结 checkpoint 的模型、Adam、RNG 一致。

### 3.4 原阻尼打分和 A1 比较

确认 full truth 为 `complete` 后：

```bash
bash scripts/cross_stage/submit_A_step2.sh score
bash scripts/cross_stage/submit_A_step2.sh status
bash scripts/cross_stage/submit_A_step2.sh compare-primary
```

先阅读：

```text
$OCT_A_STEP2_ROOT/compare_primary_vs_E50/COMPARE.md
$OCT_A_STEP2_ROOT/compare_primary_vs_E50/comparison.json
$OCT_A_STEP2_ROOT/compare_primary_vs_E50/compare_50_vs_Estar.png
```

比较程序要求 50 与 E* 的 `(seed, patient_id, oct_class)` keys 完全一致，并在两个 endpoint 都通过相应数值 gate 的同一交集上计算 Spearman、MAE、sign agreement 和 Δθ cosine。它不再把两个 independently-qualified cohorts 放在一起比较。

### 3.5 Dose truth 和 truth preflight

```bash
bash scripts/cross_stage/submit_A_step2.sh truth-dose
bash scripts/cross_stage/submit_A_step2.sh status

# dose 完成后
bash scripts/cross_stage/submit_A_step2.sh preflight
bash scripts/cross_stage/submit_A_step2.sh status
```

preflight 完成后，先看：

```text
$OCT_A_STEP2_ROOT/diag_preflight/checkpoint_ratios.csv
$OCT_A_STEP2_ROOT/diag_preflight/manifest.json
```

### 3.6 人工批准后才跑 A2 damping

只有你已经看过 A1 `COMPARE.md` 和 truth preflight，才执行：

```bash
bash scripts/cross_stage/submit_A_step2.sh approve-a2 \
  "reviewed A1 matched-cell comparison and truth preflight; proceed with damping sensitivity"
```

它生成 `A2_DAMPING_APPROVAL.json`，记录被审核文件的 SHA256；仍然不会提交 job。随后：

```bash
bash scripts/cross_stage/submit_A_step2.sh damping
bash scripts/cross_stage/submit_A_step2.sh status

# damping 完成后
bash scripts/cross_stage/submit_A_step2.sh compare-full
```

完整输出位于：

```text
$OCT_A_STEP2_ROOT/compare_full_vs_E50/
```

共同 damping 值在 50/E* 的同一 qualified-cell intersection 上比较。新加入的 `γ=0.01` 在原 v1.1 E50 sweep 中没有对应点，因此报告会明确标记为 **E*-only sensitivity**，不会伪造 50-vs-E* 对比。

## 4. 状态与安全规则

任何时候都可以运行：

```bash
bash scripts/cross_stage/submit_A_step2.sh status
```

安全规则：

- 每个 submit 子命令只提交一个 job，没有 `afterok` 长链。
- 同一个 immutable attempt 中，同一阶段不能重复提交。
- tracked source、commit、selection、panel 或已审核 A1 文件发生变化，后续阶段立即拒绝运行。
- 原 baseline/truth 从不写入；输出目录不能与代码或 baseline 重叠。
- A2 没有 approval record 时无法提交。
- Route A 的 E* 结果只是敏感性分析，不替代原 50-epoch正式 truth。

## 5. 结果判读

重点同时看三组量：

1. Level 3 预测与真实 `J10−J00` 的 matched-cell Spearman、MAE、sign agreement；
2. predicted Δθ 与 true Δθ 的 cosine、truth 尺度和 dose 线性；
3. `‖b_p‖`、`‖h‖`、eval-gradient 和 qualified coverage。

若 E* 的公式误差变小，但真实 `J10−J00` 或 `‖b_p‖` 同时接近零，只能写“信号变弱或未解析”，不能直接写“Level 3 得到验证”。若 loss 变平但 gradient/update 仍大，也不能宣称已经达到驻点。
