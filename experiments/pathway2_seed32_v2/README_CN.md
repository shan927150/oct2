# Stage 1 前五项任务，seed32 v2 修订版

本版基于用户提供的 `oct2_seed32_lds_v2.bundle`，原提交 `5e145ef`，修订分支为 `exp/pathway2-seed32-lds-v2-reviewed`。沿用原目录和训练内核，修改仅在 `experiments/pathway2_seed32_v2/`。实验优先级以 2026-10-05 会议纪要及本次“做前五项”为准。

代码与 CPU 验证已经完成，尚未提交本轮 Delta GPU 作业，也没有新的 OCT 科学结果。不要把这里的合成测试数值当作实验结果。详细改动与验证见 `REVIEW_CN.md`。

## 前五项的实现

| 任务 | 默认设计与输出 |
|---|---|
| 1 扩展 seed | 原 8 masks × seeds42–73，加 32 个 full baseline。复用原 27 个单元。140 epochs，raw/EMA；SE、独立分组排序可靠性、LDS 随 k 的曲线。额外保留 k=3 便于核对旧 pilot。 |
| 2 删除者 LDS | `-(1-m) @ tau`，与 retained sum 只差常数。保留 matched 版本 `((N/Nj)*m-1) @ tau`。 |
| 3 大小控制 | 新 32 masks × 8 seeds。按类别固定保留病人数，总图像数容差 ±2%，各类图像数容差 ±10%，并固定每轮优化步数。报告 size、composition、控制大小后的相关，以及完全相同 seed 面板下的设计比较。 |
| 4 γ 扫描 | 正式 γ 固定为 1、0.3、0.1、0.03。另有先行 `score-pilot`，只算 1−p_y × seeds42–44 × {1,0.3,0.1}，γ=1 作对照。 |
| 5 逐查询 LDS | 163 名 query patient 的梯度内积预览。按子集联合 bootstrap 和 permutation，报告均值、CI、逐病人 LDS。 |

CE/raw 是原定主分析。Brier 和 1−p_y 是看到 CE 后加入的探索目标。EMA 与 raw 来自同一训练轨迹。逐查询实现是 gradient-dot proxy，不含每个 query 的逆 Hessian，也不是完整 TRAK 算法。

## Delta 安装

将交付 zip 解压，在 Mac 终端进入解压目录后上传 bundle：

```bash
scp oct2_seed32_lds_v2_reviewed.bundle yli103@login.delta.ncsa.illinois.edu:/u/yli103/
ssh yli103@login.delta.ncsa.illinois.edu
```

下面在 Delta 使用 `/u/yli103/oct2_seed32_lds_v2_reviewed.bundle`。

```bash
# 在 Delta 执行，不覆盖原 worktree
cd /u/yli103/oct2
git fetch origin exp/pathway2-lds-formula-v1
git bundle verify /u/yli103/oct2_seed32_lds_v2_reviewed.bundle
git fetch /u/yli103/oct2_seed32_lds_v2_reviewed.bundle refs/heads/exp/pathway2-seed32-lds-v2-reviewed:refs/heads/exp/pathway2-seed32-lds-v2-reviewed
git worktree add /u/yli103/oct2-exp-seed32-v2-reviewed exp/pathway2-seed32-lds-v2-reviewed
CODE=/u/yli103/oct2-exp-seed32-v2-reviewed/experiments/pathway2_seed32_v2
module load pytorch-conda/2.8

# CPU 检查，真实 CNN 集成检查使用小型合成输入
python3 "$CODE/test_experiment.py"
python3 "$CODE/test_review.py"
python3 "$CODE/test_gpu_path_cpu.py"
```

Bundle 的前置提交是 `c4880c9`。它包含用户原版与本次修订。没有依赖远程已发布的 reviewed 分支。若已安装过同名分支，请先核对现有提交，不使用强制覆盖命令。

## 运行顺序

```bash
# 1. 新目录冻结配置。准备不会提交 GPU 作业。
OUT=/u/yli103/oct2-pathway2-runs/A/seed32_lds_v2_reviewed_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --out "$OUT"

# 2. 提交 smoke，以及依赖 smoke 的训练包。先推迟正式 γ 扫描。
python3 "$CODE/run.py" submit --out "$OUT" --defer-score
python3 "$CODE/run.py" status --out "$OUT"

# 3. 等 smoke 成功后，看实际 HVP 耗时。
cat "$OUT/smoke/result.json"
# timing.score_task_operator_estimate_hours 只估计算子耗时，不是时间上界。
# 下列 walltime 按该估计留余量调整。
python3 "$CODE/run.py" resubmit --out "$OUT" --only score-pilot --score-time 04:00:00

# 4. 等三个 pilot 作业结束。status 会打印各 γ 的资格与真实残差。
python3 "$CODE/run.py" status --out "$OUT"
# pilot 用来核对求解行为和成本，不按 LDS 结果选择 γ。
# 正式扫描仍执行 prepare 冻结的完整 γ 面板。
python3 "$CODE/run.py" resubmit --out "$OUT" --only score --score-time 04:00:00

# 5. 查询进度与排队时间。
python3 "$CODE/run.py" status --out "$OUT"
python3 "$CODE/run.py" queue-report --out "$OUT"
python3 "$CODE/run.py" queue-report --since 2026-09-28

# 对应作业结束后，按需要补未完成单元。
python3 "$CODE/run.py" resubmit --out "$OUT"

# 6. 中途汇总与最终汇总。
python3 "$CODE/run.py" collect --out "$OUT" --allow-partial
python3 "$CODE/run.py" collect --out "$OUT"
```

`prepare --eq-seeds 4` 可将新设计的训练减半，须在第一次 prepare 时决定。默认训练单元 544 个，其中 27 个复用，517 个新训。按控制器的保守时长参数估计，新训练约 34.1 GPU 小时，另加 smoke、proxy 和 γ 求解。实际耗时以 Delta 日志为准。

子集 pack 通常包含 5 次新训练，最多 8 个单元；full baseline 较慢，通常每包 2 次。训练包 walltime 为 30 分钟。`--concurrency 8` 是每个数组的并发上限，同时存在多个数组时，总并发可能超过 8。

## 必需的验证与恢复

- GPU smoke 先重训旧 subset000/seed42，再重训 full/seed42。前者 raw/EMA、后者 raw/EMA/Adam 都要逐位复现；同时实测算子耗时。CPU 测试不能替代这一步。
- Harvest 比较旧概率，旧 fp64 子集还比较 CE。新结果保存 fp64 概率与 log-softmax，CE 梯度直接从 logits 求导，避免极小正确类概率导致溢出。
- γ 门槛为真实相对残差 ≤1e-3、求解空间内无非正曲率，并通过随机 Lanczos 探针。探针不是整个 Hessian 正定的证明。
- 一个 γ 失败会保留失败原因，其他 γ 可以完成。未通过门槛的 γ 不保存 τ，也不进入 LDS。`complete.json` 表示这次评估已完成，`qualified` 才表示求解通过。CE γ=1 若与旧 CG 分数不一致，整个 score task 不写完成标记。
- 补跑先核对该类旧作业的 squeue/sacct 状态，只补尚未完成的单元。配置、结果身份和哈希用于防止混用旧数据。提交锁避免重复 sbatch，单元锁避免两个 worker 同时计算同一单元。
- `submission_pending.json` 表示 sbatch 是否成功尚不确定。先用 squeue/sacct 和其中的命令核对；确认已有作业后补录 receipt，或确认没有作业后清除该 pending 记录。不要直接重新提交。
- smoke 失败后，确认旧 smoke 结束，再用 `resubmit --only smoke`。原依赖数组可能已经被取消；新的 smoke 通过后，再用普通 resubmit 补跑。

## 如何读报告

输出包括 `REPORT_CN.md`、`summary.json`、CSV 表和 PNG 图；压缩包还保留 interface/proxy/score 的 NPZ，checkpoint 留在 Delta 原目录。

`table_same_seed_designs.csv` 专门比较相同 seed 身份下的 pilot8 与 eq32。不同设计仍有不同的抽样分布，图像数容差也不能证明所有混杂已消失。

固定全 seed 面板的标准差后，`sd/sqrt(k)` 按构造下降；有限池中抽全部 seeds 时变异按构造为零。独立 k 对 k 排序复现只在 `2k <= seeds` 时可估计。Spearman–Brown 和 ICC 外推是启发式模型，不能当成测得的 32 对 32 复现。

LDS 不必随 k 单调增加。旧记录中的 CE LDS≈0.286、size≈0.667、控制 size 后≈0.119 是历史结果。本版没有新增实际 OCT 数值。“约 0.33”只是特定衰减假设下的旧估计，不是硬上界。

Composition ridge 是交叉拟合诊断，表中不附固定预测器的 bootstrap CI 或 permutation p，因为有效推断需要在每次重采样中重新拟合整个流程。逐查询区间则共同重采样子集，条件于当前 seed 面板与查询集，不能把 163 名查询病人当成独立训练重复。

本轮只实现会议前五项。原 S1、late deletion、polish 和 Stage 2 attack J 不属于本轮新实验。
