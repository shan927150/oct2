# 设计：从 attack 训练表删除样本并重训 attack（替代旧 E2 的 Drop）

日期：2026-10-30。来源：10-30 meeting："把怀疑有问题的图从 attack 训练表里拿掉，让 attack 在剩下的数据上重新训练，再看预测和真值的相关是否上升。"

旧 E1、E2 没有做这件事：E1 是把另一类的 attack loss 从目标里去掉（每类 attack 是独立的 MLP，这样做本来就不需要重训，结果有效）；E2 的 Drop 只是把这些图的行从预测分数里去掉，真值没变。旧 E2 的 Fix、Leave 是把被删行换成重训值，不是删除。这个实验替代旧 E2 的 Drop。

## 1. 问题

把一批 shadow-3 的 member 图从 attack 训练表里移除，attack 在剩余数据上重训后，IF 对"Stage-1 子集变化引起的 attack query loss 变化"的预测与真实变化之间的 LDS，相比不删时（0.262）如何变化？

## 2. 做法

- **固定的部分**：同样的 32 个子集、Stage-1 seed 42–44、attack seed 面板、固定的 target queries、γ = 0.03、attack 阻尼和类别权重，全部沿用 v6 与接口 run。Stage-1 模型不重训，直接用已保存的重训概率与 IF 的概率变化。
- **删除**：只从 shadow 3 的 attack 训练表里去掉选中的 member 行。其他 shadow 的行、shadow 3 的 nonmember 行、target queries 都不变。Stage-1 的子集干预（删哪些图）不变，只是被去掉的行不再参与 attack 训练。
- **每个 (Stage-1 seed, attack seed, 类别)**：
  1. 过滤后的表加上完整模型的概率，重训 attack 得到 J00；同一个模型上重算 attack Hessian 和敏感度 v（被去掉的行没有 v）。
  2. 对每个子集，用该子集的重训概率替换 shadow-3 块，同样去掉选中的行，重训 attack 得到 J10。
  3. 真值 = Σ_c ω_c (J10 − J00)；预测 = Σ_c ω_c v·Δp_IF，其中 v 是第 1 步的新 v，Δp_IF 是原来的 IF 概率变化。预测和真值用同样的行、同样的 queries。
  4. 对 attack seed 和 Stage-1 seed 平均，再在 32 个子集上算 Spearman。
- **选图**（依据原实验的 `images.csv`，不使用重训值）：attack 活跃的 shadow-3 member 图中，
  - `low_margin`：margin 最低的 n 张；
  - `high_if`：|a_IF| 最大的 n 张；
  - 每组各配一组 DME/DRUSEN 数量相同的随机图（`low_margin_random`、`high_if_random`，种子 20261030）。
- **对照组 `none`**：什么都不删，用同一个流程重跑 attack。它和已存的 v6 真值比较，得到"同一个流程在这台机器上重放"的偏差；和删除组并排，才能分清 LDS 的变化来自删除还是来自重训的漂移。
- n 分别使用 50 和 200，二者是不同规模的探索性比较。此前 E2 替换概率变化和这里删 attack 训练行不是同一操作，不能由 E2 结果推断 n=200 一定更有效。n 越大对 attack 训练表的改动越大。

## 3. 文件

- `class_image_ablation.py`：选图、过滤训练表、attack 重训、v 与 J10−J00、汇总。同名替换之前的 E1/E2 脚本，旧结果文件保留，不覆盖。
- `class_image_ablation.slurm`：15 个任务 = 5 组（含 `none`）× 3 个 Stage-1 seed；每个任务一张 A100，跑全部 attack seed 与 32 个子集；可选第 5 个参数是 n。
- `test_class_image_ablation.py`：本地检查，不需要 Slurm。内容：同一张过滤后的表既用于训练也用于 adjoint；`none` 保留全部行；随机对照与被选图的类别数量一致且可复现；合成数据检查 `none` 和 `low_margin` worker。未在真实数据上运行所有任务或汇总。

## 4. 运行

```bash
pkg="$HOME/oct2-interface-case-study/experiments/pathway2_interface_case_study_v1"
frozen="/u/yli103/oct2-interface-lds-v1r1"
out="/u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_20261022T230209460267514Z"

for n in 50 200; do
  dest="/u/yli103/oct2-pathway2-runs/A/attack_train_prune_20261030_n$n"
  mkdir -p "$dest/logs"
  job_id=$(sbatch --parsable --output="$dest/logs/%A_%a.out" --error="$dest/logs/%A_%a.err" \
    "$pkg/class_image_ablation.slurm" "$pkg" "$frozen" "$out" "$dest" "$n")
  echo "$n $job_id" | tee -a "$dest/jobid.txt"
done
```

全部任务完成后汇总（两个 n 使用各自的目录）：

```bash
module load pytorch-conda/2.8
for n in 50 200; do
  dest="/u/yli103/oct2-pathway2-runs/A/attack_train_prune_20261030_n$n"
  python3 "$pkg/class_image_ablation.py" --repo "$frozen" --out "$out" --dest "$dest" --n-selected "$n" --collect
  cat "$dest/results.csv"
done
```

输出：`results.csv`（每组的 LDS）、`subset_values.csv`（每个子集的预测与真值）、每组每个 seed 的选图文件和 J00/J10 行；`none/seed*_replay.json` 是重放偏差。数组任务有失败时先查日志，不要用不完整的输出。

## 5. 先看什么

1. 四个删除组以同次运行的 `none` 为主基线，随机组作为描述性参照。每个随机组只抽一次，不是独立确认实验。
2. 看 `none` 和已存 0.262 的 LDS 差异，以及每个 Stage 1 seed 的重放偏差：逐 attack seed、类别、子集配对计算 J00、J10 和 ΔJ 的最大绝对差，还给出 vbar 和预测分数差异。微小的绝对差也可能改变排序，所以不能只靠一个绝对差判断 LDS 一致。
3. 如果 `none` 与已存结果有差异，可能原因包括硬件、软件或数值等，不能仅凭这个对照确定原因。原 v6 脚本默认也使用 A100。

## 6. 验证状态与限制

- 依据 v6 的 `attack4.py`、`core4.py`、`core6.py` 和接口 `analysis_if.py` 的真实接口实现；Python、Slurm 语法以及四项使用假 attack 的合成检查通过，包含 worker 和汇总。已用现有真实 `images.csv` 检查 50 和 200 的各组数量与类别匹配。
- 没有在 Delta 上用真实数据运行；没有运行时间数据，6 小时是初始申请。每个任务约 2 类 × 6 个 attack seed × 33 次 attack 训练加 12 次 Hessian，实际以日志为准。
- 删除 50 张可能产生较小变化；200 张是另一个规模，并不保证更有信息量。
- 选图依据来自同一批 32 个子集上的原实验，涉及的结果是探索性的。
