# 设计：按 attack 类别和按图像集的直接消融（接口 v1r1 之上）

日期：2026-10-30。来源：10-30 讨论。被删行承载桥接差距（row 1 0.262，row 3 0.730）之后，用最直接的方式看"哪一类、哪一批图限制了相关"：把怀疑的部分拿掉或换成重训值，看 LDS 怎么变。E1 对应 meeting 的第一、二条（每类的相关，去掉一类的目标），E2 对应第三条（去掉或替换图）。E1 先看，E2 随后；`--e1-only` 只出 E1。

## 1. 问题

1. 合在一起的 attack 目标（DME 0.635、DRUSEN 0.365）之下，每个类别单独的桥接相关是多少？只看其中一类的目标时，相关是多少？
2. 去掉怀疑有问题的图，LDS 变化多少？把这些图换成重训值变化多少？这些图留在 IF 而其余换成重训值，LDS 变化多少？

## 2. 复用的数据（不训练、不求解）

- v3 子集概率（重训真值）、`interface.npz` 的 IF 预测、v6 的 attack 真值，面板 seed 42–44，γ = 0.03。
- **类别 sensitivity**：v6 每个 Stage-1 seed 的 `v.npz` 里按 attack 类别存了各类的 v_c（attack seed、类别、γ 三个维度）。脚本直接读 DME 和 DRUSEN 的 v_c，取 attack seed 平均，乘 ω_c，得到类别分数 S_{j,c} = 面板平均 Σ v_c·Δp。合在一起的 v̄ = Σ ω_c v_c 与报告里用的 v̄ 逐项核对。类别分数不再依赖"按图像标签切行"。
- **类别真值**：v6 `truth` 里每个 (子集, Stage-1 seed) 的各 attack seed 行，按 `cls` 分开，取 J10 − J00 的均值。合在一起的真值是 Σ ω_c × 类别真值，和报告里的写法逐项核对。
- 单类分数对单类真值（对角线）才是"只看这一类目标"的结果。分数与真值同时换了，所以类别之间的比较只说明在这批子集上哪一类目标更容易预测，不单独说明另一类是原因。csv 里还有混合配对（某类分数对其他真值），不是主表。
- 说明：Stage-1 的四个图像类别（CNV、DME、DRUSEN、NORMAL）和 attack 目标的类别是两件事。当前 attack 目标只有 DME、DRUSEN 两类，E1 的"类别"指后者。各图像类别的概率接口 LDS 在 10-26 的结果里已有，不在这次重算。

## 3. 实验

**E1 按类别。** 分数取"合在一起、DME 的 v_c、DRUSEN 的 v_c"，真值取"合在一起、DME、DRUSEN"。每个分数报告：
row 1 全 IF；row 2 全重训；row 3 被删行重训；row 4 保留行重训；row 5 nonmember 行重训；被删行部分单独（IF、重训）；保留加 nonmember 部分单独。
主表读对角线：合在一起对合在一起，DME 对 DME，DRUSEN 对 DRUSEN。

**E2 按图像集。** 图像集只取 attack 活跃图（469 张）。选择标准不使用重训值：
- 最低 margin 的 n 张，n = 20、50、100、200、300；
- margin 低于 8、10、12；
- |a_IF| 最大的 n 张；
- 梯度范数最小的 n 张（给 `--features` 时；按 train_position、raw_index、cls 三项对齐，任一项不一致就报错）；
- DME 类、DRUSEN 类的活跃图；
- 对照：|a_true| 最大的 n 张（用了重训值选，偏乐观，标记为 ORACLE）。

每个集合做三种操作，真值始终是合在一起的 attack 真值：
- **drop**：这批图自己的行从分数里去掉（所有子集，被删和保留两种用法都去掉），报告 row 1、row 2、row 3 三种。这是改预测分数，没有改真值，不等于"去掉某类 attack loss"；
- **fix**：row 1 里只有这批图的被删行换成重训值；
- **leave**：row 3 里除这批图外的被删行都换成重训值，这批图保持 IF。

随机对照：同样大小从活跃图里抽 300 次，种子 20261030，报告均值、5% 和 95% 分位，以及该集合的 LDS 高于多少比例的随机集合。DME 类、DRUSEN 类这两个整类集合的类别构成和随机集合不同，不做随机对照（对应列为空）。

## 4. 怎么读（跑之前写定）

- E1：看 row 1 与 row 3 在三个目标上的差，被删行的差距落在哪一类；看单类 row 1 与合在一起 0.262 的相对大小。
- E2：每个集合与同样大小随机集合的均值和 5% 到 95% 范围并排看，作为描述性参照。leave 明显低于随机范围，说明这批图留在 IF 时损失的相关比随机选的多；fix 或 drop 明显高于随机范围，说明处理这批图比随机处理更有效。范围是随机选图的经验分布，不是置信区间。
- ORACLE 行使用重训结果选图，只作描述性对照；按 |a_true| 选图不保证最大化子集 LDS，不能把它说成 LDS 上界。
- 所有数字是同一批 32 个子集上的诊断，桥接与真值共享 attack seed（v6 估计约偏高 0.1）；图像集和操作较多，个别集合落在随机范围之外不单独作结论。

## 5. 文件与运行

包：`experiments/pathway2_interface_case_study_v1/`

- `class_image_ablation.py`：E1、E2 和核对值。
- `class_image_ablation.slurm`：CPU，4 核，16G，1 小时。
- `test_ablation.py`：合成数据检查，本地运行，不是提交前置步骤。

```bash
pkg=$HOME/oct2-interface-case-study/experiments/pathway2_interface_case_study_v1
out=/u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_20261022T230209460267514Z
frozen=/u/yli103/oct2-interface-lds-v1r1

sbatch --output="$out/class_image_ablation-%j.log" "$pkg/class_image_ablation.slurm" \
  "$pkg" "$frozen" "$out" --features "$out/case_study/gamma0.03_k100/features_merged.csv"
```

上述命令在同一个 CPU 作业中依次运行 E1 和 E2，`--features` 提供梯度范数。若单独运行 E1，可通过 Slurm 的前三个位置参数后传入 `--e1-only`。

输出：`$out/class_image_ablation/gamma0.03/{checks,e1_class_lds,e2_image_sets}.csv`。日志开头是核对值：

- `row1_lds`、`row3_lds` 对照已有 bridge 表的 0.262、0.730，说明口径与面板和之前一致；
- `class_scores_sum_to_row1`、`class_truths_recombine`、`class_vbar_sum_vs_mixed_vbar` 三个 max_abs 接近 0，说明类别拆分能还原原来的合在一起的分数、真值和 v̄；
- `vbar_DME/DRUSEN_max_abs_on_rows_of_other_classes` 给出各类 v_c 落在其他图像类别行上的最大值，这是 v_c 在行上分布的直接读数。

## 6. 验证状态

- 合成数据：row 1 与 row 3 和逐子集循环的写法一致到 1e-16；fix 全部活跃图等于 row 3，leave 全部活跃图等于 row 1，drop 空集等于 row 1 和 row 3；类别分数相加等于 row 1；类别 v_c 泄漏到其他类行时核对值会显示；特征文件的 raw_index、cls、行数或缺失值不一致时报错；整类集合没有随机对照。
- 端到端：用合成的 `analysis_if` 桩数据（含各类 `v.npz`）跑完整的 `main`，`--e1-only` 和完整两种都跑通。真实数据的读取沿用 `case_study.py` 和 `logit_diag.py` 里已跑通的同一套调用，`v.npz` 的读取与 `analysis_if.vbar_from_v6` 用同一个目录函数和同样的布局检查。
- 真实数据没有在这里运行。
