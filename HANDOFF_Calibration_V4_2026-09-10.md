# Cross-stage calibration v4 handoff

目的：把 v3 的可运行原型推进到可以在 Delta 开始真实 OCT 校准的版本。入口教程见 `START_DELTA_GITHUB_V4_CN.md`。基线提交 `e87f42d`，本次分支 `cross-stage-calibration-v4`。

## Hypothesis / Objective

H1：Eq.75–79 的连续 patient removal score 预测固定 target queries 上、matched OCT class 的稳定 logits CE 改变 **J10−J00**。主要问题始终是 value pathway。

H2：hybrid 预测 **J11−J00**，且应在相同有效样本上优于 relabel-only（J01−J00）。这是次要问题。J11−J10 是 relabel|P1，不是 continuous truth。

## Setup / Variables

默认校准：40k patient-complete subset 上重新生成 split；target/shadow size 参数各 2000，5 shadows；DME/DRUSEN 每类 4 位符合 5–15 张图像区间的病人。08 根据最稀缺类别的合格人数选择 1 个 shadow，冻结面板。Stage 1 seeds 42–46；共用 attack seeds 5101–5105；R=K=5；50 epochs。

原 target、非受影响 shadows 和 target queries 固定。删除采用 fixed_mask：前向 batch 和 RNG 流对齐，病人 loss 权重变为 1−α，分母保持原 batch size。filter_rechunk 仍保留作不同训练干预的 sensitivity。

## 实现变化

1. **CE 定义一致**：05 与 07 的 J_Q 都采用 logits 的 log-softmax CE；概率只用于 AUC/Brier 等指标。保存每个 condition/seed/query 的 log-probability。概率裁剪会改变极端 CE 的数值和导数，不能与推导中的 logits CE 混用。新 schema 会拒绝旧 CE 目录。
2. **逐 seed score**：为每个 attack seed 构造 v_k、h_k 并求解 w_k，保存 `ladder_attack_seed_rows.csv`；均值输出不提前丢弃 seed 维度。数学上线性保证均值等价，但单次 CG 无法恢复逐 seed 方差，因此 v4 明确承担 K 个 RHS 的求解成本。
3. **按计算依赖筛选**：L1/L2 lin、frozen_h 依赖 attack solve；L3 lin 额外依赖 score CG；L3 retrain/hybrid 只依赖 Δθ CG；L2 retrain/hybrid、frozen_self 不依赖上述 Hessian inverse。各层输出 total/finite/gate-passed/complete-panel 数量，微小样本也输出 coverage。
4. **统计口径**：09 从原始 paired seed effects 分解 crossed/nested 方差；crossed grand mean 的方差为 σS²/R + σA²/K + σSA²/(RK)。保留原始负分量并另外输出非负截断估计。缺格、K=1、R=1 明确标记，不伪造方差分解。CE/AUC/Brier 分别统计，不对 seed 平均概率求 AUC。
5. **配对与冻结**：面板必须含 raw indices 和正确图像数；恢复运行重新校验面板；拒绝重复/空 attack seed 列表；正式 runner 要求所有指定类别通过 baseline gate。no-op 检查全部 K 个 attack seeds。fixed_mask 另硬校验训练结束后的 RNG 指纹。
6. **数值与分析**：CG 重算真残差，处理零 RHS/非有限值；attack solve 失败不阻断独立的重训层。按类汇总、完整 Stage 1 seed 面板均值、共同有效行比较、hybrid 相对 relabel baseline 的 MAE 改善、可靠 damping-grid 比较。
7. **运行入口**：`10_run_calibration.py` 和 `10_calibration.slurm` 分阶段启动，记录软件版本、GPU、Git commit、命令和 job ID。生产默认 fixed_mask、crossed seeds、8 位病人；不自动提交正式大面板。

## Validation / Numbers

本地环境：Python 3.12.14，PyTorch 2.8.0+cpu，NumPy 2.3.5，SciPy 1.17.0，scikit-learn 1.8.0，无 CUDA。

- 8 个数学/配对测试：attack 有限差分、Stage 1 固定分母重训、非驻点 detach、CG 曲率/真残差/零 RHS、fixed-mask RNG/内容独立性、crossed seed 函数、panel handshake、稳定 CE。
- 7 个分析/launcher 测试：依赖 gate、完整面板、小样本 coverage、damping gate、已知 crossed ANOVA 数值、nested/欠识别情况、真实 JSON 配对与缺格、启动参数映射（部分检查合并在同一个测试方法中）。
- 最终完整链：08 → 05 → 07 → 09，full/dose=0.25/late 三种条件；另运行 06。使用 16×16 合成图像、160 个合成病人、2 Stage 1 epochs、5 attack epochs；full 是 2 位病人 × 2 Stage 1 seeds × 2 attack seeds，dose/late 使用 1 个 Stage 1 seed。
- 逐 seed score 平均与主表一致；J00 复现容差 1e−6；NPZ 的逐 query log-probability 能以 1e−12 容差还原逐 seed CE；dose 缩放、window N/A、独立层保留和 crossed 方差全部通过。
- 极端 CE 回归例：稳定 logits CE=30.000000，旧 float32 概率裁剪 CE=15.942385。
- 合成 full 的 4 行均未通过 attack/Stage 1 数值 gate；L2 retrain 的 4 行全部保留。这验证筛选依赖，不能据此评价 L3 的质量。

此前另完成了 128×128 合成图像的 05 full 训练与删除链，之后停止了耗时的 CPU 07。该日志仅作补充：当时尚未加入最终的稳定 CE 修改，不能当成最终 v4 完整链的证据。交付中正式证据以 `final_chain` 为准。

## Results / Analysis / Next steps

已经验证实现、数据对齐、数值保护和输出能衔接；还没有真实 50-epoch OCT 的 score 质量结论，也没有在 Delta 提交作业或推送 GitHub。上传 bundle 后先运行 tests/preflight/baseline/truth/score。

真实校准先判断 CE/AUC 的效应尺度与 seed 方差、数值 gate coverage、JVP 投影幅度、L1→L2→L3 误差增量。L3 失败不等于理论推导错误；尤其是非驻点有限步 Adam 与静态阻尼隐式近似之间仍可能有差距。按数值诊断选择 damping，再冻结正式配置，不按相关性调参。

Adam trajectory score（Eq.42+74）仍未实现。early/late 只跑 value truth 和有真实 ΔP/Δθ 的 oracle 层，静态 L3/frozen N/A。正式推断还需要另行冻结患者/seed 面板和正确的聚类重采样；09 的描述性 ANOVA SE 不替代 patient/query population CI。
