# OCT MIA 跨阶段实验：正确性审计 + 正式实验设计（含 score 实现）

日期：2026-09-09
依据：`continue_pathway2.pdf`（Eq. 1–87）、`HANDOFF_End_to_End_Patient_LOO_05_COMPLETE_CN_2026-09-09.md`、GitHub `shan927150/oct2` 代码（commit `21ca1b2`）
状态：审计完成；新增 3 个脚本（06 / 07 / tests）+ 05 脚本扩展；正式实验分 4 个 phase

---

## 0. 一句话结论

05 pilot 的 **estimand、四个反事实条件、配对训练顺序、gate、no-op replay 和 primary endpoint 都与 PDF 推导一致，实现是正确的**；但有两处会直接影响"结果是否可信"和"score 该预测什么"：

1. **配对不是完全配对**：删除病人后 batch 组成和 dropout 随机流从第 2 个 epoch 起就与 baseline 不同。no-op floor = 0 只证明"同数据可完全复现"，**不能**给出 ΔCE 的噪声底。当前 |Δ_full| ≈ 0.006–0.018，而 baseline CE 的跨 seed SD 是 0.006（DME）/ 0.028（DRUSEN），**必须先跑 placebo 才能说效应是真的**。
2. **PDF 里定义的 score（Eq. 79）从未被实现过**。仓库里测试过的只有 frozen proxy（Eq. 80，`02_pair_separation_test.py` 里的 `C_i`），PDF 第 10 节自己就说它 "is not Equation (79)"。所以"score 表现不好"这个结论目前只对 frozen proxy 成立，对我们定义的 score 还没有任何证据。本次交付的 `07_cross_stage_score_ladder.py` 第一次实现了 Eq. 75–79（implicit 形式），并且**只用 pilot_v3 已有的 checkpoint 和 NPZ 就能离线跑完**，不需要重训 Stage 1。

先跑 §3.3 Phase 0（离线、几分钟），再决定 Phase 1–3 的规模。

---

## 1. 正确性审计（代码 ↔ 公式 ↔ handoff）

### 1.1 与推导一致的部分（无需改动）

| 项目 | PDF | 代码 | 结论 |
|---|---|---|---|
| Stage 1 输出 $p_{sj}=\mathrm{softmax}(F_{\theta_s}(x_{sj}))$ | Eq. 3 | `get_predictions` → `_predict_probs` | ✓ |
| Attack 按 class 分别训练，$A_{\phi_c}:\mathbb R^4\to\mathbb R^2$ | Eq. 5–7 | `evaluate_attack_condition` 按 `cls` 循环，MLP 4→64→2 | ✓ |
| $J_{Q,c}$ = 固定 target query 上的 CE | Eq. 8 | `binary_metrics` → `log_loss` on `target["x"][te]` | ✓ |
| Matched-class endpoint = $\omega_c=1$ 的 Eq. 9 | Eq. 9 | `endpoint_deltas(..., matched_class)` | ✓ |
| 四条件 $J_{ab}$ 只改 $P$ 或 $M$，row universe / query 固定 | Eq. 83–87 | `replace_affected_vectors` + `membership_after_removal`，两者都只动 affected shadow 的行 | ✓ |
| 同 seed 内 attack 训练完全配对 | Eq. 23 finite unrolling 假设 | `attack_seed = seed*100+cls`，epoch order 由 `seed+900000` 生成，与 $P,M$ 无关 | ✓ |
| Stage 1 epoch 顺序配对，LOO 只过滤 | §2 "fixed minibatch order" | `make_epoch_orders` + `excluded` 过滤 | ✓（但见 1.2-A） |
| no-op replay floor = 0 | — | `noop_records` exact | ✓ |
| Patient-disjoint split | — | `validate_patient_split` 硬性 raise | ✓ |
| 选人规则与 outcome 无关 | — | `choose_patients` 只用 seed 42006 与图像数区间 | ✓ |

`07` 在真实 pilot_v3 目录上运行时会重新算一遍 J00（`J00_recomputed`），在合成数据上已验证与 `runs/*.json` 里的 J00 **逐位相同**，说明 attack 阶段离线复现是精确的。

### 1.2 需要注意 / 需要修正的地方

**A. 配对训练不等于配对噪声（最重要）**
`train_classifier_from_orders` 过滤掉病人图像后，同一 epoch 内其后的所有 batch 边界整体前移，最后一个 batch 变小；`SmallCNN` 有 `Dropout(0.2)`，其 RNG 消耗量依赖 batch 形状，所以从第 2 个 epoch 起 dropout mask 流就与 baseline 脱钩。于是
$$\Delta_{\text{observed}} = \Delta_{\text{patient}} + \Delta_{\text{trajectory reshuffle}}$$
第二项在 no-op replay 里恒为 0（同数据同流），无法被现有 gate 检出。PDF Eq. 62–69 的 finite-unrolling 假设的是"$B_t$ 固定、只是 $i$ 不在里面"，也就是**不含**第二项。
→ 处理：05 新增 `--placebo_replays N`：不删任何人，只换 dropout/trajectory RNG（模型初始化不变），得到 $\Delta CE^{\text{placebo}}$ 的分布，作为判定病人效应的噪声底。判定规则见 §3.2。

**B. $J_{10}$、$J_{01}$ 是"标签-向量不一致"的反事实**
$J_{01}$：病人的行仍是 baseline 的 member-like 高置信向量，却被标成 nonmember；$J_{10}$ 反之。这等价于往 attack 训练集里注入 ~10 行标签噪声，所以 interaction 大是这种三项分解的结构性产物，不完全是"机制发现"。建议同时报告**顺序分解**（无 interaction 项）：
$$\Delta_{\text{full}} = \underbrace{(J_{10}-J_{00})}_{\text{value, continuous}} + \underbrace{(J_{11}-J_{10})}_{\text{relabel}\mid P_1}$$
这正是你提的"J10 和 J11 做差值"。`06` 已输出 `relabel_given_P1_ce = J11 − J10`、`value_given_M1_ce = J11 − J01` 以及两者平均的 Shapley 份额。**continuous score 的目标量按 PDF §11 仍是 $J_{10}-J_{00}$**，不变。

**C. 仓库里的 slurm 文件与正式 run 不一致**
`05_end_to_end_patient_loo.slurm` 仍是 `--affected_shadow 0`、`pilot_v2`；正式结果是 shadow 1 / `pilot_v3`（job 21331407）。请把实际命令提交进仓库，否则别人无法复现。

**D. Δ_value 可能低于分辨率**
handoff §14：value-only 均值 DME −0.00024、DRUSEN −0.0045。如果 placebo floor 的 SD 在 0.003–0.005 量级，那么 **continuous pathway 在 patient 级别可能根本没有可测目标**，任何 score 对 Δ_value 的相关性都会被噪声淹没——这与 score 好坏无关。`07` 的 `--attack_seed_reps` 能先把 Stage 2 的那部分噪声量化出来（很便宜）。

**E. Implicit 形式的两个假设在当前训练配置下不严格成立**
- Attack：50 epoch Adam lr=0.01、无 L2，$\nabla_\phi L_A(\phi^*)\neq 0$，Hessian 可能有负特征值。07 报告 `train_grad_norm`、`hessian_eig_min/max`，并用 $\gamma_A$ 阻尼（Eq. 21）。attack 只有 450 个参数，Hessian 直接显式求解，无 CG 误差。
- Stage 1：Adam + dropout + weight decay，$\theta_T$ 不是 $L_s$ 的驻点；07 用 eval-mode Hessian（Eq. 54，含 $\lambda I$）+ CG 阻尼 $\gamma_s$，报告 $\|\nabla L_s(\theta_T)\|$。真正忠实的是 unrolled 形式（Eq. 74），需要轨迹 checkpoint——这就是 §3.4 节点截断实验要解决的。

**F. CE vs AUC**
score 通过 $J_Q$ 定义，而 $J_Q$ 就是 CE（Eq. 8），所以**score 按构造只预测 CE**。AUC 是 rank 统计量，不可微；而且 DME 的 107×95 对里翻转一对只改 AUC $\approx 10^{-4}$，pilot 的 ΔAUC（$-0.0016\sim+0.0006$）本来就在分辨率边缘。
→ Primary 保持 matched-class CE；AUC 作 secondary。如果审稿要求 AUC 上的 influence，把 Eq. 8 换成可微的 pairwise surrogate $J^{\text{pair}}_Q=\frac{1}{|Q^+||Q^-|}\sum\log\sigma(\text{logit}_{q^+}-\text{logit}_{q^-})$，07 里只需改 `attack_implicit_v` 中 `Jq` 一行，其余链条不变。另外建议加 Brier score 作为对 CE 极端值不敏感的 robustness endpoint。

**G. 小问题**
- `seed_sign_concordance` 无方向（handoff 已指出）→ 06 补齐 majority direction、SD、range。
- `endpoint_deltas` 里 macro 是对 qualified classes 平均，若只 DME/DRUSEN 通过 gate，macro 与 paper 里的 4 类 macro 含义不同，写报告时要注明。
- `frozen_self`（Eq. 80）在 07 里保留作对照，便于直接回答"新 score 比旧 proxy 好多少"。

---

## 2. 我们定义的 score 到底预测什么

**Score（implicit 形式，07 实现）**，对 affected shadow $s$、病人 $i$ 的图像集合 $\mathcal I_i$，matched class $c$：
$$
\begin{aligned}
q_c &= \nabla_\phi J_{Q,c}(\phi_c^*) & \text{(Eq. 75)}\\
u_c &= (H_{A,c}+\gamma_A I)^{-1} q_c & \text{(Eq. 21)}\\
v_{sj} &= -B_{sj}^{\top}u_c = -\frac{\partial}{\partial p_{sj}}\big[u_c^{\top}\nabla_\phi L_{A,c}\big] & \text{(Eq. 76)}\\
h_s &= \nabla_\theta \sum_{j} v_{sj}^{\top} p_{sj}(\theta) & \text{(Eq. 77)}\\
w_s &= (H_s+\lambda I+\gamma_s I)^{-1} h_s & \text{(CG)}\\
\widehat{\Delta}_{\text{value}}(i) &= \frac{1}{n_s}\sum_{r\in\mathcal I_i} w_s^{\top} g_{sr} & \text{(Eq. 61, 一阶可加)}
\end{aligned}
$$
它估计的是 **$J_{10}-J_{00}$**（membership label 与 row 组成固定，PDF §11）。它**不**估计 $J_{11}-J_{00}$。

**能否反映 CE 上的 influence？** 按定义能——目标量就是 CE。真正的问题是链条哪一环失真。07 的 "oracle ladder" 用 pilot_v3 已有的真值逐环替换：

| 层 | 用真值替换什么 | 预测量 | 检验的环节 |
|---|---|---|---|
| actual | — | $J_{10}-J_{00}$ | 真值 |
| L1_lin | 真 $P_1$（NPZ 里的 `p_loo`） | $\sum_j v_j^{\top}(p^{(1)}_j-p^{(0)}_j)$ | Attack 阶段线性化 + implicit（Eq. 16–20） |
| L2_lin | 真 $\Delta\theta=\theta_{\text{LOO}}-\theta_{\text{base}}$（两个 .pt） | $h_s^{\top}\Delta\theta$ | 再加上 $p(\theta)$ 线性化（Eq. 47） |
| L2_retrain | 真 $\Delta\theta$ | 用 $\hat P_1=P_0+J_p\Delta\theta$ **精确重训 attack** | 只剩 $p$ 线性化 |
| L3_lin | 无 | $\widehat\Delta_{\text{value}}$ 上式 | 完整 score（再加 Eq. 56 的 Stage 1 implicit） |
| L3_retrain | 无 | 用 $\hat\theta$ 线性化 $P$ 后精确重训 attack | Stage 1 implicit 的质量（`dtheta_cosine` 直接给 $\cos(\Delta\theta,\hat{\Delta\theta})$） |
| **L3_hybrid** | 无 | 同上但 attack 用 $M_1$ 重训 | **预测 $J_{11}-J_{00}$（Δ_full）**，Stage 2 离散部分精确处理 |
| frozen_h | — | $\sum_r h_s^{\top}g_{sr}$ | 单 checkpoint TracIn 风格 |
| frozen_self | — | Eq. 80 | 旧 proxy 对照 |

判读：从上到下相关性在哪一层塌掉，问题就在哪一环。特别是 **L3_hybrid**：因为 Stage 2 只有 450 参数、几秒钟就能重训，把 relabel 与 interaction 精确算出来是免费的；只有 Stage 1 才需要 continuous 近似。这是在"score 只对 Δ_value 有定义"和"真实效应主要在 Δ_full"之间最自然的桥。

预期风险：pilot 里 deleted patient rows 的 mean JS = 0.35，说明被删图像的 $p$ 从 memorized 变到 nonmember-like，是**大幅非线性变化**，$P_0+J_p\Delta\theta$ 对这些行可能不准（07 输出 `p1hat_true_mean_js_deleted` vs `p1_true_mean_js_deleted` 直接对比）。如果 L2 就塌，就进入 §3.4 的节点截断。

---

## 3. 正式实验设计

### 3.1 三个噪声源与 seed 数量

| 噪声源 | 控制变量 | 成本 | 处理 |
|---|---|---|---|
| Stage 2 attack 训练随机性 | attack seed | 每次 ~1–2 s（GPU） | `--attack_seed_reps K`（05 与 07 均支持），每个条件取 K 个 attack seed 的均值；K=10 把 Stage 2 噪声 SD 缩小 √10 |
| Stage 1 trajectory（init / order / dropout） | Stage 1 seed | 每次 ~15–20 s（pilot 566 s / ~27 次训练） | seeds 数由 placebo 决定，见下 |
| 病人抽样 | patient set | — | 扩大到每类 ≥16 人，分层随机 |

Seed 数量的决定规则（先跑 Phase 1 拿到 placebo）：令 $\sigma_p$ = placebo ΔCE 的 SD（同一 seed 内、无删除、换 trajectory RNG），$\delta$ = 想分辨的最小病人效应（pilot 里 DRUSEN 中位数约 0.014，DME 约 0.007）。要 paired mean 的 SE 达到 $\delta/2.5$，需要
$$n_{\text{seeds}} \ge \left(\frac{2.5\,\sigma_p}{\delta}\right)^2 .$$
例：$\sigma_p=0.005,\ \delta=0.007 \Rightarrow n\ge 3.2\to 5$；$\sigma_p=0.010 \Rightarrow n\ge 13$。**若 $\sigma_p$ 超过 0.01，就不要再堆 seed，而应先降 CE 噪声**（attack_seed_reps、Brier、或 attack 加小 L2 让 $\phi^*$ 更稳定——后者会改 estimand，需要重新 gate）。

### 3.2 Endpoint 与判定标准（预注册）

- Primary：matched-class CE，$\Delta_{\text{full}}$ 与 $\Delta_{\text{value}}$；顺序分解 $J_{11}-J_{10}$；Shapley 份额。
- Secondary：AUC、balanced acc、Brier。
- 病人效应存在性：$|\bar\Delta_{\text{full}}(i)| > 2\,\sigma_p/\sqrt{n_{\text{seeds}}}$ 且 seed 方向一致 ≥ 80%。
- Score 有效性（对 n 个病人）：Spearman $\rho$（score vs Δ）≥ 0.5 且 patient-level bootstrap 95% CI 不含 0；sign agreement ≥ 0.75。分别对 (L3_lin, Δ_value)、(L3_hybrid, Δ_full) 评估，**不要**拿 L3_lin 对 Δ_full。
- 所有相关性同时报告 signed 与 |Δ| 两种。

### 3.3 Phase 计划

**Phase 0 — 离线（今天，Delta 上 GPU 节点几分钟，不重训 Stage 1）**
```bash
cd ~/oct2
python scripts/cross_stage/06_pilot_seed_stats.py \
    --pilot_dir results/cross_stage_patient_loo_05_pilot_v3
python scripts/cross_stage/07_cross_stage_score_ladder.py \
    --pilot_dir results/cross_stage_patient_loo_05_pilot_v3 --data_dir ./data \
    --attack_seed_reps 10 --cg_iters 100
```
产出：每病人 ΔCE 的 SD/range/majority、$J_{11}-J_{10}$；ladder 各层相关性；Stage-2-only 噪声 SD（`stage2_only_noise`）；`dtheta_cosine`。
决策：
- 若 `attackseed_value_sd` ≈ |Δ_value|：value pathway 在 Stage 2 噪声内 → 主目标改为 Δ_full（L3_hybrid）。
- 若 L1 相关高而 L2 低：问题在 $p(\theta)$ 线性化（deleted rows 非线性）→ 进入 Phase 3。
- 若 L1 就低：attack 端 implicit 假设不成立 → 给 attack 加 L2 或改 unrolled（Eq. 42，attack 便宜可存全轨迹）。

**Phase 1 — 噪声底 + 复现 gate（新目录 `pilot_v4`，~30 min）**
```bash
python scripts/cross_stage/05_end_to_end_patient_loo_pilot.py \
  --output_dir results/cross_stage_patient_loo_05_v4 --data_dir ./data \
  --n_total_samples 12000 --target_data_size 800 --shadow_data_size 800 \
  --n_shadow 5 --affected_shadow 1 --n_patients 8 --classes 1 2 \
  --min_patient_images 5 --max_patient_images 15 --seeds 42 43 44 45 46 \
  --shadow_epochs 50 --attack_epochs 50 --noop_replays 1 --noop_tolerance 0 \
  --enforce_attack_gate --enforce_noop_gate \
  --attack_seed_reps 10 --placebo_replays 5 --save_epoch_checkpoints 25 50 \
  --baseline_only
```
产出 `placebo_replays.json`（5 seeds × 5 replays = 25 个 placebo ΔCE/类）→ $\sigma_p$。同时 baseline 的 epoch-25/50 checkpoint 为 Phase 3 备用。

**Phase 2 — 正式 LOO（同目录去掉 `--baseline_only`）**
- 同样 8 人先跑，与 pilot_v3 对照（新 seeds 45、46 是 replication）。
- 再扩到 `--n_patients 32`（每类 16），需要新目录；若 DME/DRUSEN 合格病人不足，把 `--max_patient_images` 放到 20 并在报告中把图像数作为协变量。
- 增加 affected shadow：对 shadow 2、3 各跑一遍（各自新目录），要求同一选人规则。
- 估算：每个 patient-seed 4 次 attack 训练 × K=10 + 1 次 Stage 1 ≈ 60 s → 32 人 × 5 seeds ≈ 2.7 h/shadow。
- 每个新目录跑完后立即跑 06、07（07 支持 `--seeds` 子集）。

**Phase 3 — Training-stage 节点截断（你提的"前一半 + 后一半 + 结合"）**
```bash
# late-half deletion: 病人只在 epoch 25-49 被删；epoch 0-24 与 baseline 逐位相同
python scripts/cross_stage/05_end_to_end_patient_loo_pilot.py ... \
  --output_dir results/cross_stage_patient_loo_05_v4_late --removal_epochs 25:50 ...
# early-half deletion: 只在 epoch 0-24 被删，之后加回
python scripts/cross_stage/05_end_to_end_patient_loo_pilot.py ... \
  --output_dir results/cross_stage_patient_loo_05_v4_early --removal_epochs 0:25 ...
```
因为训练是逐位确定的，`removal_epochs 25:50` 的前 25 个 epoch 与 baseline 完全相同，**不需要保存 optimizer state 就实现了从节点 $\theta_{25}$ 开始的截断**。三组真值：
$$\Delta^{\text{early}},\quad \Delta^{\text{late}},\quad \Delta^{\text{full}}\ (=\text{Phase 2}),\qquad \text{非可加残差 } r=\Delta^{\text{full}}-\Delta^{\text{early}}-\Delta^{\text{late}}.$$
Score 端对应三种截断：
- late：Eq. 69/74 只对 $t\ge 25$ 求和，adjoint 从 $\theta_{50}$ 回传到 $\theta_{25}$，所用 checkpoint 就是 `--save_epoch_checkpoints` 存的（unrolled 需要更密的 checkpoint，建议 Phase 3 用 `--save_epoch_checkpoints 25 30 35 40 45 50` 的 TracIn-CP 近似）；
- early：同样公式对 $t<25$ 求和，但 adjoint 要穿过后半段（$a_{25}=\prod_{t\ge25}(I-\eta_tH_t)^{\top}h_s$），这一项衡量"后半段训练把前半段影响遗忘了多少"；
- 结合：两段相加 vs $\Delta^{\text{full}}$，残差 $r$ 告诉你线性叠加在哪个尺度失效。
预期：late 段视野短、非线性弱，score 应最准；如果连 late 都不准，说明问题不在时间视野而在 attack 端或 $p$ 线性化（回到 Phase 0 的 ladder）。

**Phase 4 — 分析与报告**
- Patient-level bootstrap（重抽病人）给 $\rho$ 的 CI；层级模型 `Δ ~ class + n_images + (1|patient)`。
- 报告顺序分解与 Shapley 份额，不再单独强调 interaction。
- 所有 "支持/阻碍 attack" 的判定都以 placebo floor 为参照。

### 3.4 必须保持的控制（沿用 handoff §25 Priority 4）
Fixed target queries、fixed attack row universe、per-class gate、paired epoch orders、no-op gate、fixed target 与 unaffected shadows、deletion-size 控制、选人与 outcome 无关。新增：**placebo floor**、**attack_seed_reps 固定为同一 K**、**节点截断只通过 `--removal_epochs`，不手动改 checkpoint**。

---

## 4. 本次交付的代码

| 文件 | 作用 | 验证 |
|---|---|---|
| `scripts/cross_stage/06_pilot_seed_stats.py` | 从 `patient_seed_summary.csv`（+ `runs/*.json`）算每病人 SD/var/min/max/majority、$J_{11}-J_{10}$、$J_{11}-J_{01}$、interaction、Shapley 份额；输出 `patient_seed_stats.csv/json` | 合成 CSV 通过 |
| `scripts/cross_stage/07_cross_stage_score_ladder.py` | Eq. 75–79 implicit score + oracle ladder + `frozen_h` / `frozen_self` + `--attack_seed_reps` Stage-2 噪声；只读 pilot 目录 | 合成数据端到端跑通（CPU）；`J00_recomputed == J00` 逐位相同 |
| `scripts/cross_stage/tests/test_score_ladder_math.py` | 有限差分验证：attack implicit $v$（相对误差 0.6%，corr 0.99999）；Stage 1 implicit $(1/n)w^{\top}g_i$ vs 精确 refit | 通过 |
| `scripts/cross_stage/05_end_to_end_patient_loo_pilot.py`（扩展） | 新增 `--removal_epochs start:end`、`--attack_seed_reps K`、`--placebo_replays N`、`--save_epoch_checkpoints ...`；默认值下行为与 pilot 逐位相同；旧目录 resume 兼容 | 合成数据 resume 无变化 + 新 flag 跑通（见 §5） |

07 的默认阻尼：$\gamma_A=10^{-3}$、$\gamma_s=10^{-2}$，CG 100 步、tol $10^{-4}$。在 A40 上 Stage 1 HVP 每步约 0.1–0.3 s（800 张 128×128），每 seed·class 一次 CG + 每病人一次 CG，总量分钟级。若 `cg_final_rel_residual` > 0.1，增大 `--damping_shadow` 或 `--cg_iters`。双反向传播比较吃显存/内存，`--hvp_batch`（默认 64）可以调小；`--classes 1` 可以只跑一个类。Delta 上直接 `sbatch --export=ALL,PILOT_DIR=results/cross_stage_patient_loo_05_pilot_v3 scripts/cross_stage/07_score_ladder.slurm`（会先跑 06 再跑 07）。

已知输出字段（`score_ladder/ladder_rows.csv`）：`actual_value / actual_full / actual_relabel_given_P1`、`L1_lin_value`、`L2_lin_value / L2_retrain_value / L2_hybrid_full`、`L3_lin_value / L3_retrain_value / L3_hybrid_full`、`frozen_h / frozen_self`、`dtheta_cosine`、`p1hat_true_mean_js_deleted` vs `p1_true_mean_js_deleted`、`attackseed_value_sd / attackseed_full_sd`；`ladder_summary.json` 里是 patient-seed 级和 patient 均值级的 Spearman / Pearson / sign agreement，以及 attack Hessian 特征值、$\|\nabla L_A\|$、$\|\nabla L_s\|$、CG 残差等诊断。

---

## 5. 下一次对话首先做什么

1. 把 Phase 0 的两条命令在 Delta 上跑完，把 `patient_seed_stats.csv` 与 `score_ladder/ladder_summary.json` 贴回来。
2. 根据 §3.3 Phase 0 的决策规则确定主目标是 Δ_value 还是 Δ_full。
3. 提交 Phase 1（`--baseline_only` + placebo），拿到 $\sigma_p$ 后用 §3.1 公式定 seed 数。
4. 修正仓库 slurm 里的 `affected_shadow` / 输出目录，并把 05 扩展版与 06/07/tests 提交进仓库。
