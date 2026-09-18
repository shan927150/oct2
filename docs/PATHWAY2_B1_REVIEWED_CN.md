# B1 / B2 公式核对与修订说明（2026-09-18）

审阅对象：用户提供的 B1-smoke `631ac3f`，父提交为已验收 E0/B0 代码 `0e1cece`；公式来源为 `continue_pathway-2.pdf`，共 7 页。本修订只推进 B 分支，不修改冻结的 baseline、05 原训练函数、原 truth 或 A 分支。

## 判断

保留 B1 → B2 路线。已有小模型数据支持“某些短前缀在 float64 下可以验证切向传播，并且结果对计算精度敏感”，不能据此断言“原 float32 全流程的导数已经正确”“切向量寄生在死坐标”“式 (74) 前提已被破坏”或“float32 永远无法测量这个导数”。Delta 的真实模型量级仍需本次实验。

E0/B0 已验收执行与输入完整性。B0 的四个 cell 未找到稳定的相邻剂量区间，因此 B1 当前要回答的是局部响应在哪个训练前缀、哪个剂量与精度下能够被验证；不能直接跳到完整删除预测。E0 的残差分解也不能简化成“只有没收敛一个原因”。

## 对照实际公式

**符号与归一化。** 原实现按病人 downweight，固定 minibatch 原分母：

\[
L_{t,\alpha}(\theta)=\frac1{|B_t|}\sum_{i\in B_t}
(1-\alpha\mathbf1_{i\in p})\ell_i(\theta)+\frac\lambda2\|\theta\|^2.
\]

论文式 (51) 为平均风险外加 `epsilon * loss_i`；式 (64) 的直接项带 `1/|B_t|`，对应 batch 内相对样本权重。因此这两处 epsilon 的归一化应明确区分：全量 batch 下 `epsilon_global = epsilon_relative / N`。本实现始终使用相对删除强度 alpha，既不再额外除 N，也不把删除方向当成 upweight 方向。对一个病人，式 (74) 的相对 upweight 各样本方向之和取负，才对应这里的 `d/dalpha`。

令 `w=(theta,m,v,step)`，`u=dw/dalpha`。式 (73)–(74) 的前向对偶是：

\[
u_{t+1}=D_wV_t\,u_t+\partial_\alpha V_t,\qquad u_0=0.
\]

耦合 weight decay 必须进入梯度与导数：

\[
g_t=\nabla_\theta L_{t,0},\qquad
q_t=\frac{dg_t}{d\alpha}=H_tu_{\theta,t}
-\frac1{|B_t|}\sum_{i\in B_t\cap p}\nabla\ell_i.
\]

这里 `H_t` 已包含 `lambda I`。代码用 `torch.func.jvp` 计算梯度映射的切向量 q，再传播 Adam 的 theta/m/v 三组切向量；计步器无扰动。每步 detach，AD 图不随轨迹长度累积；保存的少量前缀状态与固定随机掩码单独计入存储。

**第一步 epsilon 机制。** 零初始 moments 时，式 (38) 化为：

\[
\theta_1=\theta_0-\eta\frac{g}{|g|+\epsilon},\qquad
\frac{d\theta_1}{d\alpha}=-\eta\frac{\epsilon}{(|g|+\epsilon)^2}\frac{dg}{d\alpha}.
\]

`epsilon/(|g|+epsilon)^2` 是归一化响应关于 g 的斜率；参数更新的增益上界为 `eta/epsilon`，实际病人敏感度还乘 q。`g≈0` 可能来自真实小梯度、batch 内抵消或 weight decay 抵消，不等于“死坐标”。新 G5 同时记录 g、q、学习率、第一步解析式核对，以及小梯度坐标承载的实际 `||u||²` 比例。后续步骤受 Adam 历史状态影响，只报告集中度，不用第一步公式为全部后续现象归因。

若某坐标 g 与全部 moments 为精确零，裸 `jvp(sqrt(g*g))` 会产生 `0*inf`。对可达零状态，复合更新的方向导数仍可有限；实现显式使用这一极限。非零 moment 却有零二阶 moment 等退化状态会报错，不改 epsilon、不裁剪梯度。

**精度与光滑性。** float64 沿相同初值、batch、随机掩码和数学更新规则运行，但其数值轨迹不同于 float32。两者切向量差异是精度敏感性诊断，不能全当作“沿同一个点的 JVP 误差”。AD 对算术操作求导，不对 IEEE 舍入本身求导；ReLU 边界也需要局部验证。高精度 FD 不匹配可能由剂量过大、过小、分支变化或导数实现问题引起，不能自动归因为“切向量错了”。

**B2 必须闭合 Stage 2。** 论文第 9 节末明确要求，两处逆 Hessian 都换为轨迹形式；Adam 对应 Stage 2 式 (42) 与 Stage 1 式 (74)。因此：

\[
\dot p_{sj}=D_\theta p_{sj}\,u_{\theta,T},\qquad
\dot J_Q=\sum_{s,j}v_{sj}^{\mathsf T}\dot p_{sj}
=h_s^{\mathsf T}u_{\theta,T}.
\]

v 必须来自原 attack 训练轨迹的式 (42)。也可将固定 `dot P` 作为方向，前向传播原 attack 的 Adam 状态，取得同一个标量方向导数，避免必须显式存出全部 v。保留 attack 初始化、每个 attack seed、顺序、随机操作、正则项、原轮数与固定 target query；macro 权重只计一次。先验证 Stage 2 单步/短窗口，再推进原完整 horizon。

这估计论文式 (83) 的连续 value pathway，即固定 membership、row identity、类别路由和数据组成的 `J10-J00` 的局部响应。不能直接宣称估计了式 (86) 的 `J11-J00`。只更换 Stage 1 而继续使用隐式 h，仍应标注为混合版本。B1 本次不训练 attack，因此 `ready_for_B2=false`，不输出完整跨阶段 score。

## 代码修订与理由

| 原问题 | 修订 |
|---|---|
| 默认 1 步/1 epoch/5 epochs 没碰到 epoch 25/50，G1 可空跑通过 | 每个 seed 独立重放原配置的全部 epochs；比较所有已存 epoch 的参数、完整 optimizer state、RNG 与最终 baseline。默认真实配置仍由文件决定，不硬编码 50 |
| 开始时强行关闭 TF32 改变原训练环境 | 记录而不改两个 TF32 开关；G1 判定该环境是否精确重现原 baseline |
| 用 output!=0 猜掩码会丢掉零激活上的随机抽样 | 从 dropout 前 RNG 在全 1 张量上重放，恢复真正的所有掩码，同时保留原输出后的 RNG |
| 函数化 Adam 重排运算，容差通过被写成原算法通过 | 尽量对齐原生运算次序；逐项报告 theta/m/v/step 是否逐位相同，近似相同不冒充精确相同；每个扰动也比较 native 与 functional |
| 预测 FD 混用 native 扰动与 functional baseline | 每一 lane 使用自己的未扰动 baseline；先转 double 再相减，预测按原 dtype 分批计算 |
| float64 剂量仍限于 float32 的 `2^-24` | 独立的 float64 阶梯包含 `2^-28,2^-32`；记录完整曲线，不择优隐藏结果 |
| G4 只看参数空间一个最佳剂量，未暴露方向可能误过或报错 | 增加每个病人的首次暴露前缀；零方向单列；相邻两个实测剂量同时通过参数与预测误差/方向阈值才标记 resolved |
| 断言“float32 必须失败”的硬件相关测试 | 改为符号、归一化、耦合 wd、掩码/RNG、非零和零方向、独立 Adam 与 FD 的正确性检查 |
| B1 新文件没有进入 branch_source 哈希保护 | 扩展清单，保留原已有哈希条目；原 baseline 冻结合同不变 |
| 没有 B1 验收工具 | 新 collector 同时检查 sacct、测试成功标记、前后哈希、cell 覆盖和科学结果，生成小型审阅包 |

PyTorch 2.8 的 Adam 在 CUDA 默认可选用 foreach；数值等价不能代替逐位重放。参见 [Adam 官方文档](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html) 和 [数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)。

## 本次并行实验

一次提交：1 个 GPU 回归测试任务，成功后启动 2 个 seed array 任务（42、43，最多同时 2 个 GPU）。每个 seed 包含 807、2085 与监测病人 1369；不单独增加 A 微调。

- G1：完整原 baseline 重放，仅作为真实性验收。
- B1 导数窗口：1 步、1 epoch、5 epochs，加各病人首次暴露的实际 batch；都从原初始化运行，每个窗口配自己的 FD truth。
- float32：`alpha=2^-{10,14,18,22,24}`；同时测原生训练与函数化路径。
- float64 数值控制：`alpha=2^-{10,14,18,22,24,28,32}`；各 alpha 只跑一次最长前缀并沿途取样。
- 参数与预测共同门限：相对误差不大于 0.01、cosine 不小于 0.9999，至少两个相邻实测剂量；不改变历史 B0 判据。
- UNRESOLVED 是合法科学结果，完整保存后可收包；它不等于“导数不存在”。若 G1/G3、有限性或输入完整性失败，则任务失败。

当前 CPU PyTorch 2.8 共 42 项测试通过（B1 15、collector 1、原 E0/B0/v1.1 26）；CPU 小模型首次更新的高精度检查找到共同参数/预测通过区间。该结果仅是代码验证，不能替代 Delta A40 真实模型验收。旧版 631ac3f 的数值表保留为旧实验记录，不直接改写为新协议的通过证据。

## 安装与提交

本机终端上传下载到的修订 ZIP：

```bash
scp ~/Downloads/oct2_b1_smoke_reviewed_20260918.zip yli103@login.delta.ncsa.illinois.edu:~/
```

Delta 登录节点上安装（原仓库与 B worktree 用实际已有路径）：

```bash
oct2_b1_setup=$(mktemp -d "$HOME/oct2-b1-reviewed.XXXXXX")
unzip -q "$HOME/oct2_b1_smoke_reviewed_20260918.zip" -d "$oct2_b1_setup"
python3 "$oct2_b1_setup/oct2_b1_smoke_reviewed_20260918/install_b1.py" \
  --repo "$HOME/oct2-calibration-v4" \
  --worktree "$HOME/oct2-exp-b" \
  --push
```

安装器接受当前 B 为 `0e1cece`、原 B1 `631ac3f` 或本包最终提交；只快进 B，保留原仓库工作区及 main/A 等引用，不自动提交 GPU。

```bash
cd ~/oct2-exp-b
export OCT_BASELINE_ROOT="$HOME/oct2-calibration-v4"
export OCT_RUNS_ROOT="$HOME/oct2-pathway2-runs/B"
export OCT_DATA_DIR=/u/yli103/oct2/data
bash scripts/cross_stage/submit_b1.sh
```

记下返回的 `GPU_TEST`、`B1_ARRAY` 和 `CODE`，任务运行期间保持 B 分支代码不变。`squeue` 空只代表没有排队/运行任务，最终状态看 `sacct`。

```bash
sacct --array -X -j <GPU_TEST>,<B1_ARRAY> \
  --format=JobID%24,JobName%24,State,ExitCode,Elapsed,NodeList
```

验收/打包（将 `<GPU_TEST>` 换成此次返回的数字）：

```bash
python3 -B scripts/cross_stage/collect_b1.py \
  --submission "$HOME/oct2-pathway2-runs/B/submission_b1_<GPU_TEST>.txt" \
  --output "$HOME/oct2_b1_review_<GPU_TEST>.tar.gz"
sha256sum "$HOME/oct2_b1_review_<GPU_TEST>.tar.gz"
```

本机下载该审阅包后提供给后续分析：

```bash
scp yli103@login.delta.ncsa.illinois.edu:~/oct2_b1_review_<GPU_TEST>.tar.gz ~/Downloads/
```

collector 验收成功不表示 G2/G4 科学假设成立。下一步由真实结果决定：原生/函数化不匹配先修 fidelity；高精度短链 unresolved 先看窗口、剂量、非光滑边界与实现；短链验证后才延长 Stage 1，随后闭合 Stage 2 式 (42)。不据本轮结果自动切换到微调 baseline。
