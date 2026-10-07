# seed32 v2 修订记录与验证

日期为 2026-10-06（America/Chicago）。基于用户 bundle 的 `5e145ef44eb03b4880386c7526e8598cc54c4e4b`，不是此前另写的 next5 目录。

## 改了什么

| 问题 | 修订 |
|---|---|
| 一个奇异 γ 会使整个 multi-shift 扫描退出 | 每个 shift 单独记录失败，保留其他结果；失败的 τ 不输出。 |
| CE 复现失败发生在写完成标记之后 | 一致性检查先于完成标记；保留临时目录中的失败诊断，补跑不会误跳过。 |
| 极小 p_y 经 `-w/p_y` 计算 CE 梯度可能溢出 | 直接对 log-softmax CE 求导；概率接口继续保留，新增 log 概率用于重算 CE。 |
| 等图像数容差仍可跨 batch 边界 | 以预先定义的目标图像数确定步数，拒绝跨步数的 masks；报告实际步数。 |
| pilot32-seed 和 eq8-seed 比较混入 seed 数差异 | 另出相同 seed 身份的设计比较表，保留各设计原有全部结果。 |
| 逐 query 只有均值和 permutation | 增加共同重采样子集的 bootstrap CI，处理常量 query；重复 query 不会缩窄 CI。 |
| 不完整 proxy/score 可能被收集为完整报告 | 缺失产物单列，严格 collect 要求齐全；允许显式 partial collect。 |
| 仅凭完成标记存在来补跑 | 校验结果身份与哈希，检查 Slurm 作业终态，保留不确定的提交 receipt。 |
| 缺少低成本 γ pilot 入口 | 新增独立 score-pilot，1−p_y × 3 seeds × {1,0.3,0.1}，不与正式分数混用。 |
| 若干统计表述过强 | 更正 SE、独立可靠性、不可逆性和逐 query 独立性的解释；CE 标为主目标。 |

原 `stable_core.py`、LDS v1 `design.py` / `analysis_stats.py`、`src/`、`scripts/cross_stage/` 都保持原字节。修订目录中保留已有接口，不增加依赖或通用框架。按用户要求，去掉了重复的整任务锁，只保留提交和单元写入的必要互斥。

## 已执行验证

| 检查 | 结果 |
|---|---|
| 原有公式、设计、调度与完整合成 collect 测试 | 17 项通过。包含删除者符号、matched 归一化、dense solve/CG 对照、报告图表和压缩包。 |
| 新增回归测试 | 18 项通过。包含奇异 shift 隔离、联合 query CI、S=1、负可靠性外推、manifest、重复提交、未知 receipt、γ pilot 与 GPU 工时统计。 |
| 真实 CNN 代码路径的 CPU 集成 | 1 项综合测试通过。raw/EMA/Adam 精确 replay、harvest、新 seed 训练、proxy 重组、γ 求解、pilot 分流、CE 失败不标记完成、未合格不输出 τ、极端 logits 下的 CE 均已覆盖。 |
| 修改范围与补丁格式 | `git diff --check` 通过；仅实验目录有改动。 |

共 36 项 unittest 测试。CPU CNN 用实际 503,044 参数架构、合成 16×16 输入、2 epochs。其求解对照特意用较强 L2 构造合格数值案例；不能据此推断真实 OCT 的 γ 是否合格。该案例的 γ=1/0.3/0.03 真实残差分别约 7.4e-11、3.1e-8、9.0e-7。

上述 CNN 测试执行时使用 Python 3.12、PyTorch 2.14.1+cpu。实际 Delta 仍需在 PyTorch 2.8 环境通过 140 epoch replay 后才执行正式训练。

## 当前交付与下一步

本版没有提交 Delta 作业。此前连接 `login.delta.ncsa.illinois.edu` 的 DNS 解析失败，无法在这里执行真实 GPU 验收。

GitHub 发布未完成。此前 push 被自动审批拒绝，理由是尚未明确授权把含内部 Delta 路径和运行细节的代码发布到该远程。没有重试 push 或绕过限制。本次以可审阅的 bundle 和增量 patch 交付，Delta 安装不依赖远程 reviewed 分支。

下一步按 README 安装新 worktree，先 prepare 和 GPU smoke，再运行原 8 masks 的 seed 扩展与 eq32；γ 先行 pilot 核对数值门槛和成本，再按冻结面板完成正式扫描。新 OUT 与原版 OUT 分开，避免新 mask/输出契约与旧产物混用。
