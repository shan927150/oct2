# 学长稳定训练配方的隔离移植 v1

从 ca332634a5f50860cea1eb9c1839755232b9434b 创建，仅新增本目录。
参考 2026-09-28 上传 oct2.tar.gz 中 src/models.py、src/config.py 与 05 pilot。
保留当前确定性池化、严格 deterministic 检查、fixed-mask、原 batch 位置与分母。
没有直接覆盖上传包里的旧版 src 或 scripts。

从相同 seed 的初始化开始训练 shadow3，split/N=2045/接口4343行不变。
Adam lr=.001、wd=1e-5、batch128。dropout p=0，前10% step线性 warmup，
之后 cosine，EMA decay=.99，从初始化开始每 step 更新。EMA不反馈原模型优化。
调度按上传代码的 step/total 约定，最后一个实际 step 的 lr 接近0但不恰好等于0。
raw 与 EMA 同时保存。140 epochs 是本次冻结的完整预算，从头训练，不从 E140 继续。
E50/E100 checkpoint 使用140-epoch调度的中间状态，不能与50-epoch调度混淆。

## 阶段一 baseline

3 seeds=42/43/44，各1条训练，加seed42完整no-op，共4条。
raw与EMA来自同一条训练，不算两次独立重复。smoke额外执行2轮小训练精确重放。
完整no-op在 collect 比较 raw/EMA/Adam/RNG/接口，未通过不能 prepare paired。

每epoch记录在线weighted/unweighted CE、raw eval CE、weighted objective、
完整eval梯度和相对参数更新。EMA每10轮/末轮同样观测。终点两种权重都做FP64 g/b审计。
记录heldout CE/accuracy；跨seed meanTV反映输出敏感性，不是删除效应的严格零分布。

```bash
CODE=/u/yli103/oct2-exp-a-stable-training-v1/experiments/pathway2_stable_v1
OUT=/u/yli103/oct2-pathway2-runs/A/stable_training_v1_baseline_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --profile baseline --out "$OUT"
python3 "$CODE/run.py" submit --out "$OUT"
module load pytorch-conda/2.8
python3 "$CODE/run.py" collect --out "$OUT"
```

## 阶段二 paired（单独提交）

必须先完成阶段一 collect。2患者807/2085 × 3seeds × 2剂量(.1/1)=12条。
从头训练，患者降权从epoch1生效，原batch和原分母保持。复用同配方同seed baseline。
保存raw/EMA接口、参数位移和两个真实剂量的R/cosine/相对线性误差。

```bash
BASELINE_OUT=上面实际baseline输出路径
PAIR_OUT=/u/yli103/oct2-pathway2-runs/A/stable_training_v1_paired_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --profile paired --baseline-out "$BASELINE_OUT" --out "$PAIR_OUT"
python3 "$CODE/run.py" submit --out "$PAIR_OUT"
```

阶段一/二不会自动串联。各数组并发1GPU。`submit --after JOBID` 让本实验smoke
等待另一作业结束（afterany）；自己的正式数组仍必须等本smoke成功（afterok）。
同OUT禁止重复提交；不在正在运行的分支上改代码，修改应另建新attempt。

本包完成Stage1稳定性和剂量几何诊断。没有重训Stage2、没有query Delta J、
没有LDS与新L3，不能由本结果声称跨阶段预测得到改善。对照旧结果只能讨论联合配方变化，
不单独归因于dropout、warmup或EMA；raw/EMA对照仅分离同轨迹的平均输出效果。
开启新配方后target与其他shadow保持冻结，本次不是全pipeline新基线。

后续做跨阶段评价时继续保留原接口行和value-pathway标签，攻击端也需配对seed。
正式LDS需要定义患者级子集分布、子集重复训练和下游J，另行设计。

测试：`python3 "$CODE/test_experiment.py" --require-torch`。
