# 同目标 baseline polish v2

本分支现为独立 recovery 分支，父提交是原 v2 的 `3ea683c`。
恢复 seed44 的入口精度检查见 `RECOVERY_CN.md`。使用 `--retry-from` 时只跑 seed44，
下面的双 seed 首轮说明保留为原实验设计记录。

本分支从 ca332634a5f50860cea1eb9c1839755232b9434b 创建，仅新增本目录。
独立于 stable-training 分支，也不改变原 stationarity 分支。

首轮是 seed42/44、E140 decay 的两个 baseline。读取已完成的同目标审计配置、
checkpoint 哈希和数据指纹，使用真正的 `legacy.load_context` 加载 shadow 3、N=2045。

固定目标为 `F = sum(CE)/2045 + (1e-5/2)||theta||²`。关闭 dropout，分块64张计算
完整梯度后才更新一次。新建 L-BFGS，不继承 Adam 状态；history=20、strong_wolfe，
最多100次外层迭代、500次优化器 closure。监测快照和 FP64 核验另计，不包含在500内。
预算在 closure 入口强制检查；未完成的线搜索恢复到上一个已验证的端点。

前后都在同一 FP64 模型上重算 g 与 b，避免 FP64 g / FP32 b 混用。
rho = ||g||/(alpha||b||)。alpha=.1 时 rho<=.1 是工程诊断目标，
还要求 FP32/FP64 向量差各不超过 FP64 范数的10%，这个精度约定不是误差界或数学保证。
预算耗尽、停滞和数值回退必须如实报告；不把它们标成收敛。
位移大时不再解释成旧 E140 附近的微调。Hessian 条件性和非光滑性仍需另查。

原上传 polish 包的修正包括：审计配置没有 seeds 字段，改从 cells 读取；错误的
common.py / build_model_from_payload / shadow0 接口改成已验收的数据加载；
补全 source/input 哈希、输出树保护、独占提交日志、Slurm 绝对代码路径；
纠正接受端点梯度、最终迭代编号和预算边界；保存新 polished.pt；保留全套旧源码。
原 README 中“9–1600 倍证明远非驻点”的措辞改为残差诊断，确定性 eval 的已测范围
约42–1649倍，不与 MC dropout 比值混合。没有假定导师已批准某个具体 polish 配方。

Delta 命令：

```bash
CODE=/u/yli103/oct2-exp-a-objective-polish-v2/experiments/pathway2_polish_v2
OUT=/u/yli103/oct2-pathway2-runs/A/objective_polish_v2_$(date -u +%Y%m%dT%H%M%S%NZ)
python3 "$CODE/run.py" prepare --out "$OUT"
python3 "$CODE/run.py" submit --out "$OUT"
```

prepare 不提交任务。submit 先 smoke，然后 afterok 的两任务数组，并发1GPU。
每正式 cell 的2小时时限是上限，不是耗时估计。作业失败看 logs，不删除日志再重投。
同一 OUT 禁止重复 submit；新尝试必须使用新 OUT。

```bash
module load pytorch-conda/2.8
python3 "$CODE/run.py" collect --out "$OUT"
```

收集包括 summary、rho、trajectory、配置和日志；checkpoint 与 gradient_vectors.npz
留在 Delta。complete.json 记录它们的哈希。没有删除实验、Hessian、L3 或 attack 重训。
没有改动任何旧 checkpoint；新的 polished.pt 是单独文件，只支持权重加载后新建优化器。

理论依据：用户提供 continue_pathway-2.pdf 的同目标驻点与局部响应框架。
API核对：https://docs.pytorch.org/docs/2.8/generated/torch.optim.LBFGS.html

测试：`python3 "$CODE/test_experiment.py" --require-torch`。
