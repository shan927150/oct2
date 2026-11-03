# OCT removal 实验运行说明

本包保留 A、B、C 的实验计算、随机对照、结果和校验；不含汇报稿或既有结果表。运行前请使用新目录，避免混用其他版本。

## 实验含义

* A：读取已保存的 Stage-1 预测与重训概率，改变接口行分数的组成，不删除 attack 训练数据。`within_S` 比较同组 IF 加权分数与真实 Δp 计算的一阶加权参考；后者不是这组行真实的两阶段删除效应，权重来自完整 attack 训练表。
* B：读取原 supplement v2 的结果，汇总真正从 attack 训练表删行并重训 attack 的实验。补跑 B 使用原有 supplement v2 包和原 manifest，不使用本包的 worker。
* C：改变 target query 目标，worker 重放原 attack 以获得逐 query 的值；重放不创建新的训练条件。先运行 seed 42 smoke 并核对重放，再运行其余 seed。

A 的组对照按类别和 shadow member/nonmember 数量匹配，在每个子集独立抽样；它不保留原集合跨子集的图像对应关系。C 按类别与 target membership 匹配（单独比较 membership 的规则按类别匹配）。随机范围是描述性参考。Oracle 选取与评估共用子集，C crossfit 只分开 attack seed，均不是独立外部验证。

## 1. 设置路径与测试

先将 ZIP 上传到 Delta 的 `~/`，然后在 Delta 登录节点执行：

```bash
pkg="$HOME/oct_removal_abc_run_v4"
frozen="$HOME/oct2-interface-lds-v1r1"
runs="/u/yli103/oct2-pathway2-runs/A"
out="$runs/interface_lds_v1r1_20261022T230209460267514Z"

mkdir -p "$pkg"
unzip -n "$HOME/OCT_removal_ABC_run_v4.zip" -d "$pkg"
module load pytorch-conda/2.8
python3 "$pkg/test_removal_abc.py" --attack4-dir "$frozen/experiments/pathway2_cross_stage_lds_v4"
```

必须确认完整测试通过，且没有 skipped 测试。完整测试包含 PyTorch 逐 query adjoint 和小型假数据重放；测试通过不替代 C 的真实数据 smoke。

## 2. 方法 A：CPU 分数实验

```bash
adest="$runs/removal_A_run_v4"
mkdir -p "$adest"
ajob=$(sbatch --parsable --output="$adest/%j.log" \
  "$pkg/cpu_job.slurm" "$pkg" method_a_score_drop.py \
  --repo "$frozen" --out "$out" --dest "$adest")
echo "$ajob" | tee "$adest/jobid.txt"
```

用 `sacct -j "$ajob" -X -o JobID,State,ExitCode,Elapsed` 检查完成状态。输出为 `a_results.csv`、`a_subset_values.csv`、`a_checks.json`；`within_S`、keep、drop 和两个方向的替换均保留。基线重建不通过会报错。

## 3. 方法 B：补算与结果汇总

已完成的 class-member task 0–35 不必重跑。其他 task 是否完成以原目录中的实际输出和作业状态为准。原 manifest 每组按 seed 42、43、44 映射到连续的 3 个 task。可用下面命令只列出 n=50 尚缺少的 task；它不会提交任务：

```bash
v2pkg="$HOME/oct_prune_supplement_v2_20261030"
v2dest="$runs/attack_prune_supplement_v2_20261030"
export v2dest
python3 - <<'PY'
import json, os
from pathlib import Path
d = Path(os.environ['v2dest'])
m = json.loads((d / 'manifest.json').read_text())
for task in range(36, min(78, len(m['groups']) * 3)):
    group, seed = m['groups'][task // 3]['name'], (42, 43, 44)[task % 3]
    if not (d / 'results' / group / f'seed{seed}.npz').is_file():
        print(task, group, seed)
PY
squeue -u "$USER" -n oct_prune_supp -o '%.18i %.24j %.10T %.10M'
```

确认缺失 task 没有在排队或运行后，可将具体 task ID 放入 Slurm array；不要重交已有任务。n=200 对应 78–119，检查方法相同。原 worker 遇到已存在的输出会拒绝覆盖。

已完成的 B 可随时汇总（不完整的组会列入缺失清单）：

```bash
bdest="$runs/removal_B_collect_v4"
mkdir -p "$bdest"
bjob=$(sbatch --parsable --output="$bdest/%j.log" \
  "$pkg/cpu_job.slurm" "$pkg" method_b_collect.py \
  --v2-dest "$v2dest" --out-dir "$bdest")
echo "$bjob" | tee "$bdest/jobid.txt"
```

## 4. 方法 C：真实数据 smoke

```bash
cdest="$runs/removal_C_query_run_v4"
mkdir -p "$cdest/logs"
csmoke=$(sbatch --parsable --array=0 \
  --output="$cdest/logs/%A_%a.out" --error="$cdest/logs/%A_%a.err" \
  "$pkg/method_c_query.slurm" "$pkg" "$frozen" "$out" "$cdest")
echo "$csmoke" | tee "$cdest/smoke_jobid.txt"
```

完成后检查 `sacct -j "$csmoke" -X -o JobID,State,ExitCode,Elapsed`、stdout、stderr 和 `seed42_checks.json`。所有重放类别的 J00/J10 检查均通过、qualified 类别分数检查通过后，才提交 `--array=1-2`；三个 seed 完成后运行 `method_c_query.py analyze --dest "$cdest"`。每个 worker 的实际用时以 smoke 为准。
