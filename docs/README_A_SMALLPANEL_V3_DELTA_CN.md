# Route A B/C 小面板 v3 提交说明

本分支基于 `2e0b35d660fbf38cad9da3fea5a6a9ebd6aebade`，独立分支 `exp/pathway2-a-smallpanel`。所有修改是新增 B/C 文件，不替换既有训练、Route A、LR probe 或 frozen baseline 文件。

## 实验与验证范围

B 为 807（DME）、2085（DRUSEN）× seed 42–46 × full/dose01 × control/decay，共 40 cell。每个 cell 从头重建其自身 fixed-mask 轨迹，E100 必须逐位通过模型参数、Adam 状态指纹、Torch CPU/CUDA RNG 指纹、批次顺序核对，然后继续至 E140。control 的学习率恒为 1e-3；decay 在 E101–120 为 1e-4，在 E121–140 为 1e-5。复用 LR probe 的十个 E140 baseline。保存完整优化器/RNG 状态及 ΔP。不会训练 attack 或重算 L3。

C 只读取五个 decay E140 baseline，做 MC=128 的 dropout 梯度均值复核。报告向量差范数、余弦、均值向量 RMS 标准误。diff/error 是描述性比值，不是显著性检验。

本地 CPU 验证使用 PyTorch 2.8.0+cpu。19 项测试通过，包含 synthetic 2→4 epoch 轨迹、删除权重梯度 autograd 对照、完整 40-cell 报告、缺失/重复/篡改检测、MC 计算和手动提交门。不能据此声称真实 OCT E100/E140 已通过。

## 安装

下载修复包、上传 Delta、解压后执行包内安装器。安装器只创建全新的 `~/oct2-exp-a-smallpanel`，拒绝覆盖已有目录。已有 `~/oct2-exp-a` 和 `~/oct2-exp-a-lrprobe` 不会被修改。

```bash
bash ~/oct2_A_partBC_reviewed/install_A_partBC.sh ~/oct2-exp-a-smallpanel
```

安装包包含完整代码历史，不依赖对旧目录进行 fetch 或 checkout。独立仓库的 GitHub remote 设为原仓库 shan927150/oct2。要同步已审核的代码：

```bash
cd ~/oct2-exp-a-smallpanel
git push -u origin exp/pathway2-a-smallpanel
```

不 merge 到其他实验分支，不 force push。原 Route A damping 作业仍在原目录执行，若已提交不重复提交。

## 一次性 prepare

以下块只运行一次。它保存环境供后续终端使用。审批记录写实际用户授权，不虚构导师批准。

```bash
(
set -euo pipefail
cd ~/oct2-exp-a-smallpanel
module load pytorch-conda/2.8
export OCT_BASELINE_ROOT=/u/yli103/oct2-calibration-v4
export OCT_RUNS_ROOT=/u/yli103/oct2-pathway2-runs/A
export OCT_DATA_DIR=/u/yli103/oct2/data
export OCT_A_E100_ROOT=/u/yli103/oct2-pathway2-runs/A/l3_at_E100_a9e26650_20260921T144950Z
export OCT_A_PROBE_ROOT=/u/yli103/oct2-pathway2-runs/A/lr_probe_R100_X40_20260923T200120475977Z
export OCT_ACCOUNT=bgjy-delta-gpu
ENV_FILE=$HOME/routeA_smallpanel_v3.env
test ! -e "$ENV_FILE" || { echo "已有环境文件，请 source 后查看 status，不要重复 prepare"; exit 1; }
OCT_A_SP_OUT=$(bash scripts/cross_stage/submit_A_smallpanel.sh prepare \
  "User authorized reviewed B/C small-panel continuation and endpoint MC on 2026-09-24")
printf 'export OCT_A_SP_OUT=%q\n' "$OCT_A_SP_OUT" > "$ENV_FILE"
printf 'Prepared OUT=%s\n' "$OCT_A_SP_OUT"
)
```

prepare 只验证和冻结输入，不提交作业。

## 先提交测试

```bash
cd ~/oct2-exp-a-smallpanel
module load pytorch-conda/2.8
source ~/routeA_smallpanel_v3.env
bash scripts/cross_stage/submit_A_smallpanel.sh test "$OCT_A_SP_OUT"
bash scripts/cross_stage/submit_A_smallpanel.sh status "$OCT_A_SP_OUT"
```

测试作业先运行 19 项代码测试，再在真实 OCT 数据上重放 seed42、两位患者 × 两种 dose 的四条 E100 轨迹。作业必须 `COMPLETED 0:0`，日志末尾有 `A_SMALLPANEL_GPU_AND_REAL_E100_TESTS_PASSED`，且匹配的 `gpu_tests_passed.json` 存在。GPU test 不自动提交后续 array。

## 测试通过后单独提交 B 与 C

```bash
cd ~/oct2-exp-a-smallpanel
module load pytorch-conda/2.8
source ~/routeA_smallpanel_v3.env
bash scripts/cross_stage/submit_A_smallpanel.sh run "$OCT_A_SP_OUT" --max-parallel 2
bash scripts/cross_stage/submit_A_smallpanel.sh mc "$OCT_A_SP_OUT"
bash scripts/cross_stage/submit_A_smallpanel.sh status "$OCT_A_SP_OUT"
```

B 最多同时使用两张 GPU，C 使用一张 GPU；需与其他运行中的 A 作业共同考虑配额。B 每个 cell 是重放 100 epoch 再续训 40 epoch，不能把它描述为仅训练 40 epoch。时间取决于实际队列与硬件。B 的资源上限为每 task 2.5 小时；测试和 C 各 1 小时。

提交后保持这个仓库代码和冻结输入不变。脚本拒绝同一 attempt 重复提交，且排队结束开始执行时重新核对输入。失败后先查看日志和 sacct；不要删除锁或手动伪造测试通过文件。

## 报告与下载

B 完成 40/40 后，CPU 即可汇总：

```bash
cd ~/oct2-exp-a-smallpanel
module load pytorch-conda/2.8
source ~/routeA_smallpanel_v3.env
bash scripts/cross_stage/submit_A_smallpanel.sh report "$OCT_A_SP_OUT"
```

主要文件：

- `report/REPORT.md`、`summary.json`、`geometry_rows.csv`。
- `report/paired_differences.csv`，逐 patient/seed 比较 control 与 decay。
- `report/gradient_rows.csv`，baseline/full/dose01 各自的 eval-mode 梯度。删除条件使用删除权重，dropout 关闭。
- `report/interface_rows.csv`，固定 affected-shadow train/test 行上的 ΔP 几何及平均 TV。
- `endpoint_mc/endpoint_mc.csv`、`endpoint_mc_summary.json`，C 的五个终点结果。

E100 的参照值在相同两人面板上重算，不强行要求它等于之前八人面板的平均值 9.163。若 decay 比配对 control 更接近线性和局部性，支持继续研究该方案；没有改善，只能说明这套 40-epoch 配方未恢复这些性质，不能确定轨迹分叉发生在前 100 epoch，也不能排除收敛因素。

B/C 都完成后可在 Delta 打包小结果，不包含模型 checkpoint：

```bash
source ~/routeA_smallpanel_v3.env
python3 - <<'PY'
import os, tarfile
from pathlib import Path
root=Path(os.environ['OCT_A_SP_OUT'])
paths=[root/'smallpanel_config.json',root/'config.sha256',root/'gpu_tests_passed.json']
paths += sorted((root/'report').glob('*'))
paths += sorted((root/'endpoint_mc').glob('*'))
for d in sorted(root.glob('seed*')):
    paths += [d/n for n in ('manifest.json','curve.csv','replay_checks.json')]
if not (root/'report/summary.json').is_file() or not (root/'endpoint_mc/endpoint_mc_summary.json').is_file():
    raise SystemExit('B report or C summary is missing')
out=Path.home()/'OCT_A_BC_v3_results.tar.gz'
with tarfile.open(out,'w:gz') as t:
    for p in paths:
        if p.is_file(): t.add(p,arcname=str(Path(root.name)/p.relative_to(root)))
print(out)
PY
```

然后 Mac 下载：

```bash
scp yli103@login.delta.ncsa.illinois.edu:/u/yli103/OCT_A_BC_v3_results.tar.gz ~/Downloads/
```
