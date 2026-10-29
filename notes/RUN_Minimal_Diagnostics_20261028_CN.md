# 最小诊断代码与运行说明

日期：2026-10-28

本轮复用完成的 Interface LDS 和 case study，做 logit 诊断、梯度范数和特征分析。没有新增训练、Hessian 求解、smoke 作业或提交验收门槛。

## 文件和改动

- logit_diag.py：直接读取已有 v3 子集概率，再计算相对 logits。零概率记为缺失，各档 T/L/I 比较使用共同有效向量并报告数量。增加相对误差，保留 Pearson、幅度比和符号一致率。
- self_influence.py：保留文件名以便覆盖，只计算完整模型上每张训练图像的 CE 梯度范数。删除近似 self-influence、Krylov basis 和 check 参数。
- case_features.py：按 train_position、raw_index、cls 对齐后平均梯度范数。默认 seeds 42、43、44。报告全体训练图像的 removal_score，以及攻击有效图像的 a_true、a_if、gap_share，并分别汇总 DME、DRUSEN。
- self_influence.slurm：作业名 oct_grad_norm，一张 A100、64G 内存、1 小时时限。梯度计算后在同一作业中生成特征表和图。
- logit_diag.slurm：CPU 作业，4 CPU、16G 内存、1 小时时限。
- test_t2.py：本地数值测试，不作为 Slurm 提交前置步骤。

## 定义

- 梯度范数是 mean_s ||grad CE_i(theta_s)||，不是 ||mean_s grad CE_i(theta_s)||。每个 seed 先计算范数，再按图像身份平均。
- 相对 logit 为 log(p_k) - log(p_y)，去掉恒等于零的真实类别分量后每张图有三个分量。
- T 为重训真值，L 为真实参数位移的一阶近似，I 为 IF 的一阶近似。范数比衡量幅度，相对误差和 Pearson 衡量一致性。
- 概率原值为零时无法恢复对应的有限 logit，记录为缺失，不用小常数代替。种子均值保留固定面板，不对缺失 seed 单独减少平均数量。
- own_margin.csv 使用完整模型所选的固定竞争类别。这里的 margin 对比仍是半数据子集的被删减被保留对比。
- removal_score 是完整 L3 的逐图像预测评分。a_true/a_if 是自身接口行的攻击加权对比。gap_share 是 row 3 减 row 1 的平均逐图像修正项，不是 LDS 提升比例。
- attack_scale.csv 的 alpha 扫描和 alpha_LS 均为同一批子集上的描述性诊断，不作为新方法泛化结果。
- 图中的 log 坐标只画正值，相关系数使用原始有限数值，标题显示绘制数量。

## Mac 上传 Git

将压缩包放在 ~/Downloads，然后执行：

~~~bash
cd ~/Downloads
unzip -o OCT_diag_minimal_20261028.zip

(
set -e
cd ~/oct2-interface-case-study-submit
git switch exp/pathway2-interface-case-study-v1
git pull --ff-only

pkg="experiments/pathway2_interface_case_study_v1"
src="$HOME/Downloads/OCT_diag_minimal_20261028"
cp "$src/$pkg/"*.py "$pkg/"
cp "$src/$pkg/"*.slurm "$pkg/"
mkdir -p notes
cp "$src/notes/RUN_Minimal_Diagnostics_20261028_CN.md" notes/

git add "$pkg/logit_diag.py" "$pkg/self_influence.py" "$pkg/case_features.py" \
  "$pkg/logit_diag.slurm" "$pkg/self_influence.slurm" "$pkg/test_t2.py" \
  notes/RUN_Minimal_Diagnostics_20261028_CN.md
git diff --cached --stat
git -c user.name="Yushan Li" -c user.email="shane29361@gmail.com" \
  commit -m "Simplify logit diagnostics and case study gradient features"
git push origin exp/pathway2-interface-case-study-v1
)
~~~

## Delta 提交两个独立任务

登录 Delta 后执行。脚本放在诊断 checkout，--repo 继续指向原有冻结 checkout。

~~~bash
(
set -e
cd ~/oct2-interface-case-study
git switch exp/pathway2-interface-case-study-v1
git pull --ff-only

pkg="$HOME/oct2-interface-case-study/experiments/pathway2_interface_case_study_v1"
frozen="/u/yli103/oct2-interface-lds-v1r1"
out="/u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_20261022T230209460267514Z"
mkdir -p "$out/logit_diag" "$out/case_study/features"

logit_jid=$(sbatch --parsable \
  --output="$out/logit_diag/slurm-%j.log" \
  "$pkg/logit_diag.slurm" "$pkg" "$frozen" "$out")
printf '%s\n' "$logit_jid" > "$out/logit_diag/jobid.txt"
echo "LOGIT_JOB_ID=$logit_jid"

grad_jid=$(sbatch --parsable \
  --output="$out/case_study/features/slurm-%j.log" \
  "$pkg/self_influence.slurm" "$pkg" "$frozen" "$out")
printf '%s\n' "$grad_jid" > "$out/case_study/features/jobid.txt"
echo "GRAD_JOB_ID=$grad_jid"

squeue -j "$logit_jid,$grad_jid" -o "%.18i %.20j %.10T %.10M %.30R"
)
~~~

## 查看结果

~~~bash
out="/u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_20261022T230209460267514Z"
logit_jid=$(cat "$out/logit_diag/jobid.txt")
grad_jid=$(cat "$out/case_study/features/jobid.txt")

sacct -S 2026-10-28 -j "$logit_jid,$grad_jid" -X \
  -o JobID,JobName%20,State,ExitCode,Elapsed
tail -n 60 "$out/logit_diag/slurm-$logit_jid.log"
tail -n 60 "$out/case_study/features/slurm-$grad_jid.log"
~~~

输出：

- OUT/logit_diag/：logit_rungs.csv、own_margin.csv、own_margin_summary.csv、attack_scale.csv、fig6_logit.png。
- OUT/case_study/features/：features_seed42.csv、features_seed43.csv、features_seed44.csv。
- OUT/case_study/gamma0.03_k100/：features_summary.csv、features_groups.csv、features_merged.csv、fig5_features.png。

默认分析 k=100。单独分析已有 k=20 的 CSV 时，运行 case_features.py --dir 对应目录，脚本会读取该目录 groups.csv 中的 k。

## 已做的本地检查

8 项数值测试通过，覆盖极小概率恢复、零概率的共同有效集、已知幅度比和相对误差、固定竞争类别对比、跨 seed 加权分数、特征文件换行顺序后的身份对齐和总体/分类人群。使用冻结 analysis_if 的数值函数，原始数据读取由合成数据替代。

三个 Python CLI 的帮助入口和两个 Slurm 文件的 Bash 语法已检查。CPU 两个分析入口已用合成数据生成 CSV 和 PNG。尚未在 Delta 原始数据和 GPU 上运行这版代码。
