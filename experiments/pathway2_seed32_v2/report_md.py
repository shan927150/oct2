"""REPORT_CN.md: key numbers per task, written from summary.json only."""


def f(x, nd=3):
    return '—' if x is None else (f'{x:+.{nd}f}' if isinstance(x, float) else str(x))


def pf(x):
    return '—' if x is None else f'{x:.3f}'


def ci(block):
    c = block.get('bootstrap_ci_conditional_on_seed_panel')
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def render(s, c, figs, partial=False):
    L = ['# Stage 1 · 32 seed / 删除者 LDS / γ 扫描 / 逐查询 LDS 报告', '']
    if partial:
        L += [f"> **部分结果**：{s['missing_cells']} 个训练单元、共 {len(s['missing_artifacts'])} 项产物尚未完成，下表只用每个设计里全部子集都已完成的 seed。", '']
    L += ['CE/raw 为原定主分析；1−p_y 和 Brier 是看到 CE 结果后加入的探索目标。EMA 只作稳健性对照（同一轨迹，不是独立重复）。统计单位是子集，不是子集×seed。',
          '所有预测都写成**删除者之和** $\\sum_{p\\notin S_j}(-\\tau_p)$，与真实变化 $\\Delta Y_j=Y(S_j)-Y(\\text{全数据})$（seed 平均）比较。', '']
    L += ['## 设计', '', '| 设计 | 子集数 | 完成 seed 数 | 图像数范围 | 每 epoch 步数 | 拒绝抽样接受率 | 单侧 5% 临界 Spearman |', '|---|---|---|---|---|---|---|']
    for n, d in s['designs'].items():
        r = d['n_images_range']
        steps = d['steps_per_epoch_range']
        L.append(f"| {n} | {d['m']} | {len(d['seeds_complete'])} | {r[0]:.0f}–{r[1]:.0f} | {steps[0]}–{steps[1]} | {f(d['acceptance_rate'], 4)} | "
                 f"{s['critical_spearman'].get(str(d['m'])):.2f} |")
    L += ['']
    # Task 1
    L += ['## 任务 1：seed 数对真值稳定性和 LDS 的影响', '']
    L += ['| 设计 | 目标 | k=1 可靠性（split-half） | k=max 可靠性（方差分量） | IF LDS k=1（均值） | IF LDS k=max | proxy LDS k=max | 图像数基线 |',
          '|---|---|---|---|---|---|---|---|']
    for n, eps in s['task1'].items():
        for t, cur in eps.get('raw', {}).items():
            rel = cur['reliability']['rows']
            lds = cur['lds']
            get = lambda p, k: next((r['mean'] for r in lds if r['predictor'].startswith(p) and r['k'] == k), None)
            kmax = max(r['k'] for r in rel)
            L.append(f"| {n} | {t} | {f(rel[0].get('split_spearman_mean'))} | {f(rel[-1].get('icc_reliability'))} | "
                     f"{f(get('IF', 1))} | {f(get('IF', kmax))} | {f(get('proxy', kmax))} | {f(get('size', kmax))} |")
    L += ['', '读法：固定总样本标准差后，sd/sqrt(k) 按构造下降；它不是实验正常的独立证据。有限 seed 池的重采样标准差在 k=全部时按构造为零。'
          '独立排序复现只在 2k≤已完成 seed 数时可测；Spearman–Brown 秩相关外推和方差分量曲线是启发式估计。'
          'LDS 无须逐点单调；单条平坦曲线不足以排除 seed 噪声。曲线带是 seed 抽样分位数，不能当作总体 LDS 的置信区间。', '']
    # Task 2/3
    L += ['## 任务 2–3：删除者 LDS、子集大小控制与朴素基线', '']
    L += ['raw 端点（EMA 见 `table_lds.csv`）。\n', '| 设计 | 目标 | 预测器 | LDS | 95% CI | 置换 p | 扣除图像数后 | 偏秩相关 |', '|---|---|---|---|---|---|---|---|']
    for n, eps in s['task2_3'].items():
        for ep, by_t in eps.items():
            if ep != 'raw':
                continue
            for t, blk in by_t.items():
                for p, b in blk['predictors'].items():
                    L.append(f"| {n} | {t} | {p} | {f(b['spearman'])} | {ci(b)} | {pf(b.get('permutation_p_one_sided'))} | "
                             f"{f(b['residual_spearman_given_images'])} | {f(b['partial_rank_spearman_given_images'])} |")
    L += ['', '删除者写法等价性（与保留者写法的 LDS 必须一致；漏掉负号会得到相反数）：', '']
    for n, eps in s['task2_3'].items():
        for t, blk in eps.get('raw', {}).items():
            e = blk.get('deleter_equivalence')
            if e:
                L.append(f"- {n}/{t}：删除者 {f(e['deleter_spearman'])}，保留者 {f(e['retained_sum_spearman'])}，"
                         f"恒等式最大偏差 {e['max_abs_difference_deleter_minus_retained_plus_total']:.1e}，漏负号 {f(e['sign_error_spearman_if_minus_dropped'])}")
            else:
                L.append(f"- {n}/{t}：IF 不可用（{blk.get('if_status')}）")
    L += ['', '图像数与真值的 Spearman（只看子集大小能解释多少）：', '']
    for n, eps in s['task2_3'].items():
        L.append('- ' + n + '：' + '，'.join(f"{t} {f(b['truth_n_images_spearman'])}" for t, b in eps.get('raw', {}).items()))
    same = s['same_seed_design_comparison']
    L += ['', '### 同一 seed 面板的设计比较', '', f"共同 seed：{same['seeds']}。以下均为 raw；不能把 pilot8 的 32 seed 与 eq32 的 8 seed 直接解释为设计改进。",
          '', '| 设计 | 目标 | 预测器 | LDS | 偏秩相关（控制图像数） |', '|---|---|---|---|---|']
    for name, targets in same['designs'].items():
        for t, preds in targets.items():
            for p, b in preds.items():
                L.append(f"| {name} | {t} | {p} | {f(b['spearman'])} | {f(b['partial_rank_spearman_given_images'])} |")
    # Task 4
    L += ['', '## 任务 4：γ 扫描（γ 事先固定，不按 LDS 选）', '']
    designs = list(s['designs'])
    L += ['| 目标 | γ | 状态 | Spearman(IF, 梯度内积) | ' + ' | '.join(f'LDS {d}' for d in designs) + ' |',
          '|---|---|---|---|' + '---|' * len(designs)]
    for t, rows in s['task4'].items():
        for g, r in rows.items():
            L.append(f"| {t} | {g} | {r['status'][:60]} | {f(r['mean_spearman_if_vs_proxy'])} | " +
                     ' | '.join(f(r['lds'].get(d, {}).get('spearman')) for d in designs) + ' |')
    L += ['', '仅纳入全部 score seed 都通过门槛的 γ；unqualified 表示当前数值方案未通过门槛，不能据此证明矩阵不可逆。'
          'IF 与梯度内积的相关下降可说明曲率修正改变了排序，不能单独证明预测质量变好。', '']
    # Task 5
    L += ['## 任务 5：逐查询 LDS（梯度内积预览，163 名查询病人）', '',
          '| 设计 | 目标 | 平均 | 95% CI | 中位数 | 正值比例 | 置换 p | 图像数基线平均 |', '|---|---|---|---|---|---|---|---|']
    for n, eps in s['task5'].items():
        for t, r in eps.get('raw', {}).items():
            p = r['proxy']
            L.append(f"| {n} | {t} | {f(p['mean'])} | {ci(p)} | {f(p['median'])} | {f(p['fraction_positive'], 2)} | "
                     f"{pf(p['permutation_p_one_sided'])} | {f(r['size_baseline']['mean'])} |")
    L += ['', '逐查询置信区间和置换检验对所有查询共同重采样子集；163 名查询共享训练子集，不能视为 163 次独立训练。'
          '区间条件于当前 seed 面板与查询集；常量列的 LDS 未定义。', '', '## 图', ''] + [f'- `{x}`' for x in figs]
    L += ['', '## 范围与注意', '',
          '- IF 在每个 score seed 的 raw 全数据端点上计算，端点不是驻点；这是阻尼后的端点近似，不是 Eq.56 的证明。',
          '- 任务 1 的 IF 曲线固定使用 score seed 的平均 τ，只让真值的 seed 数变化；proxy 曲线用与真值相同的 k 个 seed。',
          '- composition ridge 是交叉拟合的诊断，子集数少，不是上界。',
          '- 逐查询 LDS 只用梯度内积，不含逐查询 Hessian 逆。', '']
    return '\n'.join(L)
