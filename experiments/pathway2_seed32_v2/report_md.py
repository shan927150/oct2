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
        L += [f"> **部分结果**：{s['missing_cells']} 个训练单元尚未完成，下表只用每个设计里全部子集都已完成的 seed。", '']
    L += ['主终点 raw；EMA 只作稳健性对照（同一轨迹，不是独立重复）。统计单位是子集，不是子集×seed。',
          '所有预测都写成**删除者之和** $\\sum_{p\\notin S_j}(-\\tau_p)$，与真实变化 $\\Delta Y_j=Y(S_j)-Y(\\text{全数据})$（seed 平均）比较。', '']
    L += ['## 设计', '', '| 设计 | 子集数 | 完成 seed 数 | 图像数范围 | 拒绝抽样接受率 | 单侧 5% 临界 Spearman |', '|---|---|---|---|---|---|']
    for n, d in s['designs'].items():
        r = d['n_images_range']
        L.append(f"| {n} | {d['m']} | {len(d['seeds_complete'])} | {r[0]:.0f}–{r[1]:.0f} | {f(d['acceptance_rate'], 4)} | "
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
    L += ['', '读法：SE 必然随 k 下降；若 IF 有真实信号，LDS 应随 k 上升并趋近 Spearman–Brown 预期；'
          '若 LDS 不随 k 变化，弱结果不是 seed 噪声造成的。LDS 置信区间宽主要来自子集数少，加 seed 解决不了。', '']
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
    # Task 4
    L += ['', '## 任务 4：γ 扫描（γ 事先固定，不按 LDS 选）', '']
    designs = list(s['designs'])
    L += ['| 目标 | γ | 状态 | Spearman(IF, 梯度内积) | ' + ' | '.join(f'LDS {d}' for d in designs) + ' |',
          '|---|---|---|---|' + '---|' * len(designs)]
    for t, rows in s['task4'].items():
        for g, r in rows.items():
            L.append(f"| {t} | {g} | {r['status'][:60]} | {f(r['mean_spearman_if_vs_proxy'])} | " +
                     ' | '.join(f(r['lds'].get(d, {}).get('spearman')) for d in designs) + ' |')
    L += ['', '若 γ 变小时 IF 与梯度内积的相关明显下降，说明 Hessian 开始起作用；LDS 是否随之变化单独看。', '']
    # Task 5
    L += ['## 任务 5：逐查询 LDS（梯度内积预览，163 名查询病人）', '',
          '| 设计 | 目标 | 平均 | 中位数 | 正值比例 | 置换 p | 图像数基线平均 |', '|---|---|---|---|---|---|---|']
    for n, eps in s['task5'].items():
        for t, r in eps.get('raw', {}).items():
            p = r['proxy']
            L.append(f"| {n} | {t} | {f(p['mean'])} | {f(p['median'])} | {f(p['fraction_positive'], 2)} | "
                     f"{pf(p['permutation_p_one_sided'])} | {f(r['size_baseline']['mean'])} |")
    L += ['', '## 图', ''] + [f'- `{x}`' for x in figs]
    L += ['', '## 范围与注意', '',
          '- IF 在每个 score seed 的 raw 全数据端点上计算，端点不是驻点；这是阻尼后的端点近似，不是 Eq.56 的证明。',
          '- 任务 1 的 IF 曲线固定使用 score seed 的平均 τ，只让真值的 seed 数变化；proxy 曲线用与真值相同的 k 个 seed。',
          '- composition ridge 是交叉拟合的诊断，子集数少，不是上界。',
          '- 逐查询 LDS 只用梯度内积，不含逐查询 Hessian 逆。', '']
    return '\n'.join(L)
