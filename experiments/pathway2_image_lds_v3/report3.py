"""REPORT_CN.md for the image-level LDS, written from summary.json only."""


def f(x, nd=3):
    return '—' if x is None else (f'{x:+.{nd}f}' if isinstance(x, float) else str(x))


def p(x):
    return '—' if x is None else f'{x:.3f}'


def render(s, c, figs, partial):
    d = s['design']
    L = ['# 按图像随机删除 50% 的标准 LDS（Stage 1）', '']
    if partial:
        L += [f"> **部分结果**：{len(s['missing'])} 项产物未完成，只用全部子集都已完成的 seed。", '']
    L += [f"设计 {d['name']}：{d['m']} 个子集，每个保留 {d['retained_images']:,} / {d['n_train']:,} 张训练图像（均匀随机，不按病人）；"
          f"完成 seed {d['seeds_complete']}。子集图像数完全相同，−Nⱼ 无法排序。统计单位是子集；单侧 5% 门槛 "
          f"{s['critical_spearman']:.2f}。主目标：{d['n_query']:,} 张查询图像等权的 CE，raw 端点。", '']
    gates = s.get('patient_level_gates') or {}
    if gates:
        L += ['## 实现门槛（图像分数按病人加总后复现 LDS v1）', '', '| seed | 通过 | 相对 L2 | Spearman | 真实残差 |', '|---|---|---|---|---|']
        for k, g in gates.items():
            L.append(f"| {k} | {g['passed']} | {g['rel_l2_difference']:.2e} | {f(g['spearman'])} | {g['true_rel_residual']:.1e} |")
        L.append('')
    L += ['## 宏目标 LDS（删除者写法，raw；EMA 见 table_lds.csv）', '', '| 端点 | 目标 | 权重 | 预测器 | LDS | 95% CI | 置换 p |', '|---|---|---|---|---|---|---|']
    for ep, by in s.get('macro', {}).items():
        if ep != 'raw':
            continue
        for key, blk in by.items():
            t, w = key.rsplit('_', 1)
            for name, b in blk['predictors'].items():
                ci = b.get('bootstrap_ci_conditional_on_seed_panel')
                L.append(f"| {ep} | {t} | {w} | {name} | {f(b['spearman'])} | "
                         f"{'—' if not ci else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'} | {p(b.get('permutation_p_one_sided'))} |")
    eq = s.get('equivalence') or {}
    if eq:
        L += ['', '删除者 / 保留者 / matched 写法（Nⱼ 恒定，三者排序必然相同；漏负号得到相反数）：', '']
        for t, e in eq.items():
            L.append(f"- {t}：{f(e['deleter'])} / {f(e['retained_sum'])} / {f(e['matched'])}；漏负号 {f(e['sign_flip'])}")
    L += ['', '## 真值稳定性与 LDS 随 seed 数（image-mean）', '', '| 目标 | k=1 可重复性 | 最大可测 k | 该 k 可重复性 | proxy LDS k=1 → max |', '|---|---|---|---|---|']
    for key, cur in s.get('seed_curves', {}).items():
        if not key.endswith('_img'):
            continue
        rows = [r for r in cur['reliability']['rows'] if r.get('split_spearman_mean') is not None]
        lds = [r for r in cur['lds'] if r['predictor'].startswith('proxy')]
        L.append(f"| {key[:-4]} | {f(rows[0]['split_spearman_mean']) if rows else '—'} | {rows[-1]['k'] if rows else '—'} | "
                 f"{f(rows[-1]['split_spearman_mean']) if rows else '—'} | "
                 f"{(f(lds[0]['mean']) + ' → ' + f(lds[-1]['mean'])) if lds else '—'} |")
    L += ['', '## 逐查询 LDS（proxy；先逐个算再平均）', '', '| 目标 | 单位 | 平均 | 95% CI | 中位数 | 正值比例 | 置换 p | 有定义数 |', '|---|---|---|---|---|---|---|---|']
    for key, r in s.get('per_query', {}).items():
        t, unit = key.rsplit('_', 1)
        q = r['proxy']
        ci = q.get('bootstrap_ci_conditional_on_seed_panel')
        L.append(f"| {t} | {unit} | {f(q['mean'])} | {'—' if not ci else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'} | {f(q['median'])} | "
                 f"{f(q['fraction_positive'], 2)} | {p(q['permutation_p_one_sided'])} | {q['defined']}/{q['n_queries']} |")
    L += ['', '## γ 扫描（γ 事先固定，不按 LDS 选）', '', '| 目标 | γ | 状态 | Spearman(IF, proxy) | LDS |', '|---|---|---|---|---|']
    for t, rows in s.get('gamma', {}).items():
        for g, e in rows.items():
            L.append(f"| {t} | {g} | {str(e['status'])[:70]} | {f(e['mean_spearman_if_vs_proxy'])} | "
                     f"{f((e.get('lds_img_raw') or {}).get('spearman'))} |")
    L += ['', '## 图', ''] + [f'- `{x}`' for x in figs]
    L += ['', '## 范围', '',
          '- IF 在各 score seed 的 raw 全数据端点计算，端点不是驻点；这是阻尼端点近似。',
          '- 逐查询 LDS 只用梯度内积，没有逐查询 Hessian 逆。',
          '- 类别构成 ridge 是交叉拟合诊断，不给 CI/p，不是上界。', '']
    return '\n'.join(L)
