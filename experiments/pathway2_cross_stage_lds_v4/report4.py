"""REPORT_CN.md for the cross-stage ladder, written from summary.json only."""


def f(x, nd=3):
    return '—' if x is None else (f'{x:+.{nd}f}' if isinstance(x, float) else str(x))


def p(x):
    return '—' if x is None else f'{x:.3f}'


def ci(b):
    c = (b or {}).get('bootstrap_ci_conditional_on_seed_panel')
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def render(s, c, figs, partial):
    d, fr = s['design'], s['frozen']
    L = ['# 跨阶段 LDS 梯子：按图像随机删 50%（v4）', '']
    if partial:
        L += [f"> **部分结果**：{len(s['missing'])} 项未完成，只用全部子集都完成的训练 seed。", '']
    L += [f"{d['m']} 个子集（每个保留 {d['retained_images']:,} 张 shadow3 训练图像），训练 seed {d['seeds_complete']}，"
          f"attack seed {d['attack_seeds']}，score seed {d['score_seeds']}。真值 Δvalue = J10 − J00（成员标签不变）；"
          f"J 是合格类别 target query 按图像等权的攻击 CE。统计单位是子集，单侧 5% 门槛 {s['critical_spearman']:.2f}。", '',
          f"冻结（smoke，只用全数据模型）：合格类别 {fr['qualified_classes']}，权重 {fr['omega']}，attack 阻尼 γ_A* = {fr['attack_gamma_star']:g}。", '']
    L += ['## Attack 质量（全数据模型，smoke）', '', '| 类别 | 合格 | ω | AUC 均值 | AUC 最小 | bal acc 最小 | J00 | λmin(H_A) 范围 | ‖∇L_A‖ 最大 |',
          '|---|---|---|---|---|---|---|---|---|']
    for cls, a in s['attack_quality'].items():
        L.append(f"| {cls} | {a['qualified']} | {a['omega']:.3f} | {a['auc_mean']:.3f} | {a['auc_min']:.3f} | "
                 f"{a['balanced_accuracy_min']:.3f} | {a['J00_mean']:.4f} | [{a['hessian_eig_min_range'][0]:.2e}, "
                 f"{a['hessian_eig_min_range'][1]:.2e}] | {a['train_grad_norm_max']:.2e} |")
    t = s.get('truth') or {}
    if t:
        L += ['', '## 真值', '', f"子集间 Δ 均值 {t['mean']:+.5f}，标准差 {t['sd_across_subsets']:.5f}，"
              f"{t['positive_subsets']}/{d['m']} 个子集为正（攻击 CE 上升）。", '']
        rows = [r for r in t['seed_reliability']['rows'] if r.get('split_spearman_mean') is not None]
        if rows:
            L.append('训练 seed 不重叠两组的子集排序可重复性：' + '；'.join(f"k={r['k']}：{r['split_spearman_mean']:.3f}" for r in rows) + '。')
        if t.get('attack_seed_pairwise'):
            L.append('attack seed 两两（对全部训练 seed 平均后）：' + '；'.join(f"{r['a']}–{r['b']}：{f(r['spearman'])}" for r in t['attack_seed_pairwise']) + '。')
    L += ['', '## 梯子（左边用的真值越多）', '', '| 层 | LDS | 95% CI | 置换 p | 斜率 | 相对误差 |', '|---|---|---|---|---|---|']
    for name, b in s['ladder'].items():
        L.append(f"| {name} | {f(b.get('spearman'))} | {ci(b)} | {p(b.get('permutation_p_one_sided'))} | "
                 f"{f(b.get('slope_truth_on_prediction'), 2)} | {f(b.get('relative_error'), 3)} |")
    if s.get('differences'):
        L += ['', '相邻层的成对子集 bootstrap 差值：', '']
        for k, v in s['differences'].items():
            L.append(f"- {k}：{v['difference']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]，不优于的比例 {v['fraction_not_better']:.3f}")
    L += ['', '## L3 的 Stage-1 阻尼 γ（事先固定，不按 LDS 选）', '', '| γ | 状态 | LDS |', '|---|---|---|']
    for g, e in s['l3_gamma'].items():
        L.append(f"| {g} | {str(e['status'])[:70]} | {f((s['ladder'].get(f'L3 (gamma={g})') or {}).get('spearman'))} |")
    L += ['', f"主报告的 γ = 全部 score seed 都合格的最小 γ：{s.get('ladder_gamma_star')}。", '']
    L += ['## 逐类别（不加权，只含用真值的层）', '', '| 类别 | 合格 | L1 | L2-retrain | L2-lin |', '|---|---|---|---|---|']
    for cls, e in s['ladder_per_class'].items():
        r = e['rungs']
        L.append(f"| {cls} | {e['qualified']} | {f(r['L1'].get('spearman'))} | {f(r['L2-retrain'].get('spearman'))} | {f(r['L2-lin'].get('spearman'))} |")
    dr = s.get('drift') or {}
    if dr:
        L += ['', '## 接口漂移（shadow3 行，JS 散度均值）', '', '| 行 | 真实 | 线性化 | 真实 vs 线性化 |', '|---|---|---|---|']
        for g in ('deleted', 'retained', 'nonmember'):
            L.append(f"| {g} | {dr[g]['js_true']:.4f} | {dr[g]['js_linearized']:.4f} | {dr[g]['js_true_vs_linearized']:.4f} |")
        L += ['', f"JVP 跑出单纯形的行比例 {dr['jvp_row_clip_rate_mean']:.3f}；‖Δθ‖ 均值 {dr['dtheta_norm_mean']:.2f}。"]
    L += ['', '## 图', ''] + [f'- `{x}`' for x in figs]
    L += ['', '## 范围', '',
          '- 静态阻尼 IF，在 raw 全数据端点计算；Stage 1 与 attack 都不是驻点。',
          '- L1、L2 用了重训后的真值，是定位诊断，不是可部署的归因。',
          '- 区间条件于当前训练 seed 与 attack seed 面板；p 值未做多重比较校正。', '']
    return '\n'.join(L)
