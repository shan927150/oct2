"""REPORT_CN.md for v6, from summary.json only."""


def _f(x, nd=3, sign=True):
    if x is None:
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def _ci(c):
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def render(s, c, partial):
    d, g = s['design'], s.get('gate') or {}
    L = ['# 跨阶段 LDS v6：attack seed 3 → 6，member-only h', '']
    if partial:
        L += [f"> **部分结果**：{len(s['missing'])} 项未完成。", '']
    L += [f"{d['m']} 个子集 × 训练 seed {d['seeds_complete']}；attack seed：v5 r1 的 {d['old']} + 新 {d['new']}。"
          f"Stage-1 模型、稳定 target 的 attack 表、主类别、ω、γ_A* = {s['frozen']['attack_gamma_star']:g} 都继承自 v5 r1。"
          f"统计单位是子集；单侧 5% 门槛 {s['critical_spearman']:.2f}，2.5% 门槛 {s['critical_spearman_alpha_0_025']:.2f}。", '']
    L += ['## 门与复现', '',
          f"- 新 seed 的 CE 质量门：{'通过' if g.get('passed') else '未通过'}"
          + (' ' + str({k: round(v['J00_max'], 4) for k, v in g.get('per_class', {}).items()}) if g else '') + '。',
          f"- 继承的 γ_A* 在新 seed 上合格：{g.get('gamma_star_qualified_on_new_seeds')}。",
          f"- 旧 attack 在本运行中的复现（smoke J00 最大差 {g.get('old_attack_replay_max_abs')}；每个 truth 单元的 J10 复现最大差 "
          f"{s['replay']['j10_replay']:.1e}）；复制的 v5 r1 L1 / L2-lin 与本运行数组的最大差 {s['replay']['L1']:.1e} / {s['replay']['L2']:.1e}。"]
    sd = s.get('score_diagnostics') or {}
    if sd:
        cos = [v['cos_hbar_full_member'] for v in sd.values()]
        rep = [v['hbar_replay_vs_v5'] for v in sd.values()]
        L.append(f"- 3-seed hbar 复现 v5 r1：最大差 {max(rep):.1e}；hbar_full 与 hbar_member 的余弦 {min(cos):.3f}–{max(cos):.3f}。")
    t = s['truths']
    if t:
        ar = s['attack_seed_reliability']
        L += ['', '## 真值', '', '| 真值 | 平均 Δ | 子集间 SD |', '|---|---|---|']
        for k, name in (('all', '6 个 attack seed'), ('old', 'v5 r1 的 3 个'), ('new', '新的 3 个')):
            L.append(f"| {name} | {t[k]['mean']:+.4f} | {t[k]['sd_across_subsets']:.4f} |")
        rel = {r['k']: r for r in t['seed_reliability_all']['rows']}
        L += ['', f"旧 3 个 vs 新 3 个 attack seed 的子集排序相关（split-half）：{_f(ar['split_half_old_vs_new'])}；"
                  f"attack seed 两两平均 {ar['pairwise_mean']:+.3f}（v5 r1 三个之间 {ar['pairwise_old_mean']:+.3f}）。"
                  f"训练 seed 可重复性：k=4 {_f(rel.get(4, {}).get('split_spearman_mean'))}，k=8 ICC {_f(rel.get(8, {}).get('icc_reliability'))}。"]
    pr = s.get('primary') or {}
    if pr:
        L += ['', '## 预设主检验（单侧置换 p，Bonferroni α = 0.025）', '', '| 检验 | 层 | LDS | 95% CI | p | 通过 |', '|---|---|---|---|---|---|']
        for key in ('P1', 'P2'):
            e = pr.get(key)
            if e and 'lds' in e:
                L.append(f"| {key} | {e['rung']} | {_f(e['lds'])} | {_ci(e['ci'])} | {_f(e['perm_p_one_sided'], 3, False)} | {e['passed']} |")
            elif e:
                L.append(f"| {key} | {e['rung']} | — | — | — | {e['status']} |")
        e = pr.get('P3')
        if e:
            L += ['', f"P3（描述性）{e['rung']}：{e['difference']:+.3f} {_ci(e['ci'])}，不优于的比例 {e['fraction_not_better']:.3f}。"]
        L += ['', 'P2 的想法来自 v5 r1 在同一组 32 个子集、8 个 Stage-1 seed 上的 L1 分解；它在计算任何 L3_member 之前就预设了，'
                  '但不独立于这些子集。要确认需要新的子集。']
    L += ['', '## 全部层（对 6 seed 真值；另列对旧 / 新 attack seed 真值的 LDS）', '',
          '| 层 | LDS | 95% CI | p | 对旧 3 seed 真值 | 对新 3 seed 真值 |', '|---|---|---|---|---|---|']
    for name, b in s['ladder'].items():
        L.append(f"| {name} | {_f(b['spearman'])} | {_ci(b.get('bootstrap_ci_conditional_on_seed_panel'))} | "
                 f"{_f(b.get('permutation_p_one_sided'), 3, False)} | {_f(b.get('lds_truth_old'))} | {_f(b.get('lds_truth_new'))} |")
    sec = s.get('secondary') or {}
    if sec:
        L += ['', '## 次要比较（配对子集 bootstrap）', '', '| 比较 | 差 | 95% CI | 不优于的比例 |', '|---|---|---|---|']
        for name, e in sec.items():
            if e and 'difference' in e:
                L.append(f"| {name} | {e['difference']:+.3f} | {_ci(e['ci'])} | {e['fraction_not_better']:.3f} |")
            elif e:
                L.append(f"| {name} | {_f(e.get('lds_a'))}（LDS） | — | — |")
    curves = s.get('seed_curves')
    if curves:
        L += ['', '## L3 随训练 seed 数', '']
        for r in curves:
            L.append(f"- {r['predictor']}，k = {r['k']}：{r['mean']:+.3f} [{r['q025']:+.2f}, {r['q975']:+.2f}]")
    L += ['', '## 读法（事先写下）', '', '| 看到什么 | 说明 |', '|---|---|',
          '| P1 通过 | 6 个 attack seed 下，L3_full 与真值正相关（单侧 2.5%） |',
          '| P2 通过且 P3 区间在 0 以上 | member-only 在这组子集上优于 full；要作为结论，需在新子集上重复 |',
          '| P2 通过但 P3 区间含 0 | 两个变体都有信号，不能说 member-only 更好 |',
          '| 对新 3 seed 真值的 LDS 与对旧 3 seed 的接近 | 结果不依赖 v5 r1 的 attack 随机性 |',
          '| P1、P2 都不通过 | v5 r1 的 0.32 可能部分来自 attack seed 噪声 |']
    return '\n'.join(L) + '\n'
