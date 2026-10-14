"""REPORT_CN.md for v5: the v4 report (unchanged renderer) plus gate, target and hybrid sections."""
import report4  # v4, unchanged


def _f(x, nd=3, sign=True):
    if x is None:
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def _pct(x):
    return '—' if x is None else f'{x:.1%}'


def render(s, c, figs, partial):
    text = report4.render(s, c, figs, partial)
    text = text.replace('（v4）', '（v5：target 改为稳定配方）', 1)
    text = text.replace('shadow3：稳定配方；target：旧模型固定。', 'shadow3：稳定配方；target：稳定配方（v5 唯一改动）。', 1)
    L = ['', '## v5：校准门（smoke，只用全数据模型，事先固定）', '']
    g = s.get('gate') or {}
    if g:
        ln2 = '（ln 2）' if abs(g['threshold'] - 0.6931471805599453) < 1e-12 else ''
        L += [f"规则：每个主类别在全部全数据 baseline（训练 seed × attack seed）上的 J00 都低于 {g['threshold']:.4f}{ln2}。"
              f"结果：**{'通过' if g['passed'] else '未通过'}**。", '',
              '| 类别 | J00 最大 | J00 均值 | AUC 均值 | ECE 均值 | 自信判错率 | 通过 |', '|---|---|---|---|---|---|---|']
        for cls, v in g['per_class'].items():
            L.append(f"| {cls} | {v['J00_max']:.4f} | {v['J00_mean']:.4f} | {v['auc_mean']:.3f} | {v['ece_mean']:.3f} | "
                     f"{_pct(v['confident_wrong_mean'])} | {v['passed']} |")
        jr = g.get('jvp_reverse_identity') or {}
        L += ['', f"前向 JVP 与反向 h 的精确恒等式：相对误差 {jr.get('relative_error', float('nan')):.2e}（门槛 {jr.get('tolerance')}）。", '',
              'p_true 饱和程度（member 中 p_true < 0.99 的比例；v4 时 shadow 约 0.2–0.6%，旧 target 为 12.5–24.8%）：', '',
              '| 模型 | member 均值 | member <0.99 | nonmember 均值 | nonmember <0.99 |', '|---|---|---|---|---|']
        for name, v in g.get('p_true_profile', {}).items():
            L.append(f"| {name} | {_f(v['member']['mean'], 4, False)} | {_pct(v['member']['frac_below_0_99'])} | "
                     f"{_f(v['nonmember']['mean'], 4, False)} | {_pct(v['nonmember']['frac_below_0_99'])} |")
    t = s.get('target') or {}
    if t:
        L += ['', f"稳定配方 target：seed {t['seed']}，训练 {t['train_images']} 张、测试 {t['test_images']} 张；"
                  f"训练准确率 {t['train_accuracy']:.3f}，测试准确率 {t['test_accuracy']:.3f}。"]
    ts = s.get('truth_shape')
    if ts:
        r = ts['v4_reference']
        L += ['', '## v5：真值的形状（公共平移 vs 子集间差异）', '',
              '| | 平均 Δ | 子集间 SD | 平移/SD | 平移/J00 | ΔAUC 均值 |', '|---|---|---|---|---|---|',
              f"| v5 | {_f(ts['mean'], 4)} | {ts['sd_across_subsets']:.4f} | {_f(ts['shift_to_spread'], 1, False)} | "
              f"{_pct(ts['relative_shift'])} | {_f(ts['delta_auc_mean'], 4)} |",
              f"| v4（参照） | {r['mean']:+.4f} | {r['sd_across_subsets']:.4f} | {r['shift_to_spread']:.0f} | "
              f"{_pct(r['relative_shift'])} | {r['delta_auc_mean']:+.4f} |", '',
              f"ΔAUC 与 ΔCE 的子集排序相关：{_f(ts['spearman_delta_auc_vs_delta_ce'])}。"]
    h = s.get('hybrid') or {}
    if h.get('status') == 'complete':
        L += ['', '## v5：混合单元（诊断，不属于预设梯子）', '',
              '只把一组 shadow3 行换成子集模型的预测，其余行保持全数据预测，重训 attack。L1 按行组拆开，'
              f"用 γ_A* = {s['frozen']['attack_gamma_star']:g}。", '',
              '| 行组 | 平均 Δ | 占全效应 | 子集间 SD | 与全真值的 Spearman | 组内 L1 的 LDS（对组真值） | 95% CI | 组内 L1 对全真值 |',
              '|---|---|---|---|---|---|---|---|']
        for g, v in h['groups'].items():
            b = v['l1_group_lds_on_group_truth']
            ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
            L.append(f"| {g} | {_f(v['truth_mean'], 4)} | {_pct(v['fraction_of_full_shift'])} | {v['truth_sd']:.4f} | "
                     f"{_f(v['spearman_with_full_truth'])} | {_f(b['spearman'])} | "
                     f"[{_f(ci[0], 2)}, {_f(ci[1], 2)}] | {_f(v['l1_group_lds_on_full_truth'])} |")
        a = h['additivity']
        L += ['', f"可加性：Δdeleted + Δnonmember 的平均 {a['sum_mean']:+.4f}，全效应 {a['full_mean']:+.4f}，"
                  f"交互项平均 {a['interaction_mean']:+.4f}；两者的子集排序相关 {_f(a['spearman_sum_vs_full'])}。"
                  f"（retained 行的 L1 平均 {h['l1_retained_mean']:+.4f}；L1 按组拆分的加和误差 {h['l1_partition_max_abs_error']:.1e}。）"]
    elif h:
        L += ['', '## v5：混合单元', '', '未全部完成，未汇总。']
    L += ['', '## v5：读法（事先写下）', '',
          '| 看到什么 | 说明 |', '|---|---|',
          '| 校准门通过、平移/SD 明显小于 v4，L1 与 L2 变好、L3 仍差 | 断点在 Stage 1（IF 本身或 50% 删除的线性化） |',
          '| 校准门通过，但 L1 仍约为 0 | attack 一侧对 50% 删除的线性化本身不成立；下一步用 attack 展开超梯度或更小删除比例 |',
          '| L3 显著为正 | 跨阶段 IF 在配方一致时有效；v4 的阴性来自配方不一致 |',
          '| 某一行组的混合真值与全真值高度相关 | 真值的排序主要由该组行携带，可据此缩小归因对象 |']
    return text + '\n'.join(L) + '\n'
