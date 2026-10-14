"""REPORT_CN.md for v5 r1: the v4 report (unchanged renderer) plus the v5 sections, from summary.json only."""
import report4  # v4, unchanged

LN2 = 0.6931471805599453


def _f(x, nd=3, sign=True):
    if x is None:
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def _pct(x):
    return '—' if x is None else f'{x:.1%}'


def _ci(d):
    c = d.get('ci') if isinstance(d, dict) else None
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def _bci(b):
    c = (b or {}).get('bootstrap_ci_conditional_on_seed_panel')
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def render(s, c, figs, partial):
    text = report4.render(s, c, figs, partial)
    text = text.replace('（v4）', '（v5 r1：target 改为稳定配方）', 1)
    text = text.replace('shadow3：稳定配方；target：旧模型固定。',
                        'shadow3：稳定配方；target：稳定配方（唯一新增的 CNN；背景 shadow 复用 v4）。', 1)
    L = []
    g = s.get('gate') or {}
    if g:
        tag = '（ln 2）' if abs(g['threshold'] - LN2) < 1e-12 else ''
        L += ['', '## v5：CE 质量门（smoke，只用全数据模型，事先固定）', '',
              f"规则：每个主类别在全部全数据 baseline 上的 J00 都低于 {g['threshold']:.4f}{tag}。这是筛掉\"自信判错\"attack 的质量门，"
              f"不是校准检验。结果：**{'通过' if g['passed'] else '未通过'}**。", '',
              '| 类别 | J00 最大 | J00 均值 | 常数先验 CE | AUC | Brier | ECE | 自信判错率 | 旧 target J00 | 旧 target（fp64 路径）J00 | 通过 |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for cls, v in g['per_class'].items():
            L.append(f"| {cls} | {v['J00_max']:.4f} | {v['J00_mean']:.4f} | {_f(v.get('prior_ce'), 4, False)} | {_f(v.get('auc_mean'), 3, False)} | "
                     f"{_f(v.get('brier_mean'), 3, False)} | {_f(v.get('ece_mean'), 3, False)} | {_pct(v.get('confident_wrong_mean'))} | "
                     f"{_f(v.get('legacy_J00_mean'), 4, False)} | {_f(v.get('legacy_fp64_J00_mean'), 4, False)} | {v['passed']} |")
        jr, rp = g.get('jvp_reverse_identity') or {}, g.get('v4_table_replay') or {}
        pp = g.get('legacy_precision_path_max_abs')
        L += ['', f"同一批 attack 同时在三套 query 上评估：稳定 target（主）、旧 target（v4 路径）、旧 checkpoint 走 fp64 路径。"
                  f"旧 target 两条推理路径的最大概率差：{'—' if pp is None else format(pp, '.2e')}。",
              f"v4 attack 表与旧 target query 的逐字节复现：{rp.get('passed') if rp.get('checked') else '未检查'}。"
              f"前向 JVP 与反向 h 的精确恒等式：相对误差 {jr.get('relative_error', float('nan')):.2e}（门槛 {jr.get('tolerance')}）。", '']
        prof = g.get('p_true_profile') or {}
        cls_list = [str(x) for x in (s.get('frozen') or {}).get('qualified_classes', [])]
        if prof and cls_list:
            head = ' | '.join(f'类别 {x} member <0.99' for x in cls_list)
            L += ['p_true 饱和程度（各模型、主类别；member 中 p_true < 0.99 的比例）：', '',
                  f'| 模型 | 全部 member <0.99 | {head} |', '|' + '---|' * (2 + len(cls_list))]
            for name, v in prof.items():
                bc = v.get('by_class', {})
                cells = ' | '.join(_pct(bc.get(x, {}).get('member', {}).get('frac_below_0_99')) for x in cls_list)
                L.append(f"| {name} | {_pct(v['all']['member']['frac_below_0_99'])} | {cells} |")
    t = s.get('target') or {}
    if t:
        L += ['', f"稳定配方 target：seed {t['seed']}，训练 {t['train_images']} 张、测试 {t['test_images']} 张；"
                  f"训练准确率 {t['train_accuracy']:.3f}，测试准确率 {t['test_accuracy']:.3f}。"]
    ts = s.get('truth_shape')
    if ts:
        n = ts['new_target']
        L += ['', '## v5：真值的形状', '',
              f"平均 Δ {n['mean']:+.4f}，子集间 SD {n['sd_across_subsets']:.4f}，平移/SD {_f(n['shift_to_spread'], 1, False)}，"
              f"平移/J00 {_pct(n['relative_shift'])}；ΔAUC 均值 {ts['delta_auc_mean']:+.4f}，ΔAUC 与 ΔCE 的子集排序相关 "
              f"{_f(ts['spearman_delta_auc_vs_delta_ce'])}"
              + (f"，ΔBrier 与 ΔCE {_f(ts['spearman_delta_brier_vs_delta_ce'])}" if 'spearman_delta_brier_vs_delta_ce' in ts else '') + '。',
              '公共平移本身不改变 Spearman，这些数只描述真值，不解释 LDS。']
    po = s.get('projection_only')
    if po or s.get('ladder_extra'):
        L += ['', '## v5：L2-retrain 的投影对照', '']
        if po:
            L.append(f"只做截断归一化 Π(P0)、参数不动时的变化：平均 {po['mean_shift']:+.4f}，子集间 SD {po['sd_across_subsets']:.1e}"
                     '（只依赖 seed，只移动水平，不改变子集排序）。')
        L += ['', '| 层 | LDS | 95% CI | p |', '|---|---|---|---|']
        for name, b in (s.get('ladder_extra') or {}).items():
            L.append(f"| {name} | {_f(b['spearman'])} | {_bci(b)} | {_f(b.get('permutation_p_one_sided'), 3, False)} |")
        d = s.get('ladder_extra_difference')
        if d:
            L += ['', f"logit 线性化 − 概率线性化（同一子集重采样）：{d['difference']:+.3f} {_ci(d)}。"]
    lv = s.get('legacy_in_v5')
    if lv:
        L += ['', '## v5：同一批 attack 在旧 target 上', '',
              f"旧 target 真值与新 target 真值的子集排序相关：{_f(lv['spearman_truth_new_vs_old'])}；旧 target 真值平均 {lv['truth']['mean']:+.4f}，"
              f"子集间 SD {lv['truth']['sd_across_subsets']:.4f}。", '', '| 层（旧 target） | LDS | 95% CI |', '|---|---|---|']
        for name, b in lv['rungs'].items():
            L.append(f"| {name} | {_f(b['spearman'])} | {_bci(b)} |")
    pv = s.get('paired_v4') or {}
    if pv:
        L += ['', '## v5：与 v4（旧 target）的配对比较', '']
        if 'replay_max_abs' in pv:
            r = pv['replay_max_abs']
            L.append(f"复现检查（{pv['replay_rows']} 行，容差 {pv['tolerance']}）：J00 {r['J00']:.1e}，J10 {r['J10']:.1e}，"
                     f"L2-retrain {r['J_L2_retrain']:.1e}；{'通过' if pv['replay_passed'] else '**未通过**'}。"
                     f"新旧 target 真值的子集排序相关 {_f(pv.get('truth_spearman_new_vs_old'))}。")
        if pv.get('status') == 'paired':
            L += ['', '同一组子集重采样下的 LDS 差（新 − 旧）。只看差值的区间，不比较两边各自是否显著。', '',
                  '| 层 | 新 target | 旧 target（v4） | 差 | 95% CI | 不优于的比例 |', '|---|---|---|---|---|---|']
            for name, d in pv['rungs'].items():
                L.append(f"| {name} | {_f(d['lds_a'])} | {_f(d['lds_b'])} | {_f(d['difference'])} | {_ci(d)} | {d['fraction_not_better']:.3f} |")
        else:
            L.append(f"状态：{pv.get('status')}。")
    h = s.get('hybrid') or {}
    if h.get('status') == 'complete':
        L += ['', '## v5：混合单元（诊断，不属于预设梯子）', '',
              '只把一部分 shadow3 行换成子集模型的预测，其余行保持全数据预测，重训 attack（主类别）。组均值之比只作描述，不是因果占比。', '']
        for target, v in h['targets'].items():
            L += [f"**{'新' if target == 'new' else '旧'} target**", '',
                  '| 混合 | 平均 Δ | 与全效应均值之比 | 子集间 SD | 与全真值的 Spearman | 组内 L1 的 LDS（对组真值） |', '|---|---|---|---|---|---|']
            for name, e in v['groups'].items():
                b = e.get('l1_group_lds_on_group_truth')
                L.append(f"| {name} | {_f(e['truth_mean'], 4)} | {_f(e['mean_ratio_to_full'], 2, False)} | {e['truth_sd']:.4f} | "
                         f"{_f(e['spearman_with_full_truth'])} | {(_f(b['spearman']) + ' ' + _bci(b)) if b else '—'} |")
            r, i = v['nonadditive_residual'], v['interaction_deleted_nonmember']
            L += ['', f"非加性残差（full − D − R − N）平均 {r['mean']:+.4f}（SD {r['sd']:.4f}）；"
                      f"deleted × nonmember 交互（DN − D − N，retained 保持 baseline）平均 {i['mean']:+.4f}（SD {i['sd']:.4f}）。", '']
        L.append(f"L1 按行组拆分的加和误差 {h['l1_partition_max_abs_error']:.1e}。")
    ip = s.get('interp') or {}
    if ip.get('status') == 'complete':
        p = ip['panel']
        L += ['', '## v5：Stage 2 接口插值（小面板，不做 LDS 显著性宣称）', '',
              f"P(t) = P0 + t(Pj − P0)，子集 {p['subsets']}、训练 seed {p['seeds']}、全部 attack seed、主类别；"
              f"每个 t 用相同的 attack 初始化和 batch 顺序。t 是接口插值比例，不是删除比例。γ_A = {ip['attack_gamma']:g}。", '',
              '| t | target | 单元 | 真实斜率中位数 | L1 中位数 | 相对误差中位数 | 符号一致 | 斜率与 L1 的 Spearman | float32 有效比例 |',
              '|---|---|---|---|---|---|---|---|---|']
        for t, v in ip['by_t'].items():
            for target in ('new', 'legacy'):
                e = v[target]
                L.append(f"| {t} | {'新' if target == 'new' else '旧'} | {e['units']} | {e['median_true_slope']:+.2e} | {e['median_l1']:+.2e} | "
                         f"{_f(e['median_relative_error'], 2, False)} | {_pct(e['sign_agreement'])} | {_f(e['spearman_slope_vs_l1'])} | "
                         f"{_f(v['float32_effective_ratio_median'], 3, False)} |")
    L += ['', '## v5：读法（事先写下）', '',
          '| 看到什么 | 说明 |', '|---|---|',
          '| CE 质量门不过 | 配方对齐后 attack 在 target 上仍然大量自信判错；停止，先检查数据，不调门槛 |',
          '| 插值：小 t 时真实斜率与 L1 一致，大 t 失真 | attack 一侧的局部导数可信，问题在大幅接口变化的非线性；考虑较小删除比例或路径积分 |',
          '| 插值：小 t 时真实斜率稳定，但与 L1 不一致 | 静态驻点 IF 与有限步 Adam 训练映射不符；优先做 attack 展开导数或匹配驻点的简单 attack |',
          '| 插值：小 t 时响应为零或不稳定 | 先看数值分辨率和非光滑性，不能直接归因于公式 |',
          '| L2-retrain 的 logit 版与概率版差别大 | L2-retrain 的信号依赖投影规则，不能用它说明参数线性化准确 |',
          '| 新 target 的 L3 为正，且对 v4 的配对差区间在 0 以上 | 在这个 split、target seed 和 seed 面板下，配方匹配配置的 L3 优于旧配置；推广到其他 target seed 需另测 |',
          '| 某个混合真值与全真值高度相关 | 子集排序主要由这组行携带（描述性） |']
    return text + '\n'.join(L) + '\n'
