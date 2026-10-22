"""REPORT_CN.md for the interface LDS, written from summary.json only."""


def _f(x, nd=3, sign=True):
    if x is None:
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def _ci(c):
    return '—' if not c else f'[{c[0]:+.2f}, {c[1]:+.2f}]'


def _p(x):
    return '—' if x is None else f'{x:.3f}'


def render(s, c, partial):
    d, fr = s['design'], s['frozen']
    g0 = f"{d['primary_gamma']:g}"
    L = ['# 接口层 LDS：Stage-1 归因直接预测四个类别概率的变化（不经过 attack）', '']
    if partial:
        L += [f"> **部分结果**：缺 {s['missing']}，只用已完成的 seed，不作阈值判定。", '']
    L += [f"{d['subsets']} 个子集（v3 img32，每个保留 50% 的 shadow3 训练图像）；shadow3 接口行 {d['rows']} 行"
          f"（训练行 {d['training_rows']}，heldout nonmember 行 {d['nonmember_rows']}），每行四个类别概率。",
          f"真值：重训子集模型与全数据模型的概率差，对 {len(d['seeds'])} 个 Stage-1 seed 平均。预测：静态阻尼 IF "
          f"J_p (H + γI)⁻¹ Σ_{{删除}} g_i（matched 归一化），对评分 seed 面板 {d['score_panel']} 平均。",
          f"H、g、γ 网格和模型都与 v6 r1 的 L3 相同；不训练任何 attack。评分面板规则：{fr['panel_rule']}。", '']
    # ------------------------------------------------------------------ gates
    gt = s.get('gates') or {}
    br = s.get('bridge') or {}
    L += ['## 核验', '',
          f"- 全数据接口概率与 v3 的最大差：{gt.get('p0_vs_v3_max_abs', 0):.1e}；v6 伴随 v̄ 逐位一致：{gt.get('vbar_vs_v6_max_abs')}。",
          f"- v̄ 经 VJP 重现 v6 的 h̄（smoke 相对误差 {s['smoke']['checks'].get('vbar_reproduces_v6_hbar_rel', float('nan')):.1e}）；"
          f"前向/反向恒等式最大相对误差 {gt.get('forward_reverse_max_rel', float('nan')):.1e}。",
          f"- Σ_r v̄·Δp_true 重现 v6 的 L1：最大差 {br.get('L1_reproduces_v6_max_abs', float('nan')):.1e}。",
          "- Σ_r v̄·Δp̂ 重现 v6 的 L3 分数（各 γ 最大相对差）："
          + '，'.join(f"γ={g} {('—' if v is None else format(v, '.1e'))}" for g, v in (br.get('v6_reproduction') or {}).items()) + '。']
    sol = s.get('solver') or {}
    if sol:
        L.append('- 共享子空间求解：' + '；'.join(f"seed{k} 维度 {v['dim']}、{v['matvecs']} 次 HVP、{v['hours']:.2f} 小时"
                                         for k, v in sol.items()) + '。')
    L.append('- 各 γ 的数值合格：' + '，'.join(f"γ={g} {'合格' if ok else '**不合格**'}" for g, ok in s['gamma_qualified'].items()) + '。')
    # ------------------------------------------------------------------ primary
    pr = s.get('primary') or {}
    L += ['', f'## 预设主检验（γ = {g0}）', '']
    if 'mean' in pr:
        ce = (s.get('ceilings') or {}).get('nonmember', {})
        L += [f"nonmember 行 × 4 个类别（{pr['outputs']} 个输出），每个输出在 {d['subsets']} 个子集上的 Spearman，取平均：**{pr['mean']:+.3f}**，"
              f"95% CI {_ci(pr['ci'])}，联合置换单侧 p = {_p(pr['perm_p_one_sided'])}"
              + ('' if pr.get('passed') is None else f"（{'达到' if pr['passed'] else '未达到'} 0.05）") + '。',
              f"真值噪声上限：4 对 4 seed 的逐输出可重复性 {_f(ce.get('half'))}，Spearman-Brown 到 8 seed {_f(ce.get('spearman_brown'))}，"
              f"对应可达到的平均 LDS 约 {_f(ce.get('ceiling'), 3, False)}。"]
    else:
        L.append(pr.get('status', '未计算'))
    # ------------------------------------------------------------------ per-output table
    po = s.get('per_output') or {}
    L += ['', f"## 逐输出 LDS（每行每个类别一个输出，跨 {d['subsets']} 个子集的 Spearman 再取平均）", '',
          '| 层 | nonmember 行 | 95% CI | p | 只看真实类别 | 训练行（含自身删除） | 95% CI | p |', '|---|---|---|---|---|---|---|---|']
    for name, e in po.items():
        if 'nonmember' not in e:
            L.append(f"| {name} | {e.get('status')} | | | | | | |")
            continue
        a, b = e['nonmember'], e['training']
        L.append(f"| {name} | {_f(a['mean'])} | {_ci(a.get('ci'))} | {_p(a['perm_p_one_sided'])} | {_f(a['p_true_only_mean'])} | "
                 f"{_f(b['mean'])} | {_ci(b.get('ci'))} | {_p(b['perm_p_one_sided'])} |")
    prim = f'IF (gamma={g0})'
    if prim in po and 'nonmember' in po[prim]:
        bc = po[prim]['nonmember']['by_class']
        L += ['', '按类别（nonmember 行，主 γ）：' + '，'.join(f'类别 {k} {_f(v)}' for k, v in bc.items()) + '。']
    tc = s.get('training_conditional') or {}
    if tc:
        L += ['', '训练行按删除状态拆开（只用该行被保留、或只用该行被删除的子集；描述性）：', '',
              '| 层 | 保留时 | 删除时 |', '|---|---|---|']
        for name, e in tc.items():
            L.append(f"| {name} | {_f(e['retained']['mean_lds'])} | {_f(e['deleted']['mean_lds'])} |")
    ce = s.get('ceilings') or {}
    if ce:
        L += ['', '噪声上限（真值）：' + '；'.join(f"{g}：可重复性 {_f(v.get('half'))} → 8 seed {_f(v.get('spearman_brown'))}，上限 {_f(v.get('ceiling'), 3, False)}"
                                          + (f"，预测的 seed 间可重复性 {_f(v['prediction_split_half'])}" if 'prediction_split_half' in v else '')
                                          for g, v in ce.items()) + '。']
    # ------------------------------------------------------------------ functionals
    fu = s.get('functionals') or {}
    if prim in fu:
        L += ['', f"## 行组平均的概率变化（γ = {g0}，跨 {d['subsets']} 个子集的 LDS）", '', '| 函数 | IF | 95% CI | proxy | 真实 Δθ 线性化 |', '|---|---|---|---|---|']
        for key, e in fu[prim].items():
            L.append(f"| {key} | {_f(e['spearman'])} | {_ci(e.get('ci'))} | {_f(fu.get('proxy', {}).get(key, {}).get('spearman'))} | "
                     f"{_f(fu.get('true dtheta (linear)', {}).get(key, {}).get('spearman'))} |")
    # ------------------------------------------------------------------ removal effect
    re_ = s.get('removal_effect') or {}
    if prim in re_:
        L += ['', f"## 自身删除效应（每个训练行：被删时的变化减去被保留时的变化，跨 {d['training_rows']} 行排序）", '',
              f"真值平均 Δp_true {re_['truth_mean_p_true']:+.4f}；真值的 seed 间可重复性 {_f(re_['truth_split_half_p_true'])}。", '',
              '| 层 | p_true 的 Spearman | 95% CI | 均值之比（预测/真值） |', '|---|---|---|---|']
        for name, e in re_.items():
            if isinstance(e, dict):
                L.append(f"| {name} | {_f(e['spearman_p_true'])} | {_ci(e['ci'])} | {_f(e['mean_ratio'], 3, False)} |")
    # ------------------------------------------------------------------ magnitude
    mg = s.get('magnitude') or {}
    if prim in mg:
        L += ['', '## 幅度（matched 预测；每个子集内跨行跨类别的 Pearson，再取平均）', '',
              '| 层 | 行组 | 子集内 Pearson | 真值对预测的斜率 | 范数比（预测/真值） |', '|---|---|---|---|---|']
        for name in (prim, f'IF logit (gamma={g0})', 'true dtheta (linear)', 'true dtheta (logit)'):
            for grp, e in (mg.get(name) or {}).items():
                L.append(f"| {name} | {grp} | {_f(e['within_subset_pearson'])} | {_f(e['slope_truth_on_pred'], 3, False)} | "
                         f"{_f(e['norm_ratio_pred_over_truth'], 3, False)} |")
    # ------------------------------------------------------------------ bridge
    if br.get('rungs'):
        L += ['', '## 回到 attack：用 v6 伴随 v̄ 收缩（对 v6 的 6 attack seed 真值）', '',
              'L3 = Σ_r v̄·Δp̂；把某一行组的 Δp̂ 换成重训的真实变化，看 L1 与 L3 之间的差距落在哪一组行。', '',
              '| 层 | LDS | 95% CI | p |', '|---|---|---|---|']
        for name, e in br['rungs'].items():
            L.append(f"| {name} | {_f(e['spearman'])} | {_ci(e.get('ci'))} | {_p(e.get('perm_p'))} |")
        if br.get('paired'):
            L += ['', '相对 L3 的配对差（同一组子集重采样）：', '']
            for name, e in br['paired'].items():
                if e:
                    L.append(f"- {name}：{e['difference']:+.3f} {_ci(e['ci'])}，不优于的比例 {e['fraction_not_better']:.3f}")
    # ------------------------------------------------------------------ reading
    L += ['', '## 读法（事先写下）', '',
          '| 看到什么 | 说明 |', '|---|---|',
          '| 主检验与 L3 同一量级（v6 r1 在 γ = 0.03 时 L3 = 0.27） | 导师的预期成立：跨阶段 L3 的水平主要由 Stage-1 IF 本身决定 |',
          '| 主检验明显高于 L3，接近噪声上限 | Stage-1 IF 对一般方向预测得好；L3 的损失来自 attack 关心的特定方向，看桥接表 |',
          '| 换掉被删行后 L3 接近 L1 | L1→L3 的差距主要来自被删行的自身删除效应预测 |',
          '| 换掉 nonmember 行后 L3 接近 L1 | 差距主要来自对 heldout 行的交叉影响预测 |',
          '| 训练行"删除时"明显低于"保留时" | IF 对样本自身被删的响应预测较差，这一项直接影响 membership |',
          '| 真实 Δθ 线性化明显差于 IF | 50% 删除的参数位移太大，IF 给出的局部响应反而更好（与 v5、v6 一致） |']
    return '\n'.join(L) + '\n'
