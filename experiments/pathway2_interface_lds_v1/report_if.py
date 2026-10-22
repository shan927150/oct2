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
          f"真值：重训子集模型与全数据模型的概率差。预测：静态阻尼 IF，采用 matched 归一化。"
          f"两者都对同一个 Stage-1 seed 面板 {d['score_panel']} 平均。全部 {len(d['seeds'])} 个 seed 真值另作补充。",
          f"H、g、γ 网格和模型都与 v6 r1 的 L3 相同；不训练任何 attack。评分面板规则：{fr['panel_rule']}。", '']
    L += ['预测公式：Δp̂_j = J_p(H+γI)⁻¹[(N/K_j)Σ_deleted g_i − (N/K_j−1)Σ_all g_i]，g_i=∇CE_i/N。'
          'proxy 使用同一括号内向量，省略逆 Hessian。', '',
          '这是复用 v3 子集的机制诊断，不是新子集上的确认实验。主检验的置换和 bootstrap 单位均为完整子集，'
          '区间条件于固定接口行和模型种子，不涵盖新患者或新训练种子的总体不确定性。'
          '四类概率有和为一的约束，不能把全部输出当作独立重复。', '']
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
        L += [f"nonmember 行 × 4 个类别（{pr['outputs']} 个有效输出），每个输出在 {d['subsets']} 个子集上的 Spearman，取平均：**{_f(pr['mean'])}**，"
              f"95% CI {_ci(pr['ci'])}，联合置换单侧 p = {_p(pr['perm_p_one_sided'])}"
              + ('' if pr.get('passed') is None else f"（{'达到' if pr['passed'] else '未达到'} 0.05）") + '。',
              f"真值的 balanced split-half Spearman 可重复性 {_f(ce.get('half'))}（{ce.get('seeds')} seeds，"
              f"每半 {ce.get('half_size')} seeds，{ce.get('splits')} 个不同拆分）。",
              f"固定有效输出集合的 bootstrap：{pr.get('bootstrap_valid')}/{pr.get('bootstrap_requested')} 次有效。"
              '重采样后存在常数输出的整次抽样记为无效，若有效比例不足 95%，不提供 CI。',
              'split-half 不是本检验的严格噪声上限。另报 Pearson 的条件性衰减参考值，仅对正可靠性输出计算，'
              '需要独立、可交换的加性噪声等假设；预测与真值共享模型种子时这些假设未必成立。']
        sec = s.get('secondary_all_seed_truth')
        if sec:
            L += [f"补充：预测面板对全部 seed 真值的 nonmember LDS {_f(sec['mean'])}，CI {_ci(sec.get('ci'))}。"
                  '这是另一个 estimand，不替代主检验。']
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
        L += ['', '真值可重复性：' + '；'.join(f"{g}：主面板 Spearman {_f(v.get('half'))}，全部 seed Spearman {_f(v.get('all_seed_truth', {}).get('half'))}，"
                                          f"Pearson 衰减参考 {_f(v.get('attenuation_reference_pearson_positive_only'), 3, False)}"
                                          f"（正可靠性输出 {v.get('positive_reliability_outputs', 0)}/{v.get('total_outputs', 0)}）"
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
              f"真值平均 Δp_true {_f(re_['truth_mean_p_true'], 4)}；真值的 seed 间可重复性 {_f(re_['truth_split_half_p_true'])}。"
              f"仅纳入同时出现过删除和保留的 {re_.get('eligible_rows')} 行。该表描述性，不对共享患者/子集的图像作独立 bootstrap。", '',
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
        L += ['', '## 回到 attack：用 v6 伴随 v̄ 收缩（历史复现与行组诊断）', '',
              'L3 = Σ_r v̄·Δp̂；把某一行组的 Δp̂ 换成重训的真实变化。各行组比较使用相同 Stage-1 seed 面板的预测和 attack 真值。'
              '此处沿用 v6 的 attack seeds，只做历史复现，不报告同种子置换显著性。'
              '后续跨阶段正式评估仍需评分器与真值使用不相交的 attack seeds。', '',
              '| 层 | LDS | 95% CI | p |', '|---|---|---|---|']
        for name, e in br['rungs'].items():
            L.append(f"| {name} | {_f(e['spearman'])} | {_ci(e.get('ci'))} | {_p(e.get('perm_p'))} |")
        if br.get('paired'):
            L += ['', '相对 L3 的配对差（同一组子集重采样）：', '']
            for name, e in br['paired'].items():
                if e:
                    L.append(f"- {name}：{e['difference']:+.3f} {_ci(e['ci'])}，bootstrap 中不优于的比例 {_p(e['fraction_not_better'])}（不是 p 值）")
    # ------------------------------------------------------------------ reading
    L += ['', '## 读法（事先写下）', '',
          '| 看到什么 | 说明 |', '|---|---|',
          '| 主检验与 L3 数值相近 | 与导师的预期一致，但两个指标不同，不能据此分配跨阶段误差或认定 Stage 1 是唯一来源 |',
          '| 主检验高于 L3 | 逐输出排名与 attack 加权方向可能表现不同，需结合幅度、符号和桥接诊断 |',
          '| 换掉某组行后 L3 接近 L1 | 在固定 v̄ 下，该组修正改善了这个指标；各组的 Spearman 改善不能相加为误差占比 |',
          '| 训练行"删除时"低于"保留时" | 该分组上排序较差；还需考虑有效子集数与真值可重复性差异 |',
          '| 真实 Δθ 线性化较差 | 提示有限参数位移下的局部线性化值得检查，不能仅凭此表证明 Hessian 是唯一问题 |']
    return '\n'.join(L) + '\n'
