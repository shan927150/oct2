"""Collect-side analysis for v5: the v4 analysis unchanged, plus the calibration gate, the stable
target, a shift-versus-spread decomposition of the truth, and the hybrid diagnostic cells.

Hybrid conditions (per subset j, training seed s, attack seed k, class c):
    J_deleted   : only the deleted shadow-3 member rows take their subset predictions
    J_nonmember : only the shadow-3 nonmember rows take their subset predictions
They answer which rows carry the truth ranking (Spearman with the full truth), whether the effect is
additive (Delta_deleted + Delta_nonmember vs Delta_full), and whether the attack linearization works
when only one group moves (LDS of L1 restricted to the group against that group's hybrid truth).
"""
from pathlib import Path
import numpy as np
import runtime as rt
import analysis4 as A4           # v4, unchanged
import stats2 as S               # seed32 v2, unchanged
from analysis_stats import spearman  # LDS v1, unchanged

GROUPS = ('deleted', 'nonmember')


def load(root, c, allow_partial=False):
    root = Path(root)
    data = A4.load(root, c, allow_partial=allow_partial)
    h = rt.sha(root / 'config.json')
    data['gate'] = rt.read(root / 'smoke/gate.json')
    d = root / 'background' / 'target'
    rt.completed(d, h, dict(kind='target'))
    data['target'] = rt.read(d / 'result.json')
    hybrid = {}
    for cell in c['cells']:
        d = root / 'hybrid' / cell['name']
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(name=cell['name'], kind='hybrid', subset=cell['subset'], seed=cell['seed']))
            hybrid[(cell['subset'], cell['seed'])] = rt.read(d / 'result.json')
        else:
            data['missing'].append('hybrid/' + cell['name'])
    if data['missing'] and not allow_partial:
        raise ValueError(f"{len(data['missing'])} artefacts incomplete (first: {data['missing'][:5]}).")
    data['hybrid'] = hybrid
    return data


def hybrid_arrays(data, c, seeds):
    m = len(data['M'])
    K, C, gA = c['attack_seeds'], c['classes'], c['attack_gammas']
    shape = (m, len(seeds), len(K), len(C))
    out = {f'D_{g}': np.full(shape, np.nan) for g in GROUPS}
    out.update({f'L1_{g}': np.full(shape + (len(gA),), np.nan) for g in (*GROUPS, 'retained')})
    for j in range(m):
        for si, s in enumerate(seeds):
            rows = {(r['attack_seed'], r['cls']): r for r in data['hybrid'][(j, s)]['rows']}
            for ki, k in enumerate(K):
                for ci, cls in enumerate(C):
                    r = rows[(k, cls)]
                    for g in GROUPS:
                        out[f'D_{g}'][j, si, ki, ci] = r[f'J_{g}'] - r['J00']
                    for g in (*GROUPS, 'retained'):
                        out[f'L1_{g}'][j, si, ki, ci] = [r[f'L1_{g}'][f'{x:g}'] for x in gA]
    return out


def analyze(data, c):
    summary, tables = A4.analyze(data, c)
    summary['gate'] = data['gate']
    summary['target'] = data['target']
    summary['v4_reference_out'] = c.get('v4_reference_out')
    tables['hybrid'] = []
    ar = A4.arrays(data, c)
    seeds = ar['seeds']
    if not seeds:
        return summary, tables
    fr = data['frozen']
    omega, gstar = fr['omega'], fr['attack_gamma_star']
    nimg = data['M'].sum(1)
    D = A4.weighted(ar['D'], omega, c['classes'])            # [m, S, K]
    truth = D.mean((1, 2))
    # ------------------------------------------------ shift versus spread, and the AUC truth
    base_auc = {(r['seed'], r['attack_seed'], r['cls']): r['metrics']['auc'] for r in data['attack_base']}
    m = len(data['M'])
    dauc = np.zeros((m, len(seeds), len(c['attack_seeds']), len(c['classes'])))
    for j in range(m):
        for si, s in enumerate(seeds):
            for r in data['truth'][(j, s)]['rows']:
                dauc[j, si, c['attack_seeds'].index(r['attack_seed']), c['classes'].index(r['cls'])] = \
                    r['auc10'] - base_auc[(s, r['attack_seed'], r['cls'])]
    auc_truth = A4.weighted(dauc, omega, c['classes']).mean((1, 2))
    sd = float(truth.std(ddof=1))
    summary['truth_shape'] = dict(
        mean=float(truth.mean()), sd_across_subsets=sd,
        shift_to_spread=float(abs(truth.mean()) / sd) if sd > 0 else None,
        relative_shift=float(truth.mean() / summary['truth']['J00_mean']) if summary['truth'].get('J00_mean') else None,
        delta_auc_mean=float(auc_truth.mean()), delta_auc_sd=float(auc_truth.std(ddof=1)),
        spearman_delta_auc_vs_delta_ce=spearman(auc_truth, truth),
        v4_reference=dict(mean=-0.741, sd_across_subsets=0.0045, shift_to_spread=165., relative_shift=-0.645,
                          delta_auc_mean=-0.004))
    # ------------------------------------------------------------------- hybrid
    if not all((j, s) in data['hybrid'] for j in range(m) for s in seeds):
        summary['hybrid'] = dict(status='incomplete')
        return summary, tables
    H = hybrid_arrays(data, c, seeds)
    gi = c['attack_gammas'].index(gstar)
    hyb = dict(status='complete', groups={})
    parts = {}
    for g in GROUPS:
        t = A4.weighted(H[f'D_{g}'], omega, c['classes']).mean((1, 2))
        l1 = A4.weighted(H[f'L1_{g}'], omega, c['classes'])[..., gi].mean((1, 2))
        parts[g] = t
        b = S.lds_block(l1, t, nimg)
        hyb['groups'][g] = dict(truth_mean=float(t.mean()), truth_sd=float(t.std(ddof=1)),
                                fraction_of_full_shift=float(t.mean() / truth.mean()) if truth.mean() != 0 else None,
                                spearman_with_full_truth=spearman(t, truth),
                                l1_group_lds_on_group_truth=b,
                                l1_group_lds_on_full_truth=S.lds_block(l1, truth, nimg)['spearman'])
        ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
        tables['hybrid'].append(dict(group=g, truth_mean=float(t.mean()), truth_sd=float(t.std(ddof=1)),
                                     spearman_with_full_truth=hyb['groups'][g]['spearman_with_full_truth'],
                                     l1_lds=b['spearman'], l1_ci_low=ci[0], l1_ci_high=ci[1],
                                     l1_lds_on_full_truth=hyb['groups'][g]['l1_group_lds_on_full_truth']))
    add = parts['deleted'] + parts['nonmember']
    hyb['additivity'] = dict(sum_mean=float(add.mean()), full_mean=float(truth.mean()),
                             interaction_mean=float((truth - add).mean()), spearman_sum_vs_full=spearman(add, truth))
    l1_ret = A4.weighted(H['L1_retained'], omega, c['classes'])[..., gi].mean((1, 2))
    l1_parts = sum(A4.weighted(H[f'L1_{g}'], omega, c['classes'])[..., gi].mean((1, 2)) for g in GROUPS) + l1_ret
    l1_full = A4.weighted(ar['L1'], omega, c['classes'])[..., gi].mean((1, 2))
    hyb['l1_partition_max_abs_error'] = float(np.abs(l1_parts - l1_full).max())   # L1 is additive by construction
    hyb['l1_retained_mean'] = float(l1_ret.mean())
    summary['hybrid'] = hyb
    return summary, tables
