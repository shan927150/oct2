"""Collect-side analysis for v5 r1. The v4 analysis runs unchanged on the stable-target ladder; this
module adds:

- CE quality gate, stable target and p_true profiles (from smoke);
- extra L2 diagnostics: projection-only shift J(A(Pi(P0))) - J00, L2-retrain relative to Pi(P0)
  (same subset ranking as L2-retrain by construction, only the level moves), and a logit-linearized
  L2-retrain that needs no projection;
- the legacy target inside v5 (same attacks), and the pairing with the v4 run: an exact replay
  check of J00 / J10 / L2-retrain on the legacy target against the v4 cells, then LDS(new target)
  - LDS(old target) per rung, bootstrapped over the same 32 subsets;
- hybrids with correct names: Delta_full - Delta_D - Delta_R - Delta_N is the total non-additive
  residual, Delta_DN - Delta_D - Delta_N is the deleted x nonmember interaction with retained rows
  at baseline; group shares of a mean are descriptive, not causal percentages;
- the interpolation panel: J(t) - J(0) against t * L1 on a fixed small panel (no LDS claim).
"""
from pathlib import Path
import numpy as np
import runtime as rt
import analysis4 as A4           # v4, unchanged
import stats2 as S               # seed32 v2, unchanged
from analysis_stats import spearman  # LDS v1, unchanged

HYBRIDS = ('deleted', 'retained', 'nonmember', 'deleted_nonmember')
GROUPS = ('deleted', 'retained', 'nonmember')


# ------------------------------------------------------------------------- loading
def load(root, c, allow_partial=False):
    root = Path(root)
    data = A4.load(root, c, allow_partial=allow_partial)
    h = rt.sha(root / 'config.json')
    manifest = rt.completed(root / 'smoke', h, dict(kind='xsmoke'))
    for f in ('gate.json', 'attack_baselines_all_targets.json'):
        if f not in manifest['output_sha256']:
            raise ValueError(f'smoke manifest lacks {f}; not a v5 r1 smoke')
    data['gate'] = rt.read(root / 'smoke/gate.json')
    data['baselines_all'] = rt.read(root / 'smoke/attack_baselines_all_targets.json')['rows']
    rt.completed(root / 'background' / 'target', h, dict(kind='target'))
    data['target'] = rt.read(root / 'background/target/result.json')
    data['hybrid'] = {}
    for cell in c['cells']:
        d = root / 'hybrid' / cell['name']
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(name=cell['name'], kind='hybrid', subset=cell['subset'], seed=cell['seed']))
            data['hybrid'][(cell['subset'], cell['seed'])] = rt.read(d / 'result.json')
        else:
            data['missing'].append('hybrid/' + cell['name'])
    if (root / 'interp/complete.json').exists():
        rt.completed(root / 'interp', h, dict(kind='interp'))
        data['interp'] = rt.read(root / 'interp/result.json')
    else:
        data['interp'] = None
        data['missing'].append('interp')
    data['v4'] = None
    if c.get('v4_out'):
        v4 = Path(c['v4_out'])
        if rt.sha(v4 / 'config.json') != c['v4_config_sha256']:
            raise ValueError('v4 config changed since prepare')
        v4c = rt.read(v4 / 'config.json')
        data['v4'] = dict(c=v4c, data=A4.load(v4, v4c, allow_partial=False))
    if data['missing'] and not allow_partial:
        raise ValueError(f"{len(data['missing'])} artefacts incomplete (first: {data['missing'][:5]}).")
    return data


# ------------------------------------------------------------------------- helpers
def ladder_inputs(data, c):
    """Truth and rung predictions per subset, exactly as analysis4.analyze builds them."""
    ar = A4.arrays(data, c)
    fr = data['frozen']
    omega, C = fr['omega'], c['classes']
    truth = A4.weighted(ar['D'], omega, C).mean((1, 2))
    preds = {f'L1 (gamma_A={g:g})': A4.weighted(ar['L1'], omega, C)[..., gi].mean((1, 2))
             for gi, g in enumerate(c['attack_gammas'])}
    preds['L2-retrain'] = A4.weighted(ar['L2r'], omega, C).mean((1, 2))
    preds['L2-lin'] = A4.weighted(ar['L2'], omega, C).mean((1, 2))
    for g in c['gammas']:
        P, _ = A4.l3_predictions(data, c, g)
        if P is not None:
            preds[f'L3 (gamma={g:g})'] = P.mean(0)
    if c['score_seeds'] and all(s in data['scores'] for s in c['score_seeds']):
        preds['proxy'] = np.mean([data['scores'][s]['arrays']['pred_proxy'] for s in c['score_seeds']], 0)
    return truth, preds, ar['seeds']


def field(data, c, seeds, name, minus='J00'):
    """[m, S, K, C] array of row[name] - row[minus] from the v5 truth cells."""
    m = len(data['M'])
    out = np.full((m, len(seeds), len(c['attack_seeds']), len(c['classes'])), np.nan)
    for j in range(m):
        for si, s in enumerate(seeds):
            for r in data['truth'][(j, s)]['rows']:
                out[j, si, c['attack_seeds'].index(r['attack_seed']), c['classes'].index(r['cls'])] = \
                    r[name] - (r[minus] if minus else 0.)
    return out


def paired(p1, t1, p2, t2, draws=2000, seed=20261014):
    """LDS(p1, t1) - LDS(p2, t2) with the same subset resample for both."""
    a, b = spearman(p1, t1), spearman(p2, t2)
    if a is None or b is None:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        i = rng.integers(len(t1), size=len(t1))
        x, y = spearman(p1[i], t1[i]), spearman(p2[i], t2[i])
        if x is not None and y is not None:
            vals.append(x - y)
    vals = np.asarray(vals)
    return dict(lds_a=float(a), lds_b=float(b), difference=float(a - b), ci=np.quantile(vals, [.025, .975]).tolist(),
                fraction_not_better=float((vals <= 0).mean()), draws=len(vals))


def shape(truth, j00_mean=None):
    sd = float(truth.std(ddof=1))
    return dict(mean=float(truth.mean()), sd_across_subsets=sd, shift_to_spread=float(abs(truth.mean()) / sd) if sd > 0 else None,
                relative_shift=float(truth.mean() / j00_mean) if j00_mean else None)


def block(pred, truth, nimg):
    b = S.lds_block(pred, truth, nimg)
    ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
    return b, dict(spearman=b['spearman'], ci_low=ci[0], ci_high=ci[1], perm_p=b.get('permutation_p_one_sided'))


# ------------------------------------------------------------------------- analysis
def analyze(data, c):
    summary, tables = A4.analyze(data, c)
    for key in ('ladder_extra', 'paired_v4', 'hybrid', 'interp', 'legacy_in_v5'):
        tables[key] = []
    summary['gate'] = data.get('gate')
    summary['target'] = data.get('target')
    truth, preds, seeds = ladder_inputs(data, c)
    if not len(seeds):
        return summary, tables
    fr, C = data['frozen'], c['classes']
    omega = fr['omega']
    nimg = data['M'].sum(1)
    W = lambda x: A4.weighted(x, omega, C).mean((1, 2))
    has = lambda name: all(name in r for t in data['truth'].values() for r in t['rows'])
    # ------------------------------------------------ truth shape and AUC / Brier views
    base_auc = {(r['seed'], r['attack_seed'], r['cls']): r['metrics']['auc'] for r in data['attack_base']}
    m = len(data['M'])
    dauc = np.zeros((m, len(seeds), len(c['attack_seeds']), len(C)))
    for j in range(m):
        for si, s in enumerate(seeds):
            for r in data['truth'][(j, s)]['rows']:
                dauc[j, si, c['attack_seeds'].index(r['attack_seed']), C.index(r['cls'])] = \
                    r['auc10'] - base_auc[(s, r['attack_seed'], r['cls'])]
    auc_truth = W(dauc)
    summary['truth_shape'] = dict(new_target=shape(truth, summary['truth'].get('J00_mean')),
                                  delta_auc_mean=float(auc_truth.mean()),
                                  spearman_delta_auc_vs_delta_ce=spearman(auc_truth, truth),
                                  note='A common shift does not change Spearman; it only describes the truth.')
    if has('brier10'):
        summary['truth_shape']['spearman_delta_brier_vs_delta_ce'] = spearman(W(field(data, c, seeds, 'brier10', 'brier00')), truth)
    # ------------------------------------------------ extra L2 diagnostics (new target)
    extra = {}
    if has('J_proj0'):
        proj_shift = W(field(data, c, seeds, 'J_proj0'))
        summary['projection_only'] = dict(mean_shift=float(proj_shift.mean()), sd_across_subsets=float(proj_shift.std(ddof=1)),
                                          note='Pi(P0) depends on the seed only; it moves the level, not the subset ranking.')
        extra['L2-retrain (vs Pi(P0))'] = W(field(data, c, seeds, 'J_L2_retrain', 'J_proj0'))
    if has('J_L2_logit'):
        extra['L2-retrain (logit linearization)'] = W(field(data, c, seeds, 'J_L2_logit'))
    summary['ladder_extra'] = {}
    for name, p in extra.items():
        b, row = block(p, truth, nimg)
        summary['ladder_extra'][name] = b
        tables['ladder_extra'].append(dict(rung=name, **row))
    if 'L2-retrain (logit linearization)' in extra:
        summary['ladder_extra_difference'] = paired(extra['L2-retrain (logit linearization)'], truth, preds['L2-retrain'], truth)
    # ------------------------------------------------ legacy target inside v5 (same attacks)
    if has('J10_legacy'):
        t_old = W(field(data, c, seeds, 'J10_legacy', 'J00_legacy'))
        leg = dict(truth=shape(t_old), spearman_truth_new_vs_old=spearman(truth, t_old), rungs={})
        for name, fld, minus in (('L2-retrain', 'J_L2_retrain_legacy', 'J00_legacy'),
                                 ('L2-retrain (logit linearization)', 'J_L2_logit_legacy', 'J00_legacy')):
            b, row = block(W(field(data, c, seeds, fld, minus)), t_old, nimg)
            leg['rungs'][name] = b
            tables['legacy_in_v5'].append(dict(rung=name, **row))
        summary['legacy_in_v5'] = leg
    # ------------------------------------------------ pairing with the v4 run
    summary['paired_v4'] = pair_with_v4(data, c, truth, preds, seeds, tables)
    # ------------------------------------------------ hybrids and interpolation
    summary['hybrid'] = hybrid_summary(data, c, seeds, truth, tables)
    summary['interp'] = interp_summary(data.get('interp'), fr, tables)
    return summary, tables


def pair_with_v4(data, c, truth, preds, seeds, tables):
    v4 = data.get('v4')
    if not v4:
        return dict(status='no v4 run configured')
    d4, c4 = v4['data'], v4['c']
    worst = dict(J00=0., J10=0., J_L2_retrain=0.)
    n = 0
    for (j, s), cell in data['truth'].items():
        old = {(r['attack_seed'], r['cls']): r for r in d4['truth'][(j, s)]['rows']}
        for r in cell['rows']:
            o = old[(r['attack_seed'], r['cls'])]
            for key, new in (('J00', 'J00_legacy'), ('J10', 'J10_legacy'), ('J_L2_retrain', 'J_L2_retrain_legacy')):
                worst[key] = max(worst[key], abs(r[new] - o[key]))
            n += 1
    tol = c['j00_tol']
    out = dict(replay_max_abs=worst, replay_rows=n, tolerance=tol, replay_passed=all(v <= tol for v in worst.values()))
    t4, p4, s4 = ladder_inputs(d4, c4)
    out['truth_spearman_new_vs_old'] = spearman(truth, t4)
    out['old_target_truth'] = shape(t4)
    if not out['replay_passed'] or list(s4) != list(seeds):
        out['status'] = 'replay failed or seed panels differ: pairing not reported'
        return out
    out['status'] = 'paired'
    out['rungs'] = {}
    for name in preds:
        if name in p4:
            d = paired(preds[name], truth, p4[name], t4)
            if d is None:
                continue
            out['rungs'][name] = d
            tables['paired_v4'].append(dict(rung=name, lds_new=d['lds_a'], lds_old=d['lds_b'], difference=d['difference'],
                                            ci_low=d['ci'][0], ci_high=d['ci'][1], fraction_not_better=d['fraction_not_better']))
    return out


def hybrid_summary(data, c, seeds, truth, tables):
    m = len(data['M'])
    if not data['hybrid'] or not all((j, s) in data['hybrid'] for j in range(m) for s in seeds):
        return dict(status='incomplete')
    fr, C, K, gA = data['frozen'], c['classes'], c['attack_seeds'], c['attack_gammas']
    omega, gi = fr['omega'], gA.index(fr['attack_gamma_star'])
    W = lambda x: A4.weighted(x, omega, C).mean((1, 2))
    shape4 = (m, len(seeds), len(K), len(C))
    D = {t: {h: np.zeros(shape4) for h in HYBRIDS} for t in ('new', 'legacy')}   # weight-0 classes stay 0
    L1 = {g: np.zeros(shape4) for g in GROUPS}
    for j in range(m):
        for si, s in enumerate(seeds):
            for r in data['hybrid'][(j, s)]['rows']:
                idx = (j, si, K.index(r['attack_seed']), C.index(r['cls']))
                for g in GROUPS:
                    L1[g][idx] = r['L1_by_group'][g][f'{gA[gi]:g}']
                if 'J' in r:
                    for h in HYBRIDS:
                        D['new'][h][idx] = r['J'][h] - r['J00']
                        D['legacy'][h][idx] = r['J_legacy'][h] - r['J00_legacy']
    full = dict(new=truth)
    if all('J10_legacy' in r for t in data['truth'].values() for r in t['rows']):
        full['legacy'] = W(field(data, c, seeds, 'J10_legacy', 'J00_legacy'))
    out = dict(status='complete', targets={})
    l1 = {g: W(L1[g]) for g in GROUPS}
    l1_full = W(A4.arrays(data, c)['L1'][..., gi])
    out['l1_partition_max_abs_error'] = float(np.abs(sum(l1.values()) - l1_full).max())
    for target, f in full.items():
        hv = {h: W(D[target][h]) for h in HYBRIDS}
        groups = {}
        for h in HYBRIDS:
            e = dict(truth_mean=float(hv[h].mean()), truth_sd=float(hv[h].std(ddof=1)),
                     mean_ratio_to_full=float(hv[h].mean() / f.mean()) if f.mean() != 0 else None,
                     spearman_with_full_truth=spearman(hv[h], f))
            if h in GROUPS and target == 'new':
                b, row = block(l1[h], hv[h], data['M'].sum(1))
                e['l1_group_lds_on_group_truth'] = b
            groups[h] = e
            tables['hybrid'].append(dict(target=target, hybrid=h, truth_mean=e['truth_mean'], truth_sd=e['truth_sd'],
                                         spearman_with_full_truth=e['spearman_with_full_truth'],
                                         l1_lds=e.get('l1_group_lds_on_group_truth', {}).get('spearman')))
        resid = f - hv['deleted'] - hv['retained'] - hv['nonmember']
        inter = hv['deleted_nonmember'] - hv['deleted'] - hv['nonmember']
        out['targets'][target] = dict(
            groups=groups,
            nonadditive_residual=dict(mean=float(resid.mean()), sd=float(resid.std(ddof=1)), definition='full - D - R - N'),
            interaction_deleted_nonmember=dict(mean=float(inter.mean()), sd=float(inter.std(ddof=1)),
                                               definition='DN - D - N, retained rows at baseline'))
    return out


def interp_summary(interp, fr, tables):
    if not interp:
        return dict(status='missing')
    g = f"{fr['attack_gamma_star']:g}"
    rows = interp['rows']
    out = dict(status='complete', panel=interp['panel'], attack_gamma=fr['attack_gamma_star'], by_t={})
    for t in interp['panel']['ts']:
        sel = [r for r in rows if r['t'] == t]
        res = {}
        for target, dj, l1 in (('new', 'dJ', 'L1'), ('legacy', 'dJ_legacy', 'L1_legacy')):
            d = np.array([r[dj] for r in sel])
            pred = np.array([t * r[l1][g] for r in sel])
            rel = np.abs(d - pred) / np.maximum(np.abs(d), 1e-300)
            res[target] = dict(units=len(sel), median_true_slope=float(np.median(d / t)), median_l1=float(np.median(pred / t)),
                               median_relative_error=float(np.median(rel)), sign_agreement=float(np.mean(np.sign(d) == np.sign(pred))),
                               spearman_slope_vs_l1=spearman(d / t, pred / t), zero_responses=int((d == 0).sum()))
            tables['interp'].append(dict(t=t, target=target, **{k: v for k, v in res[target].items()}))
        eff = np.array([r['dP_norm_float32'] / r['dP_norm'] for r in sel if r['dP_norm'] > 0])
        res['float32_effective_ratio_median'] = float(np.median(eff)) if len(eff) else None
        out['by_t'][f'{t:g}'] = res
    return out
