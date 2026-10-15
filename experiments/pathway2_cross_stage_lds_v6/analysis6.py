"""Collect-side analysis for v6 (numpy only). Unit of inference = subset (m = 32).

Truths (primary classes, omega-weighted, averaged over training seeds and attack seeds):
    T_all (6 attack seeds, primary), T_old (v5 r1 seeds, equals the v5 r1 truth), T_new (new seeds only).
Rungs: L1 / L1_member (gamma_A*), L2-lin / L2-lin_member, L3_full / L3_member per Stage-1 gamma, proxies.
Prespecified: P1 L3_full(0.03) and P2 L3_member(0.03) vs T_all, one-sided permutation p at alpha 0.025;
P3 the paired bootstrap difference LDS(L3_member) - LDS(L3_full) on T_all.
"""
from pathlib import Path
import numpy as np
import runtime as rt
import analysis4 as A4           # v4, unchanged (weighted)
import stats2 as S               # seed32 v2, unchanged
from analysis_stats import spearman  # LDS v1, unchanged

KS = (1, 2, 4, 8)
ALPHA = 0.025


def load(root, c, allow_partial=False):
    root = Path(root)
    h = rt.sha(root / 'config.json')
    rt.completed(root / 'smoke', h, dict(kind='xsmoke'))
    data = dict(frozen=rt.read(root / 'smoke/frozen.json'), gate=rt.read(root / 'smoke/gate.json'),
                M=np.asarray(rt.read(root / 'designs.json')['design']['masks'], dtype=float), truth={}, scores={},
                base={}, missing=[])
    for s in c['seeds']:
        d = root / 'base' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='base6', seed=s))
            data['base'][s] = rt.read(d / 'result.json')
        else:
            data['missing'].append(f'base/seed{s}')
    for cell in c['cells']:
        d = root / 'truth' / cell['name']
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(cell, kind='truth6'))
            data['truth'][(cell['subset'], cell['seed'])] = rt.read(d / 'result.json')
        else:
            data['missing'].append('truth/' + cell['name'])
    for s in c['score_seeds']:
        d = root / 'score' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='score6', seed=s))
            with np.load(d / 'scores.npz') as z:
                data['scores'][s] = dict(arrays={k: z[k] for k in z.files}, result=rt.read(d / 'result.json'))
        else:
            data['missing'].append(f'score/seed{s}')
    data['v5_scores'] = {}
    v5 = Path(c['v5_out'])
    for s in c['score_seeds']:
        p = v5 / 'score' / f'seed{s}' / 'scores.npz'
        if str(p) in c['input_sha256']:
            rt.verify_hashes({str(p): c['input_sha256'][str(p)]})
            with np.load(p) as z:
                data['v5_scores'][s] = {k: z[k] for k in z.files if k.startswith('pred_')}
    if data['missing'] and not allow_partial:
        raise ValueError(f"{len(data['missing'])} artefacts incomplete (first: {data['missing'][:5]}). Use --allow-partial.")
    return data


def arrays(data, c):
    m = len(data['M'])
    seeds = [s for s in c['seeds'] if all((j, s) in data['truth'] for j in range(m))]
    K, C, gA = c['attack_seeds'], c['classes'], c['attack_gammas']
    shape = (m, len(seeds), len(K), len(C))
    out = dict(seeds=seeds, D=np.full(shape, np.nan), L1=np.full(shape + (len(gA),), np.nan),
               L1m=np.full(shape + (len(gA),), np.nan), L2=np.full(shape, np.nan), L2m=np.full(shape, np.nan),
               checks=dict(L1=0., L2=0., j10_replay=0.))
    for j in range(m):
        for si, s in enumerate(seeds):
            cell = data['truth'][(j, s)]
            out['checks']['j10_replay'] = max(out['checks']['j10_replay'], cell['j10_replay_abs'])
            for r in cell['rows']:
                idx = (j, si, K.index(r['attack_seed']), C.index(r['cls']))
                out['D'][idx] = r['J10'] - r['J00']
                out['L1'][idx] = [r['L1'][f'{g:g}'] for g in gA]
                out['L1m'][idx] = [r['L1_member'][f'{g:g}'] for g in gA]
                out['L2'][idx] = r['L2_lin']
                out['L2m'][idx] = r['L2_lin_member']
                out['checks']['L1'] = max(out['checks']['L1'], r.get('L1_check_abs', 0.))
                out['checks']['L2'] = max(out['checks']['L2'], r.get('L2_lin_check_abs', 0.))
    return out


def l3(data, c, variant, gamma, key='pred_gamma'):
    preds, why = [], []
    for s in c['score_seeds']:
        sc = data['scores'].get(s)
        if sc is None:
            return None, f'score seed{s} missing'
        q = next(x for x in sc['result']['variants'][variant]['gammas'] if x['gamma'] == gamma)
        if not q['qualified']:
            why.append(f'seed{s}: ' + '; '.join(q['reasons']))
            continue
        preds.append(sc['arrays'][f'{variant}_{key}{gamma:g}'])
    if why:
        return None, 'unqualified: ' + ' | '.join(why)
    return np.stack(preds), None


def paired(p1, p2, truth, draws=2000, seed=20261015):
    a, b = spearman(p1, truth), spearman(p2, truth)
    if a is None or b is None:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        i = rng.integers(len(truth), size=len(truth))
        x, y = spearman(p1[i], truth[i]), spearman(p2[i], truth[i])
        if x is not None and y is not None:
            vals.append(x - y)
    vals = np.asarray(vals)
    return dict(lds_a=float(a), lds_b=float(b), difference=float(a - b), ci=np.quantile(vals, [.025, .975]).tolist(),
                fraction_not_better=float((vals <= 0).mean()), draws=len(vals))


def analyze(data, c):
    fr = data['frozen']
    omega, gstar, C = fr['omega'], fr['attack_gamma_star'], c['classes']
    K, old, new = c['attack_seeds'], c['old_attack_seeds'], c['new_attack_seeds']
    M = data['M']
    m, nimg = len(M), M.sum(1)
    ar = arrays(data, c)
    summary = dict(design=dict(m=m, seeds_complete=ar['seeds'], attack_seeds=K, old=old, new=new, score_seeds=c['score_seeds']),
                   frozen=fr, gate=data['gate'], missing=data['missing'], critical_spearman=S.critical_spearman(m),
                   critical_spearman_alpha_0_025=S.critical_spearman(m, alpha=ALPHA), replay=ar['checks'],
                   truths={}, ladder={}, primary={}, secondary={}, attack_seed_reliability={})
    tables = dict(ladder=[], primary=[], secondary=[], attack_pairs=[])
    if not ar['seeds']:
        return summary, tables
    io = [K.index(k) for k in old]
    inw = [K.index(k) for k in new]
    W = lambda x: A4.weighted(x, omega, C)
    D = W(ar['D'])                                             # [m, S, K]
    T = dict(all=D.mean((1, 2)), old=D[:, :, io].mean((1, 2)), new=D[:, :, inw].mean((1, 2)))
    for name, t in T.items():
        sd = float(t.std(ddof=1))
        summary['truths'][name] = dict(mean=float(t.mean()), sd_across_subsets=sd)
    summary['truths']['spearman_old_vs_new'] = spearman(T['old'], T['new'])
    summary['truths']['seed_reliability_all'] = S.reliability_curve(D.mean(2), [k for k in KS if k <= len(ar['seeds'])])
    Yatt = D.mean(1)                                           # [m, K]
    pairs = [dict(a=K[a], b=K[b], spearman=spearman(Yatt[:, a], Yatt[:, b]))
             for a in range(len(K)) for b in range(a + 1, len(K))]
    tables['attack_pairs'] = pairs
    summary['attack_seed_reliability'] = dict(
        pairwise_mean=float(np.mean([p['spearman'] for p in pairs])),
        pairwise_old_mean=float(np.mean([p['spearman'] for p in pairs if p['a'] in old and p['b'] in old])),
        split_half_old_vs_new=summary['truths']['spearman_old_vs_new'],
        variance=S.variance_components(Yatt))
    # ---------------------------------------------------------------- rungs
    gi = c['attack_gammas'].index(gstar)
    preds = {'L1 (full)': W(ar['L1'][..., gi]), 'L1 (member)': W(ar['L1m'][..., gi]),
             'L2-lin (full)': W(ar['L2']), 'L2-lin (member)': W(ar['L2m'])}
    pred_all = {k: v.mean((1, 2)) for k, v in preds.items()}
    pred_by = {name: {'old': v[:, :, io].mean((1, 2)), 'new': v[:, :, inw].mean((1, 2))} for name, v in preds.items()}
    l3_seed, status = {}, {}
    for v in ('full', 'member'):
        for g in c['gammas']:
            P, why = l3(data, c, v, g)
            name = f'L3 {v} (gamma={g:g})'
            status[name] = 'solved' if P is not None else why
            if P is not None:
                pred_all[name] = P.mean(0)
                l3_seed[name] = P
        if all(s in data['scores'] for s in c['score_seeds']):
            pred_all[f'proxy ({v})'] = np.mean([data['scores'][s]['arrays'][f'{v}_pred_proxy'] for s in c['score_seeds']], 0)
    summary['l3_status'] = status
    for name, p in pred_all.items():
        b = S.lds_block(p, T['all'], nimg)
        summary['ladder'][name] = b
        ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
        row = dict(rung=name, spearman=b['spearman'], ci_low=ci[0], ci_high=ci[1], perm_p=b.get('permutation_p_one_sided'))
        for part in ('old', 'new'):   # L1/L2 use the matching attack seeds; L3 / proxy use the 6-seed hbar
            pp = pred_by[name][part] if name in pred_by else p
            row[f'lds_truth_{part}'] = spearman(pp, T[part])
            b[f'lds_truth_{part}'] = row[f'lds_truth_{part}']
        tables['ladder'].append(row)
    # ---------------------------------------------------------------- primary
    g0 = c['primary_gamma']
    for key, name in (('P1', f'L3 full (gamma={g0:g})'), ('P2', f'L3 member (gamma={g0:g})')):
        b = summary['ladder'].get(name)
        if b is None:
            summary['primary'][key] = dict(rung=name, status=status.get(name, 'missing'))
            continue
        p = b.get('permutation_p_one_sided')
        e = dict(rung=name, lds=b['spearman'], ci=b.get('bootstrap_ci_conditional_on_seed_panel'), perm_p_one_sided=p,
                 alpha=ALPHA, passed=bool(p is not None and p < ALPHA))
        summary['primary'][key] = e
        tables['primary'].append(dict(test=key, rung=name, lds=e['lds'], perm_p=p, passed=e['passed']))
    a, b = f'L3 member (gamma={g0:g})', f'L3 full (gamma={g0:g})'
    if a in pred_all and b in pred_all:
        d = paired(pred_all[a], pred_all[b], T['all'])
        summary['primary']['P3'] = dict(rung=f'{a} - {b}', **d)
        tables['primary'].append(dict(test='P3', rung=f'{a} - {b}', lds=d['difference'], perm_p=None, passed=None,
                                      ci_low=d['ci'][0], ci_high=d['ci'][1]))
    # ---------------------------------------------------------------- secondary
    sec = {}
    for x, y in (('L1 (member)', 'L1 (full)'), ('L2-lin (member)', 'L2-lin (full)'),
                 ('proxy (member)', 'proxy (full)')):
        if x in pred_all and y in pred_all:
            sec[f'{x} - {y}'] = paired(pred_all[x], pred_all[y], T['all'])
    v5 = data.get('v5_scores') or {}
    if b in pred_all and v5 and all(s in v5 for s in c['score_seeds']):
        key = f'pred_gamma{g0:g}'
        if all(key in v5[s] for s in c['score_seeds']):
            p_v5 = np.mean([v5[s][key] for s in c['score_seeds']], 0)
            sec['L3 full: 6-seed hbar - v5 r1 3-seed hbar (on T_all)'] = paired(pred_all[b], p_v5, T['all'])
            sec['v5 r1 L3 (3-seed hbar) on T_old'] = dict(lds_a=spearman(p_v5, T['old']))
    for name, d in sec.items():
        if d and 'difference' in d:
            tables['secondary'].append(dict(comparison=name, lds_a=d['lds_a'], lds_b=d['lds_b'], difference=d['difference'],
                                            ci_low=d['ci'][0], ci_high=d['ci'][1], fraction_not_better=d['fraction_not_better']))
    summary['secondary'] = sec
    # L3 learning curve over training seeds
    fns = {n: (lambda idx, R=l3_seed[n].T: R[:, idx].mean(1)) for n in (a, b) if n in l3_seed}
    if fns and list(c['score_seeds']) == list(ar['seeds']):
        summary['seed_curves'] = S.lds_curve(D.mean(2), fns, sorted({k for k in KS if k <= len(ar['seeds'])} | {len(ar['seeds'])}))
    summary['score_diagnostics'] = {s: dict(cos_hbar_full_member=sc['result']['cos_hbar_full_member'],
                                            hbar_replay_vs_v5=sc['result']['hbar_replay_vs_v5_max_abs'])
                                    for s, sc in data['scores'].items()}
    return summary, tables
