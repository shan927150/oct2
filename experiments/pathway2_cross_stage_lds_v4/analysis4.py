"""Collect-side analysis for the cross-stage LDS ladder (numpy only).

Unit of inference = subset (m = 32). Every rung is averaged over (training seed, attack seed)
before the Spearman correlation with the truth, which is averaged the same way.
Class weights omega_c and the qualified class set are read from smoke/frozen.json.
"""
from pathlib import Path
import numpy as np
import runtime as rt
import stats2 as S               # seed32 v2, unchanged
from analysis_stats import spearman  # LDS v1, unchanged

KS = (1, 2, 4, 8)


def load(root, c, allow_partial=False):
    root = Path(root)
    h = rt.sha(root / 'config.json')
    rt.completed(root / 'smoke', h, dict(kind='xsmoke'))
    frozen = rt.read(root / 'smoke/frozen.json')
    attack_base = rt.read(root / 'smoke/attack_baselines.json')['rows']
    smoke = rt.read(root / 'smoke/result.json')
    M = np.asarray(rt.read(root / 'designs.json')['design']['masks'], dtype=float)
    missing, base, truth, scores = [], {}, {}, {}
    for s in c['seeds']:
        d = root / 'base' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='base', seed=s))
            base[s] = rt.read(d / 'result.json')
        else:
            missing.append(f'base/seed{s}')
    for cell in c['cells']:
        d = root / 'truth' / cell['name']
        if (d / 'complete.json').exists():
            rt.completed(d, h, cell)
            truth[(cell['subset'], cell['seed'])] = rt.read(d / 'result.json')
        else:
            missing.append('truth/' + cell['name'])
    for s in c['score_seeds']:
        d = root / 'score' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='xscore', seed=s))
            with np.load(d / 'scores.npz') as z:
                scores[s] = dict(arrays={k: z[k] for k in z.files}, result=rt.read(d / 'result.json'))
        else:
            missing.append(f'score/seed{s}')
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} artefacts incomplete (first: {missing[:5]}). Use resubmit or collect --allow-partial.')
    return dict(frozen=frozen, attack_base=attack_base, smoke=smoke, M=M, base=base, truth=truth, scores=scores,
                missing=missing)


def arrays(data, c):
    """Truth and per-(j, s, k, c) rung arrays over the seeds complete for every subset."""
    m = len(data['M'])
    seeds = [s for s in c['seeds'] if all((j, s) in data['truth'] for j in range(m))]
    K, C, gA = c['attack_seeds'], c['classes'], c['attack_gammas']
    shape = (m, len(seeds), len(K), len(C))
    out = dict(seeds=seeds, D=np.full(shape, np.nan), L2r=np.full(shape, np.nan), L2=np.full(shape, np.nan),
               L1=np.full(shape + (len(gA),), np.nan), J00=np.full(shape[1:], np.nan))
    for j in range(m):
        for si, s in enumerate(seeds):
            rows = {(r['attack_seed'], r['cls']): r for r in data['truth'][(j, s)]['rows']}
            for ki, k in enumerate(K):
                for ci, cls in enumerate(C):
                    r = rows[(k, cls)]
                    out['D'][j, si, ki, ci] = r['J10'] - r['J00']
                    out['L2r'][j, si, ki, ci] = r['J_L2_retrain'] - r['J00']
                    out['L2'][j, si, ki, ci] = r['L2_lin']
                    out['L1'][j, si, ki, ci] = [r['L1'][f'{g:g}'] for g in gA]
                    out['J00'][si, ki, ci] = r['J00']
    return out


def weighted(x, omega, classes):
    """Sum over the class axis (axis 3) with omega_c; unqualified classes get weight 0."""
    w = np.asarray([omega.get(str(cls), 0.) for cls in classes])
    return np.tensordot(x, w, axes=([3], [0])) if x.ndim == 4 else np.einsum('jskcg,c->jskg', x, w)


def l3_predictions(data, c, gamma):
    preds, reasons = [], []
    for s in c['score_seeds']:
        sc = data['scores'].get(s)
        if sc is None:
            return None, f'score seed{s} missing'
        q = next(x for x in sc['result']['gammas'] if x['gamma'] == gamma)
        if not q['qualified']:
            reasons.append(f'seed{s}: ' + '; '.join(q['reasons']))
            continue
        preds.append(sc['arrays'][f'pred_gamma{gamma:g}'])
    if reasons:
        return None, 'unqualified: ' + ' | '.join(reasons)
    return np.stack(preds), None


def paired_difference(p1, p2, truth, draws=2000, seed=20261012):
    rng = np.random.default_rng(seed)
    base = spearman(p1, truth) - spearman(p2, truth)
    vals = []
    for _ in range(draws):
        i = rng.integers(len(truth), size=len(truth))
        a, b = spearman(p1[i], truth[i]), spearman(p2[i], truth[i])
        if a is not None and b is not None:
            vals.append(a - b)
    vals = np.asarray(vals)
    return dict(difference=float(base), ci=np.quantile(vals, [.025, .975]).tolist(),
                fraction_not_better=float((vals <= 0).mean()), draws=len(vals))


def analyze(data, c):
    M = data['M']
    m = len(M)
    nimg = M.sum(1)
    fr = data['frozen']
    omega, qual, gstar = fr['omega'], fr['qualified_classes'], fr['attack_gamma_star']
    ar = arrays(data, c)
    seeds = ar['seeds']
    summary = dict(design=dict(m=m, retained_images=int(nimg[0]), seeds_complete=seeds, attack_seeds=c['attack_seeds'],
                               score_seeds=c['score_seeds']),
                   frozen=fr, missing=data['missing'], critical_spearman=S.critical_spearman(m),
                   smoke_checks=data['smoke'].get('checks'), attack_quality={}, truth={}, ladder={}, ladder_per_class={},
                   differences={}, l3_gamma={}, drift={}, seed_curves={})
    tables = dict(ladder=[], ladder_per_class=[], l3_gamma=[], attack_quality=[], differences=[])
    # ---------------------------------------------------------- attack quality (full baselines, smoke)
    for cls in c['classes']:
        rows = [r for r in data['attack_base'] if r['cls'] == cls]
        aq = dict(qualified=cls in qual, omega=omega.get(str(cls), 0.),
                  auc_mean=float(np.mean([r['metrics']['auc'] for r in rows])),
                  auc_min=float(np.min([r['metrics']['auc'] for r in rows])),
                  balanced_accuracy_min=float(np.min([r['metrics']['balanced_accuracy'] for r in rows])),
                  J00_mean=float(np.mean([r['metrics']['cross_entropy'] for r in rows])),
                  hessian_eig_min_range=[float(np.min([r['hessian_eig_min'] for r in rows])),
                                         float(np.max([r['hessian_eig_min'] for r in rows]))],
                  train_grad_norm_max=float(np.max([r['train_grad_norm'] for r in rows])))
        summary['attack_quality'][str(cls)] = aq
        tables['attack_quality'].append(dict(cls=cls, **{k: v for k, v in aq.items() if not isinstance(v, list)},
                                             eig_min_low=aq['hessian_eig_min_range'][0], eig_min_high=aq['hessian_eig_min_range'][1]))
    if not seeds:
        return summary, tables
    gA = c['attack_gammas']
    D = weighted(ar['D'], omega, c['classes'])            # [m, S, K]
    truth = D.mean((1, 2))
    rungs = {f'L1 (gamma_A={g:g})': weighted(ar['L1'], omega, c['classes'])[..., gi] for gi, g in enumerate(gA)}
    rungs['L2-retrain'] = weighted(ar['L2r'], omega, c['classes'])
    rungs['L2-lin'] = weighted(ar['L2'], omega, c['classes'])
    preds = {k: v.mean((1, 2)) for k, v in rungs.items()}
    l3_seed = {}
    for g in c['gammas']:
        P, why = l3_predictions(data, c, g)
        entry = dict(status='solved' if P is not None else why)
        rows = []
        for s in c['score_seeds']:
            sc = data['scores'].get(s)
            if sc is not None:
                q = next(x for x in sc['result']['gammas'] if x['gamma'] == g)
                rows.append(dict(seed=s, qualified=q['qualified'], true_rel_residual=q['true_rel_residual'],
                                 spearman_if_vs_proxy_images=q['spearman_if_vs_proxy_images'],
                                 lanczos_steps=sc['result']['lanczos']['steps']))
        entry['by_seed'] = rows
        if P is not None:
            preds[f'L3 (gamma={g:g})'] = P.mean(0)
            l3_seed[g] = P
        summary['l3_gamma'][f'{g:g}'] = entry
    qualified_g = [g for g in c['gammas'] if g in l3_seed]
    gamma_star = min(qualified_g) if qualified_g else None
    if all(s in data['scores'] for s in c['score_seeds']):
        preds['proxy'] = np.mean([data['scores'][s]['arrays']['pred_proxy'] for s in c['score_seeds']], 0)
    # ---------------------------------------------------------------- truth stability
    Yseed = D.mean(2)                                        # [m, S]
    Yatt = D.mean(1)                                         # [m, K]
    summary['truth'] = dict(mean=float(truth.mean()), sd_across_subsets=float(truth.std(ddof=1)),
                            positive_subsets=int((truth > 0).sum()),
                            J00_mean=float(weighted(ar['J00'][None], omega, c['classes']).mean()),
                            seed_reliability=S.reliability_curve(Yseed, [k for k in KS if k <= len(seeds)]),
                            attack_seed_pairwise=[dict(a=c['attack_seeds'][a], b=c['attack_seeds'][b], spearman=spearman(Yatt[:, a], Yatt[:, b]))
                                                  for a in range(Yatt.shape[1]) for b in range(a + 1, Yatt.shape[1])],
                            variance_seed=S.variance_components(Yseed), variance_attack=S.variance_components(Yatt))
    # ---------------------------------------------------------------- ladder
    order = [f'L1 (gamma_A={gstar:g})', 'L2-retrain', 'L2-lin'] + [f'L3 (gamma={g:g})' for g in c['gammas']] + ['proxy'] \
        + [f'L1 (gamma_A={g:g})' for g in gA if g != gstar]
    for name in order:
        if name not in preds:
            continue
        b = S.lds_block(preds[name], truth, nimg)
        summary['ladder'][name] = b
        ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
        tables['ladder'].append(dict(rung=name, spearman=b['spearman'], ci_low=ci[0], ci_high=ci[1],
                                     perm_p=b.get('permutation_p_one_sided'), slope=b.get('slope_truth_on_prediction'),
                                     relative_error=b.get('relative_error'), primary=name in (
                                         f'L1 (gamma_A={gstar:g})', 'L2-retrain', 'L2-lin', 'proxy') or
                                     (gamma_star is not None and name == f'L3 (gamma={gamma_star:g})')))
    summary['ladder_gamma_star'] = gamma_star
    # paired differences between adjacent rungs
    l3n = f'L3 (gamma={gamma_star:g})' if gamma_star is not None else None
    pairs = [(f'L1 (gamma_A={gstar:g})', 'L2-retrain'), ('L2-retrain', 'L2-lin'), ('L2-lin', l3n), (l3n, 'proxy'),
             (l3n, 'L3 (gamma=1)')]
    for a, b in pairs:
        if a in preds and b in preds and a != b:
            d = paired_difference(preds[a], preds[b], truth)
            summary['differences'][f'{a} - {b}'] = d
            tables['differences'].append(dict(pair=f'{a} - {b}', difference=d['difference'], ci_low=d['ci'][0],
                                              ci_high=d['ci'][1], fraction_not_better=d['fraction_not_better']))
    # per class (no omega), truth-using rungs only
    for ci_, cls in enumerate(c['classes']):
        t = ar['D'][..., ci_].mean((1, 2))
        block = {}
        for name, x in (('L1', ar['L1'][..., ci_, gA.index(gstar)]), ('L2-retrain', ar['L2r'][..., ci_]), ('L2-lin', ar['L2'][..., ci_])):
            block[name] = S.lds_block(x.mean((1, 2)), t, nimg)
            tables['ladder_per_class'].append(dict(cls=cls, qualified=cls in qual, rung=name, spearman=block[name]['spearman'],
                                                   ci_low=(block[name].get('bootstrap_ci_conditional_on_seed_panel') or [None])[0],
                                                   ci_high=(block[name].get('bootstrap_ci_conditional_on_seed_panel') or [None, None])[1]))
        summary['ladder_per_class'][str(cls)] = dict(qualified=cls in qual, truth_mean=float(t.mean()), rungs=block)
    # seed curves: LDS of per-seed-predicted rungs vs k training seeds
    fns = {'L1': lambda idx, R=rungs[f'L1 (gamma_A={gstar:g})'].mean(2): R[:, idx].mean(1),
           'L2-retrain': lambda idx, R=rungs['L2-retrain'].mean(2): R[:, idx].mean(1),
           'L2-lin': lambda idx, R=rungs['L2-lin'].mean(2): R[:, idx].mean(1)}
    if gamma_star is not None and list(c['score_seeds']) == list(seeds):
        fns[f'L3 (gamma={gamma_star:g})'] = lambda idx, R=l3_seed[gamma_star].T: R[:, idx].mean(1)
    summary['seed_curves'] = S.lds_curve(Yseed, fns, sorted({k for k in KS if k <= len(seeds)} | {len(seeds)}))
    # ------------------------------------------------------------ L3 gamma table and drift
    for g in c['gammas']:
        e = summary['l3_gamma'][f'{g:g}']
        lds = summary['ladder'].get(f'L3 (gamma={g:g})', {}).get('spearman')
        sp = [r['spearman_if_vs_proxy_images'] for r in e['by_seed'] if r.get('spearman_if_vs_proxy_images') is not None]
        tables['l3_gamma'].append(dict(gamma=g, status=e['status'], lds=lds,
                                       mean_spearman_if_vs_proxy=float(np.mean(sp)) if sp else None,
                                       max_true_rel_residual=max((r['true_rel_residual'] or np.inf) for r in e['by_seed']) if e['by_seed'] else None,
                                       max_lanczos_steps=max(r['lanczos_steps'] for r in e['by_seed']) if e['by_seed'] else None))
    cells = list(data['truth'].values())
    for g in ('deleted', 'retained', 'nonmember'):
        summary['drift'][g] = {k: float(np.mean([x['drift'][g][k] for x in cells])) for k in cells[0]['drift'][g]}
    summary['drift']['jvp_row_clip_rate_mean'] = float(np.mean([x['jvp']['row_clip_rate'] for x in cells]))
    summary['drift']['dtheta_norm_mean'] = float(np.mean([x['dtheta_norm'] for x in cells]))
    summary['prespecified'] = dict(primary='Delta_value of image-weighted attack CE over qualified classes',
                                   attack_gamma_star=gstar, stage1_gamma_star_rule='smallest gamma qualified for every score seed',
                                   rule='No damping value is chosen by LDS.')
    return summary, tables
