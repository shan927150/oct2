"""Collect-side analysis for tasks 1-5 (numpy only, runs on the login node).

Everything is recomputed from stored probabilities; result.json values are only
used as an integrity cross-check. Unit of inference = subset.
"""
from pathlib import Path
import numpy as np
import runtime as rt
import targets as T
import designs as DS
import stats2 as S
from analysis_stats import spearman, evaluate, crossfit_ridge

ENDPOINTS = ('raw', 'ema')
KS = (1, 2, 3, 4, 8, 16, 32)  # k=3 explicitly links to the original pilot


def _values(path, labels_test, weights, units, N):
    out = {}
    with np.load(path) as z:
        for ep in ENDPOINTS:
            out[ep] = T.all_values(z[ep][N:], labels_test, weights, units,
                                  z['log_' + ep][N:] if 'log_' + ep in z else None)
    return out


def load(root, c, allow_partial=False):
    root = Path(root)
    h = rt.sha(root / 'config.json')
    frozen = rt.read(root / 'designs.json')
    base, designs = frozen['base'], frozen['designs']
    N = sum(len(p['rows']) for p in base['train_patients'])
    weights = np.asarray(base['query_weights'])
    qids, units, coef = T.query_units(base)
    labels_test = None
    cells, missing = {}, []
    for cell in c['cells']:
        d = root / 'cells' / cell['name']
        if not (d / 'complete.json').exists():
            missing.append(cell['name'])
            continue
        rt.completed(d, h, cell)
        with np.load(d / 'interface.npz') as z:
            expected_rows = [r for p in base['train_patients'] for r in p['rows']] + base['query_rows']
            # Train rows follow make_design's original train order, not patient concatenation.
            if set(z['raw_index'][:N].tolist()) != set(expected_rows[:N]) or not np.array_equal(z['raw_index'][N:], base['query_rows']):
                raise ValueError('Interface row identities differ: ' + cell['name'])
            if labels_test is None:
                labels_test = z['classes'][N:].astype(np.int64)
            elif not np.array_equal(labels_test, z['classes'][N:]):
                raise ValueError('Query labels differ between cells')
        vals = _values(d / 'interface.npz', labels_test, weights, units, N)
        stored = rt.read(d / 'result.json')
        expected_n = N if cell['kind'] == 'full' else int(np.asarray(designs[cell['design']]['masks'][cell['subset']]) @
                                                        np.asarray([len(p['rows']) for p in base['train_patients']]))
        if stored['cell'] != cell or stored['n_train_images'] != expected_n:
            raise ValueError('Cell identity or retained image count mismatch: ' + cell['name'])
        if 'epochs' in c and stored.get('optimizer_steps') != c['epochs'] * int(np.ceil(expected_n / c['recipe']['batch'])):
            raise ValueError('Optimizer step count mismatch: ' + cell['name'])
        for ep in ENDPOINTS:
            for t in T.TARGETS:
                a, b = vals[ep][t]['Y'], stored['Y'][ep][t]
                if abs(a - b) > 1e-10 * max(1., abs(b)):
                    raise ValueError(f'Stored Y differs from recomputed Y: {cell["name"]} {ep} {t}')
        cells[cell['name']] = dict(cell=cell, values=vals, n_images=stored['n_train_images'])
    missing_artifacts = ['cells/' + n for n in missing]
    train_ids = np.asarray([p['patient'] for p in base['train_patients']])
    proxies = {}
    for s in c['seeds']:
        d = root / 'proxy' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='proxy', seed=s))
            with np.load(d / 'proxy.npz') as z:
                if not np.array_equal(z['train_patient_ids'], train_ids) or not np.array_equal(z['query_patient_ids'], qids):
                    raise ValueError('Proxy patient identities differ')
                proxies[s] = {k: z[k] for k in z.files}
        else:
            missing_artifacts.append(f'proxy/seed{s}')
    scores = {}
    for s in c['score_seeds']:
        for t in c['score_targets']:
            d = root / 'score' / f'seed{s}_{t}'
            if (d / 'complete.json').exists():
                rt.completed(d, h, dict(kind='score', seed=s, target=t))
                with np.load(d / 'scores.npz') as z:
                    if not np.array_equal(z['train_patient_ids'], train_ids):
                        raise ValueError('Score patient identities differ')
                    scores[(s, t)] = dict(arrays={k: z[k] for k in z.files}, result=rt.read(d / 'result.json'))
            else:
                missing_artifacts.append(f'score/seed{s}_{t}')
    if missing_artifacts and not allow_partial:
        raise ValueError(f'{len(missing_artifacts)} artifacts incomplete (first: {missing_artifacts[:5]}). Use resubmit or collect --allow-partial.')
    return dict(base=base, designs=designs, N=N, units=units, qids=qids, coef=coef,
                cells=cells, missing=missing, missing_artifacts=missing_artifacts, proxies=proxies, scores=scores)


def truth(data, c, design, restrict_seeds=None):
    """Seed-level truth for one design using seeds complete for every subset."""
    m = len(data['designs'][design]['masks'])
    seeds = []
    for s in c['design_seeds'][design]:
        if restrict_seeds is not None and s not in restrict_seeds:
            continue
        names = [f'{design}_s{j:03d}_seed{s}' for j in range(m)] + [f'full_seed{s}']
        if all(n in data['cells'] for n in names):
            seeds.append(s)
    out = dict(seeds=seeds)
    if not seeds:
        return out
    for ep in ENDPOINTS:
        out[ep] = {}
        for t in T.TARGETS:
            Y = np.array([[data['cells'][f'{design}_s{j:03d}_seed{s}']['values'][ep][t]['Y'] for s in seeds] for j in range(m)])
            Yq = np.array([[data['cells'][f'{design}_s{j:03d}_seed{s}']['values'][ep][t]['Yq'] for s in seeds] for j in range(m)])
            Y0 = np.array([data['cells'][f'full_seed{s}']['values'][ep][t]['Y'] for s in seeds])
            Yq0 = np.array([data['cells'][f'full_seed{s}']['values'][ep][t]['Yq'] for s in seeds])
            out[ep][t] = dict(Y=Y, D=Y - Y0[None, :], Yq=Yq, Dq=Yq - Yq0[None, :, :])
    out['n_images'] = np.array([data['cells'][f'{design}_s{j:03d}_seed{seeds[0]}']['n_images'] for j in range(m)], float)
    return out


def if_scores(data, c, target, gamma):
    """Seed-mean IF score at one gamma, or (None, reason) if any score seed is
    missing/unqualified. Never silently drops a seed."""
    taus, reasons = [], []
    for s in c['score_seeds']:
        sc = data['scores'].get((s, target))
        if sc is None:
            return None, f'score seed{s}_{target} missing'
        consistency = sc['result'].get('lds_v1_consistency')
        if consistency is not None and not consistency.get('passed'):
            return None, f'seed{s}: CE gamma1 replay mismatch (entire score task withheld)'
        q = next(x for x in sc['result']['gammas'] if x['gamma'] == gamma)
        if not q['qualified']:
            reasons.append(f'seed{s}: ' + '; '.join(q['reasons']))
            continue
        tau = sc['arrays'][f'tau_gamma{gamma:g}']
        if not np.isfinite(tau).all():
            raise ValueError('Qualified tau is nonfinite')
        taus.append(tau)
    if reasons:
        return None, 'unqualified: ' + ' | '.join(reasons)
    if not taus:
        return None, 'no score seeds'
    return np.mean(taus, 0), None


def proxy_mean(data, seeds, target, key='macro'):
    if not all(s in data['proxies'] for s in seeds):
        return None
    return np.mean([data['proxies'][s][f'proxy_{key}_{target}'] for s in seeds], 0)


def analyze(data, c):
    base = data['base']
    images = np.asarray([len(p['rows']) for p in base['train_patients']], float)
    summary = dict(missing_cells=len(data['missing']), missing_artifacts=data.get('missing_artifacts', []), proxies=sorted(data['proxies']),
                   score_tasks=sorted(f'seed{s}_{t}' for s, t in data['scores']),
                   designs={}, task1={}, task2_3={}, task4={}, task5={}, critical_spearman={})
    tables = dict(lds=[], seed_curve=[], gamma=[], per_query=[], same_seed_designs=[])
    truths = {}
    for name, d in data['designs'].items():
        tr = truth(data, c, name)
        truths[name] = tr
        m = len(d['masks'])
        summary['critical_spearman'][str(m)] = S.critical_spearman(m)
        summary['designs'][name] = dict(kind=d['kind'], m=m, seeds_complete=tr['seeds'],
                                        n_images_range=d['n_images_range'],
                                        steps_per_epoch_range=[min(d['steps_per_epoch']), max(d['steps_per_epoch'])],
                                        acceptance_rate=d.get('diagnostics', {}).get('acceptance_rate'))
        if not tr['seeds']:
            continue
        masks = np.asarray(d['masks'], float)
        nimg = tr['n_images']
        comp, _ = DS.composition_matrix(d['masks'], base['train_patients'])
        summary['task1'][name], summary['task2_3'][name], summary['task5'][name] = {}, {}, {}
        for ep in ENDPOINTS:
            summary['task1'][name][ep], summary['task2_3'][name][ep], summary['task5'][name][ep] = {}, {}, {}
            for t in T.TARGETS:
                Y, D = tr[ep][t]['Y'], tr[ep][t]['D']
                Dbar = D.mean(1)
                seeds = tr['seeds']
                # ---------------- task 2/3: deleter LDS + size controls
                preds, equivalence = {}, None
                tau1, why = if_scores(data, c, t, 1.) if t in c['score_targets'] else (None, 'target not scored')
                if tau1 is not None:
                    preds['IF_gamma1_deleter'] = S.deleter_prediction(masks, tau1)
                    preds['IF_gamma1_matched'] = S.matched_prediction(masks, tau1, images)
                    retained = S.retained_sum(masks, tau1)
                    equivalence = dict(
                        deleter_spearman=spearman(preds['IF_gamma1_deleter'], Dbar),
                        retained_sum_spearman=spearman(retained, Dbar),
                        max_abs_difference_deleter_minus_retained_plus_total=float(np.max(np.abs(
                            preds['IF_gamma1_deleter'] - (retained - tau1.sum())))),
                        sign_error_spearman_if_minus_dropped=spearman(-preds['IF_gamma1_deleter'], Dbar))
                pm = proxy_mean(data, seeds, t)
                if pm is not None and ep == 'raw':
                    preds['proxy_deleter'] = S.deleter_prediction(masks, pm)
                pms = proxy_mean(data, c['score_seeds'], t)
                if pms is not None and ep == 'raw':
                    preds['proxy_deleter_score_seeds'] = S.deleter_prediction(masks, pms)
                preds['size_baseline'] = S.size_baseline(nimg)
                ridge = crossfit_ridge(comp, Y.mean(1))
                if ridge is not None:
                    preds['composition_ridge_crossfit'] = ridge
                block = {}
                for pname, p in preds.items():
                    block[pname] = S.lds_block(p, Dbar, nimg, fitted_on_truth=pname.startswith('composition_'))
                    tables['lds'].append(dict(design=name, endpoint=ep, target=t, predictor=pname, seeds=len(seeds),
                        spearman=block[pname]['spearman'], ci_low=block[pname].get('bootstrap_ci_conditional_on_seed_panel', [None, None])[0],
                        ci_high=block[pname].get('bootstrap_ci_conditional_on_seed_panel', [None, None])[1],
                        perm_p=block[pname].get('permutation_p_one_sided'),
                        residual_spearman_given_images=block[pname]['residual_spearman_given_images'],
                        partial_rank_spearman_given_images=block[pname]['partial_rank_spearman_given_images']))
                summary['task2_3'][name][ep][t] = dict(seeds=seeds, predictors=block, deleter_equivalence=equivalence,
                    if_status=why, truth_n_images_spearman=spearman(nimg, Dbar),
                    note='Predictions are deleter sums of -tau (alpha=1). Composition ridge is cross-fitted, not an upper bound.')
                # ---------------- task 1: seed curves
                ks = sorted({k for k in KS if k <= len(seeds)} | {len(seeds)})
                curve = dict(se=S.se_curve(Y, ks), reliability=S.reliability_curve(Y, ks))
                fns = {}
                if tau1 is not None:
                    fixed_if = S.deleter_prediction(masks, tau1)
                    fns['IF_gamma1 (score seeds, fixed)'] = lambda s, v=fixed_if: v
                if ep == 'raw' and all(x in data['proxies'] for x in seeds):
                    P = np.stack([data['proxies'][x][f'proxy_macro_{t}'] for x in seeds])
                    fns['proxy (same k seeds)'] = lambda s, P=P: S.deleter_prediction(masks, P[s].mean(0))
                fns['size baseline'] = lambda s, v=S.size_baseline(nimg): v
                curve['lds'] = S.lds_curve(D, fns, ks)
                summary['task1'][name][ep][t] = curve
                for row in curve['lds']:
                    tables['seed_curve'].append(dict(design=name, endpoint=ep, target=t, kind='lds', **row))
                for row in curve['reliability']['rows']:
                    tables['seed_curve'].append(dict(design=name, endpoint=ep, target=t, kind='reliability', **row))
                for row in curve['se']:
                    tables['seed_curve'].append(dict(design=name, endpoint=ep, target=t, kind='se', **row))
                # ---------------- task 5: per-query LDS (proxy preview, raw only)
                if ep == 'raw' and all(x in data['proxies'] for x in seeds):
                    Pq = np.mean([data['proxies'][x][f'proxy_query_{t}'] for x in seeds], 0)  # [Q, patients]
                    pred = -(1. - masks) @ Pq.T  # [m, Q]
                    Dq = tr[ep][t]['Dq'].mean(1)  # [m, Q]
                    res = S.per_query_lds(pred, Dq)
                    size = S.per_query_lds(np.repeat(S.size_baseline(nimg)[:, None], Dq.shape[1], 1), Dq)
                    per = res.pop('per_query')
                    size.pop('per_query')
                    summary['task5'][name][ep][t] = dict(proxy=res, size_baseline=size, per_query=per,
                        query_patient_ids=data['qids'].tolist(),
                        note='Gradient-inner-product proxy preview; no inverse Hessian per query.')
                    tables['per_query'].append(dict(design=name, target=t, predictor='proxy', **{k: v for k, v in res.items()}))
                    tables['per_query'].append(dict(design=name, target=t, predictor='size_baseline', **size))
    # ---------------- task 4: gamma sweep
    for t in c['score_targets']:
        summary['task4'][t] = {}
        for g in c['gammas']:
            row = dict(gamma=g, by_seed=[])
            for s in c['score_seeds']:
                sc = data['scores'].get((s, t))
                if sc is None:
                    row['by_seed'].append(dict(seed=s, status='missing'))
                    continue
                q = next(x for x in sc['result']['gammas'] if x['gamma'] == g)
                row['by_seed'].append(dict(seed=s, qualified=q['qualified'], reasons=q['reasons'],
                                           true_rel_residual=q['true_rel_residual'],
                                           damped_krylov_ritz_min=q['damped_krylov_ritz_min'],
                                           spearman_if_vs_proxy=q['spearman_if_vs_proxy_148_patients'],
                                           lanczos_steps=sc['result']['lanczos']['steps']))
            tau, why = if_scores(data, c, t, g)
            row['status'] = 'solved' if tau is not None else why
            sp = [x.get('spearman_if_vs_proxy') for x in row['by_seed'] if x.get('spearman_if_vs_proxy') is not None]
            row['mean_spearman_if_vs_proxy'] = float(np.mean(sp)) if sp and tau is not None else None
            row['lds'] = {}
            if tau is not None:
                for name, d in data['designs'].items():
                    tr = truths[name]
                    if not tr['seeds']:
                        continue
                    masks = np.asarray(d['masks'], float)
                    row['lds'][name] = S.lds_block(S.deleter_prediction(masks, tau), tr['raw'][t]['D'].mean(1), tr['n_images'])
            summary['task4'][t][f'{g:g}'] = row
            tables['gamma'].append(dict(target=t, gamma=g, status=row['status'],
                                        mean_spearman_if_vs_proxy=row['mean_spearman_if_vs_proxy'],
                                        **{f'lds_{k}': v['spearman'] for k, v in row['lds'].items()}))
    # Compare distributions using exactly the same seed identities, never pilot32 vs eq8.
    common = sorted(set.intersection(*(set(tr['seeds']) for tr in truths.values()))) if truths else []
    summary['same_seed_design_comparison'] = dict(seeds=common, designs={})
    if common:
        for name, d in data['designs'].items():
            tr = truth(data, c, name, common)
            block = {}
            for t in T.TARGETS:
                preds = {'size_baseline': S.size_baseline(tr['n_images'])}
                tau, _ = if_scores(data, c, t, 1.) if t in c['score_targets'] else (None, None)
                if tau is not None:
                    preds['IF_gamma1_deleter'] = S.deleter_prediction(d['masks'], tau)
                    preds['IF_gamma1_matched'] = S.matched_prediction(d['masks'], tau, images)
                pm = proxy_mean(data, common, t)
                if pm is not None:
                    preds['proxy_deleter'] = S.deleter_prediction(d['masks'], pm)
                comp, _ = DS.composition_matrix(d['masks'], base['train_patients'])
                ridge = crossfit_ridge(comp, tr['raw'][t]['Y'].mean(1))
                if ridge is not None:
                    preds['composition_ridge_crossfit'] = ridge
                block[t] = {p: S.lds_block(v, tr['raw'][t]['D'].mean(1), tr['n_images'], fitted_on_truth=p.startswith('composition_')) for p, v in preds.items()}
                for p, b in block[t].items():
                    tables['same_seed_designs'].append(dict(design=name, endpoint='raw', target=t, predictor=p,
                                                           seeds=common, **b))
            summary['same_seed_design_comparison']['designs'][name] = block
    summary['prespecified'] = dict(gammas=c['gammas'], rule='Gamma is never chosen by LDS; unqualified is not a proof of noninvertibility.',
                                   size_baseline_sign='fewer retained images -> larger loss (prediction = -N_j)',
                                   primary_endpoint='raw', primary_target='ce', exploratory_targets=['oneminusp', 'brier'],
                                   ema_role='robustness only; raw and EMA are not independent replicates')
    return summary, tables, truths
