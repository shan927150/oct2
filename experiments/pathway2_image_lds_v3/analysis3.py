"""Collect-side analysis for the image-level LDS (numpy only).

Truth is recomputed from stored log probabilities; result.json values are a
cross-check. Unit of inference = subset. Primary: image-mean CE, raw endpoint.
"""
from pathlib import Path
import numpy as np
import runtime as rt
import qtargets as Q
import imgdesign as ID
import stats2 as S  # seed32 v2, unchanged
from analysis_stats import spearman, crossfit_ridge  # LDS v1, unchanged

ENDPOINTS = ('raw', 'ema')
KS = (1, 2, 3, 4, 8, 16, 32)
STABLE_SEEDS = (42, 43, 44)


def load(root, c, allow_partial=False):
    import targets as T32
    root = Path(root)
    h = rt.sha(root / 'config.json')
    frozen = rt.read(root / 'designs.json')
    base, design = frozen['base'], frozen['design']
    M = np.asarray(design['masks'], dtype=float)
    N = M.shape[1]
    _, units, _ = T32.query_units(base)
    macro_w = np.asarray(base['query_weights'])
    cells, missing, labels_test, train_labels = {}, [], None, None
    for cell in c['cells']:
        d = root / 'cells' / cell['name']
        if not (d / 'complete.json').exists():
            missing.append('cells/' + cell['name'])
            continue
        rt.completed(d, h, cell)
        with np.load(d / 'interface.npz') as z:
            if labels_test is None:
                labels_test = z['classes'][N:].astype(np.int64)
                train_labels = z['classes'][:N].astype(np.int64)
            elif not np.array_equal(labels_test, z['classes'][N:]):
                raise ValueError('Query labels differ between cells')
            vals = {ep: Q.values(z['log_' + ep][N:], labels_test, macro_w, units) for ep in ENDPOINTS}
        stored = rt.read(d / 'result.json')
        for ep in ENDPOINTS:
            for t in Q.TARGETS:
                for wname in ('img', 'macro'):
                    a, b = vals[ep][t]['Y_' + wname], stored['Y'][ep][t][wname]
                    if abs(a - b) > 1e-10 * max(1., abs(b)):
                        raise ValueError(f'Stored Y differs from recomputed Y: {cell["name"]} {ep} {t} {wname}')
        expected = N if cell['kind'] == 'full' else int(M[cell['subset']].sum())
        if stored['n_train_images'] != expected:
            raise ValueError('Retained image count mismatch: ' + cell['name'])
        cells[cell['name']] = vals
    proxies = {}
    for s in c['proxy_tasks']:
        d = root / 'proxy' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='proxy', seed=s))
            with np.load(d / 'proxy.npz') as z:
                proxies[s] = {k: z[k] for k in z.files}
        else:
            missing.append(f'proxy/seed{s}')
    scores, checks = {}, {}
    for s in c['score_tasks']:
        for t in c['score_targets']:
            d = root / 'score' / f'seed{s}_{t}'
            if (d / 'complete.json').exists():
                rt.completed(d, h, dict(kind='score', seed=s, target=t))
                with np.load(d / 'scores.npz') as z:
                    scores[(s, t)] = dict(arrays={k: z[k] for k in z.files}, result=rt.read(d / 'result.json'))
            else:
                missing.append(f'score/seed{s}_{t}')
        if s in STABLE_SEEDS:
            d = root / 'score' / f'seed{s}_check'
            if (d / 'complete.json').exists():
                rt.completed(d, h, dict(kind='score', seed=s, target='check'))
                checks[s] = rt.read(d / 'result.json')['gate']
            else:
                missing.append(f'score/seed{s}_check')
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} artefacts incomplete (first: {missing[:5]}). Use resubmit or collect --allow-partial.')
    class_images = None if train_labels is None else ID.composition_matrix(M, train_labels)[0]
    return dict(base=base, design=design, M=M, N=N, units=units, cells=cells, missing=missing,
                proxies=proxies, scores=scores, checks=checks, labels_test=labels_test, class_images=class_images)


def truth(data, c, restrict=None):
    """Seed-level truth using seeds complete for every subset (and the full baseline)."""
    m = len(data['M'])
    name = c['design_name']
    seeds = [s for s in c['seeds'] if (restrict is None or s in restrict)
             and all(f'{name}_s{j:03d}_seed{s}' in data['cells'] for j in range(m)) and f'full_seed{s}' in data['cells']]
    out = dict(seeds=seeds)
    if not seeds:
        return out
    for ep in ENDPOINTS:
        out[ep] = {}
        for t in Q.TARGETS:
            block = {}
            for key in ('Y_img', 'Y_macro', 'Yi', 'Yq'):
                sub = np.array([[data['cells'][f'{name}_s{j:03d}_seed{s}'][ep][t][key] for s in seeds] for j in range(m)])
                full = np.array([data['cells'][f'full_seed{s}'][ep][t][key] for s in seeds])
                block[key] = sub
                block['D' + key[1:]] = sub - full[None]
            out[ep][t] = block
    return out


def if_scores(data, c, target, gamma):
    """Seed-mean image IF scores; None plus reason unless every score seed qualified."""
    taus, reasons = [], []
    for s in c['score_seeds']:
        if s in STABLE_SEEDS and s in data['checks'] and not data['checks'][s]['passed']:
            return None, f'seed{s}: patient-level gate failed'
        sc = data['scores'].get((s, target))
        if sc is None:
            return None, f'score seed{s}_{target} missing'
        q = next(x for x in sc['result']['gammas'] if x['gamma'] == gamma)
        if not q['qualified']:
            reasons.append(f'seed{s}: ' + '; '.join(q['reasons']))
            continue
        taus.append(sc['arrays'][f'tau_gamma{gamma:g}'])
    if reasons:
        return None, 'unqualified: ' + ' | '.join(reasons)
    return np.mean(taus, 0), None


def proxy_mean(data, seeds, key):
    if not seeds or not all(s in data['proxies'] for s in seeds):
        return None
    return np.mean([data['proxies'][s][key] for s in seeds], 0)


def analyze(data, c):
    M = data['M']
    m = len(M)
    nimg = M.sum(1)
    tr = truth(data, c)
    summary = dict(design=dict(name=c['design_name'], m=m, retained_images=int(data['design']['retained_images']),
                               n_train=int(data['N']), n_query=int(len(data['labels_test'])) if data['labels_test'] is not None else None,
                               n_images_range=[float(nimg.min()), float(nimg.max())], seeds_complete=tr['seeds']),
                   missing=data['missing'], proxies=sorted(data['proxies']),
                   score_tasks=sorted(f'seed{s}_{t}' for s, t in data['scores']),
                   patient_level_gates={str(k): v for k, v in data['checks'].items()},
                   critical_spearman=S.critical_spearman(m),
                   macro={}, seed_curves={}, per_query={}, gamma={}, equivalence={})
    tables = dict(lds=[], seed_curve=[], per_query=[], gamma=[])
    if not tr['seeds']:
        return summary, tables
    seeds = tr['seeds']
    ks = sorted({k for k in KS if k <= len(seeds)} | {len(seeds)})
    for ep in ENDPOINTS:
        summary['macro'][ep] = {}
        for t in Q.TARGETS:
            for wname in ('img', 'macro'):
                Y, D = tr[ep][t]['Y_' + wname], tr[ep][t]['D_' + wname]
                Dbar = D.mean(1)
                preds, status = {}, {}
                pk = f'tau_{wname}_{t}'
                pm = proxy_mean(data, seeds, pk)
                if pm is not None:
                    preds['proxy_deleter'] = ID.deleter_prediction(M, pm)
                    if ep == 'raw' and wname == 'img':
                        retained = M @ pm
                        summary['equivalence'][t] = dict(
                            deleter=spearman(preds['proxy_deleter'], Dbar), retained_sum=spearman(retained, Dbar),
                            matched=spearman(ID.matched_prediction(M, pm), Dbar),
                            sign_flip=spearman(-preds['proxy_deleter'], Dbar),
                            note='N_j is constant, so deleter, retained and matched rankings coincide by construction.')
                if wname == 'img' and t in c['score_targets']:
                    for g in c['gammas']:
                        tau, why = if_scores(data, c, t, g)
                        status[f'IF_gamma{g:g}'] = why or 'ok'
                        if tau is not None:
                            preds[f'IF_gamma{g:g}_deleter'] = ID.deleter_prediction(M, tau)
                ridge = crossfit_ridge(data['class_images'], Y.mean(1)) if data.get('class_images') is not None else None
                if ridge is not None:
                    preds['class_composition_ridge'] = ridge
                block = {}
                for pname, p in preds.items():
                    fitted = pname.startswith('class_composition')
                    block[pname] = S.lds_block(p, Dbar, nimg, fitted_on_truth=fitted)
                    b = block[pname]
                    ci = b.get('bootstrap_ci_conditional_on_seed_panel') or [None, None]
                    tables['lds'].append(dict(endpoint=ep, target=t, weighting=wname, predictor=pname, seeds=len(seeds),
                                              spearman=b['spearman'], ci_low=ci[0], ci_high=ci[1],
                                              perm_p=b.get('permutation_p_one_sided')))
                summary['macro'][ep][f'{t}_{wname}'] = dict(predictors=block, if_status=status,
                    size_baseline='undefined: every subset keeps the same number of images')
                if ep == 'raw':
                    curve = dict(se=S.se_curve(Y, ks), reliability=S.reliability_curve(Y, ks))
                    fns = {}
                    if all(s in data['proxies'] for s in seeds):
                        Pm = np.stack([data['proxies'][s][pk] for s in seeds])
                        fns['proxy (same k seeds)'] = lambda idx, Pm=Pm: ID.deleter_prediction(M, Pm[idx].mean(0))
                    for g in (1., min(c['gammas'])):
                        if f'IF_gamma{g:g}_deleter' in preds:
                            fns[f'IF gamma={g:g} (score seeds, fixed)'] = lambda idx, v=preds[f'IF_gamma{g:g}_deleter']: v
                    curve['lds'] = S.lds_curve(D, fns, ks) if fns else []
                    summary['seed_curves'][f'{t}_{wname}'] = curve
                    for kind in ('se', 'lds'):
                        for row in (curve[kind] if kind == 'se' else curve['lds']):
                            tables['seed_curve'].append(dict(target=t, weighting=wname, kind=kind, **row))
                    for row in curve['reliability']['rows']:
                        tables['seed_curve'].append(dict(target=t, weighting=wname, kind='reliability', **row))
    # ------------------------------------------------- per-query LDS (proxy)
    for t in Q.TARGETS:
        for unit, key, tkey in (('image', 'pred_qimage_', 'Di'), ('patient', 'pred_qpatient_', 'Dq')):
            pm = proxy_mean(data, seeds, key + t)
            if pm is None:
                continue
            res = S.per_query_lds(pm, tr['raw'][t][tkey].mean(1))
            per = res.pop('per_query')
            summary['per_query'][f'{t}_{unit}'] = dict(proxy=res, per_query=per)
            tables['per_query'].append(dict(target=t, unit=unit, **{k: v for k, v in res.items() if not isinstance(v, list)},
                                            ci_low=(res.get('bootstrap_ci_conditional_on_seed_panel') or [None, None])[0],
                                            ci_high=(res.get('bootstrap_ci_conditional_on_seed_panel') or [None, None])[1]))
    # ------------------------------------------------------------ gamma panel
    for t in c['score_targets']:
        summary['gamma'][t] = {}
        for g in c['gammas']:
            rows = []
            for s in c['score_seeds']:
                sc = data['scores'].get((s, t))
                if sc is None:
                    rows.append(dict(seed=s, status='missing'))
                    continue
                q = next(x for x in sc['result']['gammas'] if x['gamma'] == g)
                rows.append(dict(seed=s, qualified=q['qualified'], reasons=q['reasons'],
                                 true_rel_residual=q['true_rel_residual'],
                                 spearman_if_vs_proxy_images=q['spearman_if_vs_proxy_images'],
                                 lanczos_steps=sc['result']['lanczos']['steps']))
            tau, why = if_scores(data, c, t, g)
            sp = [r['spearman_if_vs_proxy_images'] for r in rows if r.get('spearman_if_vs_proxy_images') is not None]
            entry = dict(by_seed=rows, status='solved' if tau is not None else why,
                         mean_spearman_if_vs_proxy=float(np.mean(sp)) if sp and tau is not None else None)
            if tau is not None:
                entry['lds_img_raw'] = S.lds_block(ID.deleter_prediction(M, tau), tr['raw'][t]['D_img'].mean(1), nimg)
            summary['gamma'][t][f'{g:g}'] = entry
            tables['gamma'].append(dict(target=t, gamma=g, status=entry['status'],
                                        mean_spearman_if_vs_proxy=entry['mean_spearman_if_vs_proxy'],
                                        lds=(entry.get('lds_img_raw') or {}).get('spearman')))
    summary['prespecified'] = dict(primary='image-mean CE, raw', gammas=c['gammas'],
                                   rule='Gamma is never chosen by LDS; unqualified shifts are reported, not tuned.')
    return summary, tables
