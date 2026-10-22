"""Collect-side analysis of the interface LDS (numpy / scipy only; runs on a CPU node).

Notation: J = 32 subsets, R = 4343 shadow-3 interface rows (first N = 2045 are training rows in training
order, the rest are heldout nonmember rows), four classes. Truth T[s, j, r, c] = p_sub - p_full of the v3 run,
averaged over all 8 Stage-1 seeds. Predictions are averaged over the frozen score panel.

IF prediction (matched normalisation, the change the influence function actually predicts):
    dz_j = (N / K_j) J_z (H + gamma I)^{-1} b_j - (N / K_j - 1) J_z (H + gamma I)^{-1} sum_i g_i,
    dp_j = p0 * (dz_j - <p0, dz_j>)                         (first order, J_p)
    dp_j^logit = softmax(log p0 + dz_j) - p0                 (same step through the softmax; secondary)
With fixed K_j the per-output ranking across subsets equals that of the deleted-sum form.
Other rungs: proxy J_p b_j (no Hessian) and the true parameter change linearised, J_p (theta_j - theta_full).

Metrics:
  per-output LDS  mean over outputs (r, c) of Spearman across subsets; inference for the mean by a joint
                  permutation of subsets (exact reduction to a J x J matrix) and a subset bootstrap.
  ceiling         split-half (4 vs 4 seeds) reliability of the truth per output, Spearman-Brown to 8 seeds.
  functionals     mean change of each class probability (and of p_true) over deleted / retained / nonmember rows.
  removal effect  per training row, change when deleted minus change when retained; ranked across rows.
  magnitude       within-subset Pearson, slope and norm ratio of matched predictions against truth.
  bridge          L1 = sum_r vbar . dp_true, L3 = sum_r vbar . dp_hat with v6's adjoint; L3 must reproduce v6's
                  scores; row-group oracle replaces dp_hat by dp_true on one row group at a time.
"""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import runtime as rt
import stats2 as S               # seed32 v2, unchanged
from analysis_stats import spearman  # LDS v1, unchanged

GROUPS = ('nonmember', 'training')


# ----------------------------------------------------------------------- loading
def v3_interface(c, name, rows):
    d = Path(c['v3_out']) / 'cells' / name
    rt.verify_hashes({str(d / 'complete.json'): c['input_sha256'][str(d / 'complete.json')]})
    cell = next(x for x in c['v3_cells'] if x['name'] == name)
    rt.completed(d, c['v3_config_sha256'], cell)
    with np.load(d / 'interface.npz') as z:
        if not np.array_equal(z['raw_index'], rows):
            raise ValueError('v3 interface rows differ: ' + name)
        return np.asarray(z['raw'], dtype=np.float64)


def v6_dir(c, rel, cell):
    d = Path(c['v6_out']) / rel
    rt.verify_hashes({str(d / 'complete.json'): c['input_sha256'][str(d / 'complete.json')]})
    rt.completed(d, c['v6_config_sha256'], cell)
    return d


def vbar_from_v6(c, fr, seed):
    d = v6_dir(c, f'base/seed{seed}', dict(kind='base6', seed=seed))
    with np.load(d / 'v.npz') as z:
        V = z['v']
    gi = c['attack_gammas'].index(fr['attack_gamma_star'])
    out = np.zeros(V.shape[-2:])
    for ci, cls in enumerate(c['classes']):
        if cls in fr['qualified_classes']:
            out += fr['omega'][str(cls)] * V[:, ci, gi].mean(0)
    return out


def load(root, c, allow_partial=False):
    root = Path(root)
    h = rt.sha(root / 'config.json')
    rt.completed(root / 'smoke', h, dict(kind='ifsmoke'))
    frozen = rt.read(root / 'smoke/frozen.json')
    with np.load(root / 'smoke/rows.npz') as z:
        rows = {k: z[k] for k in z.files}
    M = np.asarray(rt.read(root / 'designs.json')['design']['masks'], dtype=float)
    if not np.array_equal(M, rows['masks']):
        raise ValueError('Design masks differ between designs.json and smoke rows')
    seeds, m = c['seeds'], c['design']['m']
    truth = np.zeros((len(seeds), m) + (len(rows['raw_index']), 4))
    full = {}
    for si, s in enumerate(seeds):
        full[s] = v3_interface(c, f'full_seed{s}', rows['raw_index'])
        for j in range(m):
            truth[si, j] = v3_interface(c, f'{c["design_name"]}_s{j:03d}_seed{s}', rows['raw_index']) - full[s]
    solves, missing = {}, []
    for s in frozen['score_panel']:
        d = root / 'solve' / f'seed{s}'
        if (d / 'complete.json').exists():
            rt.completed(d, h, dict(kind='ifsolve', seed=s))
            with np.load(d / 'interface.npz') as z:
                solves[s] = dict(arrays={k: z[k] for k in z.files}, result=rt.read(d / 'result.json'))
        else:
            missing.append(f'solve/seed{s}')
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} solves incomplete ({missing}). Use resubmit or collect --allow-partial.')
    fr6 = rt.read(v6_dir(c, 'smoke', dict(kind='xsmoke')) / 'frozen.json')
    v6_truth = {}
    for cell in c['cells']:
        d = v6_dir(c, f'truth/{cell["name"]}', dict(cell, kind='truth6'))
        v6_truth[(cell['subset'], cell['seed'])] = rt.read(d / 'result.json')['rows']
    v6_scores = {}
    for s in frozen['score_panel']:
        d = v6_dir(c, f'score/seed{s}', dict(kind='score6', seed=s))
        with np.load(d / 'scores.npz') as z:
            v6_scores[s] = {k: z[k] for k in z.files if k.startswith('full_pred_gamma')}
    vbar = {s: vbar_from_v6(c, fr6, s) for s in seeds}
    return dict(frozen=frozen, rows=rows, M=M, truth=truth, full=full, solves=solves, missing=missing, fr6=fr6,
                v6_truth=v6_truth, v6_scores=v6_scores, vbar=vbar, smoke=rt.read(root / 'smoke/result.json'))


# ----------------------------------------------------------------------- algebra
def softmax_jvp(p0, dz):
    return p0 * (dz - (p0 * dz).sum(-1, keepdims=True))


def softmax_step(p0, dz):
    z = np.log(np.clip(p0, 1e-300, None)) + dz
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True) - p0


def matched_dz(dz_if, dz_gall, K, N):
    """dz_if [J, ..., R, C] deleted-sum logit tangents, dz_gall [..., R, C]; K [J] retained counts."""
    a = (N / K).reshape((-1,) + (1,) * (dz_if.ndim - 1))
    return a * dz_if - (a - 1) * dz_gall[None]


# ----------------------------------------------------------------------- statistics
def _centered_unit_ranks(X):
    r = rankdata(X, axis=0)
    r = r - r.mean(0)
    n = np.linalg.norm(r, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, r / n, np.nan)


def per_output_lds(P, T):
    """Spearman across subsets (axis 0) for every output column; NaN for a constant column."""
    return (_centered_unit_ranks(P) * _centered_unit_ranks(T)).sum(0)


def mean_lds_inference(P, T, perms=2000, boots=2000, seed=20261022):
    """Mean per-output LDS with a joint subset permutation p (one-sided) and a subset-bootstrap CI."""
    A, B = _centered_unit_ranks(P), _centered_unit_ranks(T)
    ok = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
    A, B = A[:, ok], B[:, ok]
    O = A.shape[1]
    obs = float((A * B).sum() / O)
    C = A @ B.T / O                       # mean over outputs of a_j b_k: permutation statistic = sum_j C[j, pi(j)]
    rng = np.random.default_rng(seed)
    J = len(C)
    null = np.array([C[np.arange(J), rng.permutation(J)].sum() for _ in range(perms)])
    p = float((np.sum(null >= obs) + 1) / (perms + 1))
    boot = []
    Pk, Tk = P[:, ok], T[:, ok]
    for _ in range(boots):
        i = rng.integers(J, size=J)
        a, b = _centered_unit_ranks(Pk[i]), _centered_unit_ranks(Tk[i])
        boot.append(float(np.nanmean((a * b).sum(0))))
    return dict(mean=obs, ci=np.quantile(boot, [.025, .975]).tolist(), perm_p_one_sided=p, outputs=int(O),
                null_q95=float(np.quantile(null, .95)))


def mean_lds_perm(P, T, perms=2000, seed=20261022):
    A, B = _centered_unit_ranks(P), _centered_unit_ranks(T)
    ok = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
    A, B = A[:, ok], B[:, ok]
    O = A.shape[1]
    obs = float((A * B).sum() / O)
    C = A @ B.T / O
    rng = np.random.default_rng(seed)
    J = len(C)
    null = np.array([C[np.arange(J), rng.permutation(J)].sum() for _ in range(perms)])
    return dict(mean=obs, perm_p_one_sided=float((np.sum(null >= obs) + 1) / (perms + 1)), outputs=int(O))


def split_half(Ts, cols, splits=10, seed=20261022):
    """Ts [S, J, O]: mean over splits of the per-output Spearman between disjoint seed halves; SB to S seeds."""
    rng = np.random.default_rng(seed)
    S_ = Ts.shape[0]
    r = []
    for _ in range(splits):
        i = rng.permutation(S_)
        a, b = Ts[i[:S_ // 2]].mean(0)[:, cols], Ts[i[S_ // 2:2 * (S_ // 2)]].mean(0)[:, cols]
        r.append(per_output_lds(a, b))
    r = np.nanmean(np.stack(r), 0)
    sb = 2 * r / (1 + r)
    return dict(half=float(np.nanmean(r)), spearman_brown=float(np.nanmean(sb)),
                ceiling=float(np.nanmean(np.sqrt(np.clip(sb, 0, None)))))


def paired(p1, p2, truth, draws=2000, seed=20261022):
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
                fraction_not_better=float((vals <= 0).mean()))


# ----------------------------------------------------------------------- analysis
def predictions(data, c):
    """Per-seed predictions [S_panel, J, R, C] for every rung (matched IF; deleted-sum proxy; true dtheta)."""
    M, N = data['M'], data['M'].shape[1]
    K = M.sum(1)
    gam = c['gammas']
    panel = [s for s in data['frozen']['score_panel'] if s in data['solves']]
    out, qual = {}, {}
    for gi, g in enumerate(gam):
        flags = [next(q for q in data['solves'][s]['result']['gammas'] if float(q['gamma']) == float(g))['qualified']
                 for s in panel]
        qual[f'{g:g}'] = all(flags) and bool(panel)
    for s in panel:
        a = data['solves'][s]['arrays']
        p0 = a['p0']
        for gi, g in enumerate(gam):
            dz = matched_dz(a['dz_if'][:, gi], a['dz_gall'][gi], K, N)
            out.setdefault(f'IF (gamma={g:g})', []).append(softmax_jvp(p0, dz))
            if float(g) == float(c['primary_gamma']):
                out.setdefault(f'IF logit (gamma={g:g})', []).append(softmax_step(p0, dz))
        out.setdefault('proxy', []).append(softmax_jvp(p0, a['dz_proxy']))
        out.setdefault('true dtheta (linear)', []).append(softmax_jvp(p0, a['dz_true']))
        out.setdefault('true dtheta (logit)', []).append(softmax_step(p0, a['dz_true']))
    return {k: np.stack(v) for k, v in out.items()}, qual, panel


def flat(X, rows):
    """[J, R, C] -> [J, rows * C]."""
    return X[:, rows].reshape(X.shape[0], -1)


def analyze(data, c):
    rows, M = data['rows'], data['M']
    N, R = M.shape[1], len(rows['raw_index'])
    y = rows['y']
    J = len(M)
    nimg = M.sum(1)
    Tseeds = data['truth']                       # [S, J, R, C]
    T = Tseeds.mean(0)
    preds, qual, panel = predictions(data, c)
    g0 = f"{c['primary_gamma']:g}"
    summary = dict(design=dict(subsets=J, rows=R, training_rows=N, nonmember_rows=R - N, seeds=c['seeds'], score_panel=panel,
                               gammas=c['gammas'], primary_gamma=c['primary_gamma']),
                   frozen=data['frozen'], missing=data['missing'], smoke=data['smoke'], gamma_qualified=qual,
                   gates={}, per_output={}, primary={}, ceilings={}, training_conditional={}, functionals={},
                   removal_effect={}, magnitude={}, bridge={}, solver={})
    tables = dict(per_output=[], functionals=[], bridge=[], magnitude=[])
    if not panel:
        return summary, tables
    # ------------------------------------------------------------------ gates
    p0dev = max(float(np.abs(data['solves'][s]['arrays']['p0'] - data['full'][s]).max()) for s in panel)
    vdev = max(float(np.abs(data['solves'][s]['arrays']['vbar'] - data['vbar'][s]).max()) for s in panel)
    summary['gates'] = dict(p0_vs_v3_max_abs=p0dev, vbar_vs_v6_max_abs=vdev,
                            tau_vs_v6={s: data['solves'][s]['result']['tau_vs_v6'] for s in panel},
                            forward_reverse_max_rel=max(data['solves'][s]['result']['forward_reverse_identity_max_rel'] for s in panel))
    if p0dev > c['v3_prediction_tol'] or vdev != 0.:
        raise ValueError(f'Solve inputs differ from v3 / v6 (p0 {p0dev:.2e}, vbar {vdev:.2e})')
    summary['solver'] = {s: dict(dim=data['solves'][s]['result']['solver']['dim'],
                                 matvecs=data['solves'][s]['result']['solver']['matvecs'],
                                 ritz_min=data['solves'][s]['result']['solver']['ritz_min'],
                                 hours=data['solves'][s]['result']['timing']['elapsed_seconds'] / 3600) for s in panel}
    usable = {k: v for k, v in preds.items() if not k.startswith('IF') or qual.get(k.split('=')[1].rstrip(')'), False)}
    mean_pred = {k: v.mean(0) for k, v in usable.items()}
    group_rows = dict(nonmember=np.arange(N, R), training=np.arange(N))
    # ------------------------------------------------------------- per-output LDS
    prim = f'IF (gamma={g0})'
    boot_rungs = {prim, 'proxy', 'true dtheta (linear)'}
    for name, P in mean_pred.items():
        e = {}
        for grp, rr in group_rows.items():
            Pf, Tf = flat(P, rr), flat(T, rr)
            st = mean_lds_inference(Pf, Tf, c['permutations'], c['bootstraps']) if name in boot_rungs else \
                mean_lds_perm(Pf, Tf, c['permutations'])
            ptrue = per_output_lds(P[:, rr, y[rr]], T[:, rr, y[rr]])
            st['p_true_only_mean'] = float(np.nanmean(ptrue))
            st['by_class'] = {str(k): float(np.nanmean(per_output_lds(P[:, rr, k], T[:, rr, k]))) for k in range(P.shape[2])}
            e[grp] = st
            tables['per_output'].append(dict(rung=name, group=grp, mean_lds=st['mean'], ci_low=(st.get('ci') or [None, None])[0],
                                             ci_high=(st.get('ci') or [None, None])[1], perm_p=st['perm_p_one_sided'],
                                             p_true_only=st['p_true_only_mean'], **{f'class{k}': v for k, v in st['by_class'].items()}))
        summary['per_output'][name] = e
    for g in c['gammas']:
        if not qual[f'{g:g}']:
            summary['per_output'][f'IF (gamma={g:g})'] = dict(status='unqualified for at least one panel seed')
    if prim in mean_pred:
        e = summary['per_output'][prim]['nonmember']
        summary['primary'] = dict(statistic='mean per-output LDS, nonmember rows x 4 classes', rung=prim, **e,
                                  passed=None if data['missing'] else bool(e['perm_p_one_sided'] < .05))
    else:
        summary['primary'] = dict(status=f'primary gamma {g0} unqualified; not substituted')
    # ------------------------------------------------------------- ceilings
    for grp, rr in group_rows.items():
        cols = np.arange(len(rr) * 4)
        summary['ceilings'][grp] = split_half(Tseeds[:, :, rr].reshape(len(Tseeds), J, -1), cols)
        if len(panel) >= 2 and prim in usable:
            Ps = usable[prim][:, :, rr].reshape(len(panel), J, -1)
            summary['ceilings'][grp]['prediction_split_half'] = split_half(Ps, cols)['half']
    # ------------------------------------------------------------- training rows by deletion status
    for name in (prim, 'proxy', 'true dtheta (linear)'):
        if name not in mean_pred:
            continue
        P = mean_pred[name]
        res = {}
        for status, flag in (('retained', 1.), ('deleted', 0.)):
            vals = []
            for r in range(N):
                sel = M[:, r] == flag
                if sel.sum() >= 5:
                    for k in range(4):
                        v = spearman(P[sel, r, k], T[sel, r, k])
                        if v is not None:
                            vals.append(v)
            res[status] = dict(mean_lds=float(np.mean(vals)), outputs=len(vals))
        summary['training_conditional'][name] = res
    # ------------------------------------------------------------- group functionals
    def functionals(X):
        out = {}
        for grp in ('deleted', 'retained', 'nonmember'):
            for k in list(range(4)) + ['p_true']:
                vals = np.zeros(J)
                for j in range(J):
                    rr = np.arange(N, R) if grp == 'nonmember' else np.flatnonzero(M[j] == (1. if grp == 'retained' else 0.))
                    vals[j] = X[j, rr, y[rr]].mean() if k == 'p_true' else X[j, rr, k].mean()
                out[f'{grp} {"p_true" if k == "p_true" else "class " + str(k)}'] = vals
        return out
    Ft = functionals(T)
    for name, P in mean_pred.items():
        Fp = functionals(P)
        res = {}
        for key, t in Ft.items():
            if name in boot_rungs:
                b = S.lds_block(Fp[key], t, nimg)
                res[key] = dict(spearman=b['spearman'], ci=b.get('bootstrap_ci_conditional_on_seed_panel'),
                                perm_p=b.get('permutation_p_one_sided'))
            else:
                res[key] = dict(spearman=spearman(Fp[key], t))
            tables['functionals'].append(dict(rung=name, functional=key, **{k: v for k, v in res[key].items() if k != 'ci'},
                                              ci_low=(res[key].get('ci') or [None, None])[0], ci_high=(res[key].get('ci') or [None, None])[1]))
        summary['functionals'][name] = res
    # ------------------------------------------------------------- removal effect (self influence)
    def removal(X):
        del_mean = np.stack([X[M[:, r] == 0, r].mean(0) for r in range(N)])
        ret_mean = np.stack([X[M[:, r] == 1, r].mean(0) for r in range(N)])
        return del_mean - ret_mean                                        # [N, C]
    et = removal(T)
    rel = []
    rng = np.random.default_rng(7)
    for _ in range(10):
        i = rng.permutation(len(Tseeds))
        a, b = removal(Tseeds[i[:len(i) // 2]].mean(0)), removal(Tseeds[i[len(i) // 2:]].mean(0))
        rel.append(spearman(a[np.arange(N), y[:N]], b[np.arange(N), y[:N]]))
    summary['removal_effect']['truth_split_half_p_true'] = float(np.mean(rel))
    summary['removal_effect']['truth_mean_p_true'] = float(et[np.arange(N), y[:N]].mean())
    for name, P in mean_pred.items():
        ep = removal(P)
        a, b = ep[np.arange(N), y[:N]], et[np.arange(N), y[:N]]
        boot = []
        for _ in range(1000):
            i = rng.integers(N, size=N)
            boot.append(spearman(a[i], b[i]))
        summary['removal_effect'][name] = dict(
            spearman_p_true=spearman(a, b), ci=np.quantile(boot, [.025, .975]).tolist(),
            by_class={str(k): spearman(ep[:, k], et[:, k]) for k in range(4)},
            mean_ratio=float(a.mean() / b.mean()) if b.mean() != 0 else None)
    # ------------------------------------------------------------- magnitude (matched IF and true dtheta)
    for name, P in mean_pred.items():
        if name == 'proxy':
            continue
        res = {}
        for grp in ('deleted', 'retained', 'nonmember'):
            pr, tr = [], []
            for j in range(J):
                rr = np.arange(N, R) if grp == 'nonmember' else np.flatnonzero(M[j] == (1. if grp == 'retained' else 0.))
                pr.append(P[j, rr].ravel())
                tr.append(T[j, rr].ravel())
            within = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(pr, tr)]))
            a, b = np.concatenate(pr), np.concatenate(tr)
            res[grp] = dict(within_subset_pearson=within, slope_truth_on_pred=float((a @ b) / max(a @ a, 1e-300)),
                            norm_ratio_pred_over_truth=float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-300)))
            tables['magnitude'].append(dict(rung=name, group=grp, **res[grp]))
        summary['magnitude'][name] = res
    # ------------------------------------------------------------- attack bridge
    summary['bridge'] = bridge(data, c, preds, qual, panel, tables)
    return summary, tables


def bridge(data, c, preds, qual, panel, tables):
    fr6, M, N = data['fr6'], data['M'], data['M'].shape[1]
    J, R = len(M), len(data['rows']['raw_index'])
    gstar = f"{fr6['attack_gamma_star']:g}"
    om = {int(k): v for k, v in fr6['omega'].items()}
    seeds = c['seeds']
    T6, L1v6 = np.zeros(J), np.zeros(J)
    n = np.zeros(J)
    for (j, s), rows in data['v6_truth'].items():
        for r in rows:
            if r['cls'] in om:
                T6[j] += om[r['cls']] * (r['J10'] - r['J00'])
                L1v6[j] += om[r['cls']] * r['L1'][gstar]
        n[j] += len(fr6['attack_seeds'])
    T6, L1v6 = T6 / n, L1v6 / n
    vb = data['vbar']
    L1 = np.mean([(vb[s][None] * data['truth'][si]).sum((1, 2)) for si, s in enumerate(seeds)], 0)
    out = dict(attack_gamma_star=fr6['attack_gamma_star'], v6_truth_mean=float(T6.mean()),
               L1_reproduces_v6_max_abs=float(np.abs(L1 - L1v6).max()), rungs={}, paired={}, v6_reproduction={})
    if out['L1_reproduces_v6_max_abs'] > 1e-9 * max(1., float(np.abs(L1v6).max())):
        raise ValueError(f"vbar . dp_true does not reproduce v6 L1 ({out['L1_reproduces_v6_max_abs']:.2e})")
    pi = [seeds.index(s) for s in panel]
    Tp = data['truth'][pi]
    groups = dict(deleted=lambda j: np.flatnonzero(M[j] == 0), retained=lambda j: np.flatnonzero(M[j] == 1),
                  nonmember=lambda j: np.arange(N, R))
    rung = {'L1 (8 seeds)': L1,
            'L1 (score panel)': np.mean([(vb[s][None] * Tp[k]).sum((1, 2)) for k, s in enumerate(panel)], 0)}
    g0 = f"{c['primary_gamma']:g}"
    for g in c['gammas']:
        key = f'IF (gamma={g:g})'
        if not qual[f'{g:g}']:
            continue
        P = preds[key]                                               # [panel, J, R, C], matched
        rung[f'L3 (gamma={g:g})'] = np.mean([(vb[s][None] * P[k]).sum((1, 2)) for k, s in enumerate(panel)], 0)
        # v6 reproduction: v6 pred_j = sum_i ((N/K_j) M_ji - 1) tau_i equals vbar . dp_matched_j exactly
        dev = []
        for k, s in enumerate(panel):
            ref = data['v6_scores'][s].get(f'full_pred_gamma{g:g}')
            if ref is not None:
                ours = (vb[s][None] * P[k]).sum((1, 2))
                dev.append(float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-300)))
        out['v6_reproduction'][f'{g:g}'] = max(dev) if dev else None
        if float(g) == float(c['primary_gamma']):
            for grp, sel in groups.items():
                mixed = []
                for k, s in enumerate(panel):
                    X = P[k].copy()
                    for j in range(J):
                        rr = sel(j)
                        X[j, rr] = Tp[k, j, rr]
                    mixed.append((vb[s][None] * X).sum((1, 2)))
                rung[f'L3 (gamma={g:g}), {grp} rows <- retrained change'] = np.mean(mixed, 0)
    nimg = M.sum(1)
    for name, pred in rung.items():
        b = S.lds_block(pred, T6, nimg)
        out['rungs'][name] = dict(spearman=b['spearman'], ci=b.get('bootstrap_ci_conditional_on_seed_panel'),
                                  perm_p=b.get('permutation_p_one_sided'))
        tables['bridge'].append(dict(rung=name, spearman=b['spearman'], ci_low=(out['rungs'][name]['ci'] or [None, None])[0],
                                     ci_high=(out['rungs'][name]['ci'] or [None, None])[1], perm_p=out['rungs'][name]['perm_p']))
    base = f'L3 (gamma={g0})'
    if base in rung:
        for name in rung:
            if name != base and (name.startswith(base + ',') or name.startswith('L1')):
                out['paired'][f'{name} - {base}'] = paired(rung[name], rung[base], T6)
    return out
