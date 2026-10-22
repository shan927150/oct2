"""Collect-side analysis of the interface LDS (numpy / scipy only; runs on a CPU node).

Notation: J = 32 subsets, R = 4343 shadow-3 interface rows (first N = 2045 are training rows in training
order, the rest are heldout nonmember rows), four classes. Truth T[s, j, r, c] = p_sub - p_full of the v3 run,
averaged over the same frozen Stage-1 seed panel as the predictions. All-eight-seed truth is secondary.

IF prediction (matched normalisation, the change the influence function actually predicts):
    dz_j = (N / K_j) J_z (H + gamma I)^{-1} b_j - (N / K_j - 1) J_z (H + gamma I)^{-1} sum_i g_i,
    dp_j = p0 * (dz_j - <p0, dz_j>)                         (first order, J_p)
    dp_j^logit = softmax(log p0 + dz_j) - p0                 (same step through the softmax; secondary)
With fixed K_j the per-output ranking across subsets equals that of the deleted-sum form.
Other rungs: matched proxy J_p [(N/K)b_j - (N/K-1)sum_i g_i] (no Hessian), and J_p (theta_j - theta_full).

Metrics:
  per-output LDS  mean over outputs (r, c) of Spearman across subsets; inference for the mean by a joint
                  permutation of subsets (exact reduction to a J x J matrix) and a subset bootstrap.
  reliability     split-half reproducibility; Pearson attenuation reference under classical independent-noise
                  assumptions, not a strict ceiling on this Spearman statistic with shared model seeds.
  functionals     mean change of each class probability (and of p_true) over deleted / retained / nonmember rows.
  removal effect  per training row, change when deleted minus change when retained; ranked across rows.
  magnitude       within-subset Pearson, slope and norm ratio of matched predictions against truth.
  bridge          L1 = sum_r vbar . dp_true, L3 = sum_r vbar . dp_hat with v6's adjoint; L3 must reproduce v6's
                  scores; row-group oracle replaces dp_hat by dp_true on one row group at a time.
"""
from pathlib import Path
from itertools import combinations
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
        p = np.asarray(z['raw'], dtype=np.float64)
        if (p.shape != (len(rows), 4) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
                or not np.allclose(p.sum(-1), 1., rtol=0., atol=1e-10)):
            raise ValueError('Invalid v3 interface probabilities: ' + name)
        return p


def v6_dir(c, rel, cell):
    d = Path(c['v6_out']) / rel
    rt.verify_hashes({str(d / 'complete.json'): c['input_sha256'][str(d / 'complete.json')]})
    rt.completed(d, c['v6_config_sha256'], cell)
    return d


def vbar_from_v6(c, fr, seed):
    d = v6_dir(c, f'base/seed{seed}', dict(kind='base6', seed=seed))
    with np.load(d / 'v.npz') as z:
        V = z['v']
        if (list(z['attack_seeds']) != list(fr['attack_seeds']) or list(z['classes']) != c['classes']
                or list(z['attack_gammas']) != c['attack_gammas'] or V.ndim != 5
                or V.shape[:3] != (len(fr['attack_seeds']), len(c['classes']), len(c['attack_gammas']))
                or V.shape[-1] != 4 or not np.isfinite(V).all()):
            raise ValueError('v6 adjoint layout or values differ from the frozen panel')
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
    if (M.ndim != 2 or M.shape[0] != c['design']['m'] or not np.isin(M, [0, 1]).all()
            or np.any(M.sum(1) <= 0) or np.any(M.sum(1) >= M.shape[1])
            or len(np.unique(rows['raw_index'])) != len(rows['raw_index'])
            or not np.array_equal(rows['train_position'], np.r_[np.arange(M.shape[1]),
                                  -np.ones(len(rows['raw_index']) - M.shape[1], dtype=int)])):
        raise ValueError('Invalid subset masks or interface row layout')
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


def finite_mean(values):
    a = np.asarray(values, dtype=float)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else None


def mean_lds_inference(P, T, perms=2000, boots=2000, seed=20261022):
    """Joint subset permutation and bootstrap conditional on the fixed rows, model seeds and valid outputs.

    Every output uses the SAME subset permutation/draw. Bootstrap draws with undefined correlations on
    the original output support are discarded, never averaged over a changing set of outputs. A CI is
    withheld if fewer than 95% of draws remain defined. This condition is reported explicitly.
    """
    P, T = np.asarray(P, dtype=float), np.asarray(T, dtype=float)
    if (P.ndim != 2 or P.shape != T.shape or len(P) < 3 or not np.isfinite(P).all()
            or not np.isfinite(T).all() or perms < 1 or boots < 0):
        raise ValueError('Inference requires finite aligned [subsets >= 3, outputs] arrays and valid draw counts')
    A, B = _centered_unit_ranks(P), _centered_unit_ranks(T)
    ok = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
    A, B = A[:, ok], B[:, ok]
    O = A.shape[1]
    if O == 0:
        return dict(mean=None, ci=None, perm_p_one_sided=None, outputs=0, total_outputs=P.shape[1],
                    status='no nonconstant paired outputs', bootstrap_valid=0, bootstrap_requested=boots)
    C = A @ B.T / O                       # mean over outputs of a_j b_k: permutation statistic = sum_j C[j, pi(j)]
    obs = float(np.trace(C))
    rng = np.random.default_rng(seed)
    J = len(C)
    null = np.array([C[np.arange(J), rng.permutation(J)].sum() for _ in range(perms)])
    eps = 32 * np.finfo(float).eps * max(1., abs(obs))
    p = float((np.sum(null >= obs - eps) + 1) / (perms + 1))
    boot = []
    Pk, Tk = P[:, ok], T[:, ok]
    for _ in range(boots):
        i = rng.integers(J, size=J)
        a, b = _centered_unit_ranks(Pk[i]), _centered_unit_ranks(Tk[i])
        vals = (a * b).sum(0)
        if np.isfinite(vals).all():
            boot.append(float(vals.mean()))
    ci = np.quantile(boot, [.025, .975]).tolist() if boots and len(boot) >= max(2, .95 * boots) else None
    return dict(mean=obs, ci=ci, perm_p_one_sided=p, outputs=int(O), total_outputs=P.shape[1],
                bootstrap_valid=len(boot), bootstrap_requested=boots,
                ci_status='conditional on defined draws' if ci is not None else 'not requested or too many undefined draws',
                null_q95=float(np.quantile(null, .95)))


def mean_lds_perm(P, T, perms=2000, seed=20261022):
    return mean_lds_inference(P, T, perms=perms, boots=0, seed=seed)


def split_half(Ts, cols):
    """Enumerate balanced disjoint seed halves (35 for S=8). Not a strict Spearman noise ceiling.

    For odd S, leave one seed out in each split. Extrapolate Pearson reliability with factor S/h,
    not automatically 2. The attenuation reference requires exchangeable, additive independent noise
    and a predictor independent of that noise. Shared-seed IF predictions need not satisfy this.
    """
    S_ = Ts.shape[0]
    if S_ < 2:
        return dict(half=None, is_strict_ceiling=False, seeds=S_, splits=0, status='at least two seeds required')
    h = S_ // 2
    ranks, pearson = [], []
    for ia in combinations(range(S_), h):
        for ib in combinations([s for s in range(S_) if s not in ia], h):
            if ia >= ib:
                continue
            a, b = Ts[list(ia)].mean(0)[:, cols], Ts[list(ib)].mean(0)[:, cols]
            ranks.append(per_output_lds(a, b))
            a, b = a - a.mean(0), b - b.mean(0)
            with np.errstate(invalid='ignore', divide='ignore'):
                pearson.append((a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)))
    rp = np.stack(pearson)
    count = np.isfinite(rp).sum(0)
    r = np.divide(np.nansum(rp, axis=0), count, out=np.full(rp.shape[1], np.nan), where=count > 0)
    positive = np.isfinite(r) & (r > 0)
    factor = S_ / h
    sb = factor * r[positive] / (1 + (factor - 1) * r[positive])
    return dict(half=finite_mean(ranks), half_pearson=finite_mean(r), seeds=S_, half_size=h,
                splits=len(ranks), extrapolation_factor=factor, is_strict_ceiling=False,
                spearman_brown_pearson_positive_only=finite_mean(sb),
                attenuation_reference_pearson_positive_only=finite_mean(np.sqrt(np.clip(sb, 0, 1))),
                positive_reliability_outputs=int(positive.sum()), total_outputs=len(cols))


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
    return dict(lds_a=float(a), lds_b=float(b), difference=float(a - b),
                ci=np.quantile(vals, [.025, .975]).tolist() if len(vals) >= max(2, .95 * draws) else None,
                fraction_not_better=float((vals <= 0).mean()) if len(vals) else None,
                bootstrap_valid=len(vals), bootstrap_requested=draws)


# ----------------------------------------------------------------------- analysis
def predictions(data, c):
    """Per-seed predictions [S_panel, J, R, C], with matched normalisation for both IF and proxy."""
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
        R = len(data['rows']['raw_index'])
        expected = dict(p0=(R, 4), dz_if=(len(M), len(gam), R, 4), dz_gall=(len(gam), R, 4),
                        dz_proxy=(len(M), R, 4), dz_proxy_gall=(R, 4), dz_true=(len(M), R, 4), vbar=(R, 4))
        if not np.array_equal(a['gammas'], gam):
            raise ValueError('Solve gamma order differs from config')
        for key, shape in expected.items():
            if key not in a or a[key].shape != shape or not np.isfinite(a[key]).all():
                raise ValueError(f'Invalid solve array {key} for seed {s}; expected {shape}')
        for gi, g in enumerate(gam):
            dz = matched_dz(a['dz_if'][:, gi], a['dz_gall'][gi], K, N)
            out.setdefault(f'IF (gamma={g:g})', []).append(softmax_jvp(p0, dz))
            if float(g) == float(c['primary_gamma']):
                out.setdefault(f'IF logit (gamma={g:g})', []).append(softmax_step(p0, dz))
        out.setdefault('proxy', []).append(softmax_jvp(p0, matched_dz(a['dz_proxy'], a['dz_proxy_gall'], K, N)))
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
    preds, qual, panel = predictions(data, c)
    g0 = f"{c['primary_gamma']:g}"
    summary = dict(design=dict(subsets=J, rows=R, training_rows=N, nonmember_rows=R - N, seeds=c['seeds'], score_panel=panel,
                               gammas=c['gammas'], primary_gamma=c['primary_gamma'], primary_truth_panel=panel,
                               inference_scope='conditional on fixed interface rows and Stage-1 seed panel; reused v3 subsets'),
                   frozen=data['frozen'], missing=data['missing'], smoke=data['smoke'], gamma_qualified=qual,
                   gates={}, per_output={}, primary={}, ceilings={}, training_conditional={}, functionals={},
                   removal_effect={}, magnitude={}, bridge={}, solver={})
    tables = dict(per_output=[], functionals=[], bridge=[], magnitude=[])
    if not panel:
        return summary, tables
    panel_truth = Tseeds[[c['seeds'].index(s) for s in panel]]
    T = panel_truth.mean(0)
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
            st['p_true_only_mean'] = finite_mean(ptrue)
            st['by_class'] = {str(k): finite_mean(per_output_lds(P[:, rr, k], T[:, rr, k])) for k in range(P.shape[2])}
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
                                  passed=None if data['missing'] or e['perm_p_one_sided'] is None else bool(e['perm_p_one_sided'] < .05))
        if list(panel) != list(c['seeds']):
            summary['secondary_all_seed_truth'] = mean_lds_inference(
                flat(mean_pred[prim], group_rows['nonmember']), flat(Tseeds.mean(0), group_rows['nonmember']),
                c['permutations'], c['bootstraps'])
    else:
        summary['primary'] = dict(status=f'primary gamma {g0} unqualified; not substituted')
    # ------------------------------------------------------------- ceilings
    for grp, rr in group_rows.items():
        cols = np.arange(len(rr) * 4)
        summary['ceilings'][grp] = split_half(panel_truth[:, :, rr].reshape(len(panel_truth), J, -1), cols)
        summary['ceilings'][grp]['all_seed_truth'] = split_half(Tseeds[:, :, rr].reshape(len(Tseeds), J, -1), cols)
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
            res[status] = dict(mean_lds=finite_mean(vals), outputs=len(vals))
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
    eligible = np.flatnonzero((M.sum(0) > 0) & (M.sum(0) < J))
    def removal(X):
        if not len(eligible):
            return np.empty((0, 4))
        del_mean = np.stack([X[M[:, r] == 0, r].mean(0) for r in eligible])
        ret_mean = np.stack([X[M[:, r] == 1, r].mean(0) for r in eligible])
        return del_mean - ret_mean                                        # [N, C]
    et = removal(T)
    rel = []
    rng = np.random.default_rng(7)
    for _ in range(10 if len(panel_truth) >= 2 else 0):
        i = rng.permutation(len(panel_truth))
        a, b = removal(panel_truth[i[:len(i) // 2]].mean(0)), removal(panel_truth[i[len(i) // 2:]].mean(0))
        rel.append(spearman(a[np.arange(len(eligible)), y[eligible]], b[np.arange(len(eligible)), y[eligible]]))
    summary['removal_effect']['truth_split_half_p_true'] = finite_mean(rel)
    summary['removal_effect']['truth_mean_p_true'] = finite_mean(et[np.arange(len(eligible)), y[eligible]])
    summary['removal_effect']['eligible_rows'] = len(eligible)
    for name, P in mean_pred.items():
        ep = removal(P)
        a, b = ep[np.arange(len(eligible)), y[eligible]], et[np.arange(len(eligible)), y[eligible]]
        summary['removal_effect'][name] = dict(
            spearman_p_true=spearman(a, b), ci=None, inference='descriptive; images are not independent sampling units',
            by_class={str(k): spearman(ep[:, k], et[:, k]) for k in range(4)},
            mean_ratio=float(a.mean() / b.mean()) if len(b) and b.mean() != 0 else None)
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
    Tpanel, npanel = np.zeros(J), np.zeros(J)
    for (j, s), rows in data['v6_truth'].items():
        for r in rows:
            if r['cls'] in om:
                T6[j] += om[r['cls']] * (r['J10'] - r['J00'])
                L1v6[j] += om[r['cls']] * r['L1'][gstar]
                if s in panel:
                    Tpanel[j] += om[r['cls']] * (r['J10'] - r['J00'])
        n[j] += len(fr6['attack_seeds'])
        if s in panel:
            npanel[j] += len(fr6['attack_seeds'])
    if np.any(n == 0) or np.any(npanel == 0):
        raise ValueError('Missing v6 truth subsets in the requested seed panel')
    T6, L1v6 = T6 / n, L1v6 / n
    Tpanel /= npanel
    vb = data['vbar']
    L1 = np.mean([(vb[s][None] * data['truth'][si]).sum((1, 2)) for si, s in enumerate(seeds)], 0)
    out = dict(attack_gamma_star=fr6['attack_gamma_star'], v6_truth_mean=float(T6.mean()),
               L1_reproduces_v6_max_abs=float(np.abs(L1 - L1v6).max()), rungs={}, paired={}, v6_reproduction={},
               inference='descriptive historical same-attack-seed reproduction; no permutation significance claim',
               attack_seeds=fr6['attack_seeds'], oracle_truth_stage1_panel=panel)
    if out['L1_reproduces_v6_max_abs'] > 1e-9 * max(1., float(np.abs(L1v6).max())):
        raise ValueError(f"vbar . dp_true does not reproduce v6 L1 ({out['L1_reproduces_v6_max_abs']:.2e})")
    pi = [seeds.index(s) for s in panel]
    Tp = data['truth'][pi]
    groups = dict(deleted=lambda j: np.flatnonzero(M[j] == 0), retained=lambda j: np.flatnonzero(M[j] == 1),
                  nonmember=lambda j: np.arange(N, R))
    rung = {'L1 (all seeds)': L1,
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
        truth = T6 if name == 'L1 (all seeds)' else Tpanel
        out['rungs'][name] = dict(spearman=spearman(pred, truth), ci=None, perm_p=None,
                                  stage1_truth_panel=seeds if name == 'L1 (all seeds)' else panel)
        tables['bridge'].append(dict(rung=name, **out['rungs'][name]))
    base = f'L3 (gamma={g0})'
    if base in rung:
        for name in rung:
            if name != base and (name.startswith(base + ',') or name == 'L1 (score panel)'):
                out['paired'][f'{name} - {base}'] = paired(rung[name], rung[base], Tpanel,
                                                        draws=c['bootstraps'])
    return out
