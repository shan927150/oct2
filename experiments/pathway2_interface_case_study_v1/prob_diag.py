"""Probability-level diagnostics of the interface IF; CPU only, nothing is trained.

Everything compares the prediction of subset j with the retrained change of the same subset j (matched pairs,
as in the interface LDS). Nothing is permuted. Rows are split three ways per subset:
deleted / retained (training rows) and nonmember, and by the full-model confidence p0,y of the row's true class.

  bounds.csv         is the recovered probability p0 + dp_hat inside [0, 1]?  Per seed, group, confidence bin.
                     Violation = amount outside [0, 1]; fractions at 1e-12 (float noise), 1e-6, 1e-3 (substantive).
  cells.csv          per method x group x confidence bin: magnitude, Pearson and component sign agreement of dp.
                     Also reports the attack-weighted cell score: mean, RMSE and Spearman over subsets.
                     These aggregated scores are not per-output interface LDS. Deleted/retained cell membership
                     changes across subsets, so their score correlations can reflect class composition.
  logit_comparison.csv compares linear IF and the same logit tangent passed through softmax, using identical
                     vectors in each comparison. Includes the fraction of total squared error on violating
                     vectors. This changes the softmax approximation, not only out-of-bounds probabilities.
  reference.csv      nonmember per-output interface LDS and full-score attack LDS, to reproduce the report.

Confidence bins use the seed-panel mean p0,y, keeping the same rows in each bin for all methods and seeds.
Error statistics in cells.csv use seed means. Bounds and logit_comparison.csv retain individual seeds.
Main interpretation uses the frozen primary gamma; other qualified gammas are supplementary. No LOO is run.

    python prob_diag.py --repo ~/oct2-interface-lds-v1r1 --out <interface run dir>
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.dont_write_bytecode = True       # a __pycache__ in the frozen checkout would make it "dirty"

EDGES = [.5, .9, .99, .999, .9999]
BINS = ['<0.5', '0.5-0.9', '0.9-0.99', '0.99-0.999', '0.999-0.9999', '>=0.9999']
THR = (1e-12, 1e-6, 1e-3)
SIGN_TOL = 1e-6                    # only interpret signs of true probability changes larger than this


def rho(a, b):
    return float(spearmanr(a, b)[0])


def ratio(a, b):
    """A zero denominator makes a relative metric undefined, not zero."""
    return float(a / b) if b != 0 else np.nan


def violation(q):
    """q [..., 4] -> amount by which the worst component lies outside [0, 1] (0 if inside)."""
    return np.maximum(np.maximum(-q, q - 1), 0).max(-1)


def cell_masks(M, y, p0y):
    """{(group, confidence bin): bool [J, R]}. Per subset the three groups partition all rows."""
    J, N = M.shape
    R = len(y)
    pad = np.zeros((J, R - N), bool)
    groups = dict(deleted=np.c_[M == 0, pad], retained=np.c_[M == 1, pad],
                  nonmember=np.tile(np.arange(R) >= N, (J, 1)), all=np.ones((J, R), bool))
    b = np.digitize(p0y, EDGES)
    out = {}
    for g, m in groups.items():
        out[(g, 'all')] = m
        for k, lab in enumerate(BINS):
            out[(g, lab)] = m & (b == k)[None, :]
    return out


def bounds(masks, p0, preds, panel):
    """p0 [S, R, 4]; preds {method: dp_hat [S, J, R, 4]}. Per seed, so that averaging cannot hide a seed."""
    rows = []
    for method, P in preds.items():
        for si, seed in enumerate(panel):
            q = p0[si][None] + P[si]
            v, dev = violation(q), np.abs(q.sum(-1) - 1)
            for (g, b), m in masks.items():
                if m.any():
                    rows.append(dict(method=method, seed=seed, group=g, conf_bin=b, n_vectors=int(m.sum()),
                                     **{f'oob_gt_{t:g}': float((v[m] > t).mean()) for t in THR},
                                     max_violation=float(v[m].max()), max_sum_dev=float(dev[m].max())))
    return pd.DataFrame(rows)


def within_subset_pearson(p, t, j):
    r = np.array([np.corrcoef(p[j == k].ravel(), t[j == k].ravel())[0, 1] for k in np.unique(j)])
    ok = np.isfinite(r)
    return ratio(r[ok].sum(), ok.sum()), int(ok.sum())


def cell_stats(masks, preds, T, vb, Ta):
    """preds {method: [S, J, R, 4]}; T [S, J, R, 4] retrained change; vb [S, R, 4]; Ta [J] attack truth.
    dp statistics use the seed-averaged change (as the interface LDS); cell scores contract with vbar per seed."""
    J = T.shape[1]
    Tm = T.mean(0)
    act = (vb != 0).any(-1).any(0)
    score = lambda X, m: np.mean([(vb[s] * X[s] * m[..., None]).sum((1, 2)) for s in range(len(X))], 0)
    pm = {k: v.mean(0) for k, v in preds.items()}
    rows = []
    for (g, b), m in masks.items():
        if not m.any():
            continue
        t, j = Tm[m], np.broadcast_to(np.arange(J)[:, None], m.shape)[m]
        direction = np.abs(t) > SIGN_TOL
        st = score(T, m)
        for name, P in preds.items():
            p = pm[name][m]
            same = np.sign(p) == np.sign(t)
            sp = score(P, m)
            within, n_within = within_subset_pearson(p, t, j)
            rows.append(dict(method=name, group=g, conf_bin=b, n_vectors=int(m.sum()),
                             n_active_rows_per_subset=float((m & act).sum(1).mean()),
                             norm_ratio=ratio(np.linalg.norm(p), np.linalg.norm(t)),
                             pooled_pearson=np.corrcoef(p.ravel(), t.ravel())[0, 1],
                             within_subset_pearson=within, n_pearson_subsets=n_within,
                             sign_agree=ratio(same[direction].sum(), direction.sum()), n_sign_components=int(direction.sum()),
                             rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                             rel_error=ratio(np.linalg.norm(p - t), np.linalg.norm(t)),
                             mean_true_score=st.mean(), mean_pred_score=sp.mean(),
                             rmse_score_error=np.sqrt(np.mean((st - sp) ** 2)),
                             score_spearman_pred_vs_true=rho(sp, st), score_spearman_pred_vs_attack=rho(sp, Ta),
                             score_spearman_true_vs_attack=rho(st, Ta)))
    return pd.DataFrame(rows)


def logit_comparison(masks, p0, T, P, Pl, panel, thr=1e-6):
    """Errors pooled over seeds, split using LINEAR IF's violations for both methods.
    Squared-error shares answer whether rare violating vectors account for much of the group's error."""
    acc = {}
    for si in range(len(panel)):
        out = violation(p0[si][None] + P[si]) > thr
        for g in ('deleted', 'retained', 'nonmember'):
            m = masks[(g, 'all')]
            for part, sel in (('all', m), ('out_of_bounds', m & out), ('within_tolerance', m & ~out)):
                a = acc.setdefault((g, part), np.zeros(4))
                a += [sel.sum(), ((P[si] - T[si])[sel] ** 2).sum(), ((Pl[si] - T[si])[sel] ** 2).sum(), (T[si][sel] ** 2).sum()]
    return pd.DataFrame([dict(group=g, part=part, n_vectors=int(n),
                              fraction_vectors=ratio(n, acc[(g, 'all')][0]),
                              squared_error_share_linear=ratio(el, acc[(g, 'all')][1]),
                              rmse_linear=np.sqrt(el / (n * T.shape[-1])), rmse_logit=np.sqrt(eg / (n * T.shape[-1])),
                              rel_error_linear=np.sqrt(ratio(el, d)), rel_error_logit=np.sqrt(ratio(eg, d)))
                         for (g, part), (n, el, eg, d) in acc.items() if n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='interface run directory')
    ap.add_argument('--repo', type=Path, default=Path.cwd(), help='unchanged interface v1r1 repository checkout')
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / 'experiments/pathway2_interface_lds_v1'))
    import analysis_if
    import run
    c = run.config(a.out)
    data = analysis_if.load(Path(c['out']), c)
    P_all, qualified, panel = analysis_if.predictions(data, c)
    g0 = f"{c['primary_gamma']:g}"
    if not qualified[g0]:
        raise ValueError(f'Primary gamma {g0} did not pass the existing solve checks.')
    methods = {f'IF (gamma={g:g})': P_all[f'IF (gamma={g:g})'] for g in c['gammas'] if qualified[f'{g:g}']}
    methods['true dtheta (linear)'] = P_all['true dtheta (linear)']
    methods[f'IF logit (gamma={g0})'] = P_all[f'IF logit (gamma={g0})']
    T = data['truth'][[c['seeds'].index(s) for s in panel]]
    vb = np.stack([data['vbar'][s] for s in panel])
    p0 = np.stack([data['solves'][s]['arrays']['p0'] for s in panel])
    M, N = data['M'], data['M'].shape[1]
    y = data['rows']['y']
    R = len(y)
    masks = cell_masks(M, y, p0.mean(0)[np.arange(R), y])
    # attack truth over the 32 subsets: same seeds and weights as the bridge table
    om = {int(k): v for k, v in data['fr6']['omega'].items()}
    Ta, n = np.zeros(len(M)), np.zeros(len(M))
    for (j, s), rows in data['v6_truth'].items():
        if s in panel:
            Ta[j] += sum(om[r['cls']] * (r['J10'] - r['J00']) for r in rows if r['cls'] in om)
            n[j] += len(data['fr6']['attack_seeds'])
    Ta /= n
    b = bounds(masks, p0, methods, panel)
    cells = cell_stats(masks, methods, T, vb, Ta)
    fix = logit_comparison(masks, p0, T, methods[f'IF (gamma={g0})'], methods[f'IF logit (gamma={g0})'], panel)
    refs = []
    for name, P in methods.items():
        values = analysis_if.per_output_lds(analysis_if.flat(P.mean(0), np.arange(N, R)),
                                           analysis_if.flat(T.mean(0), np.arange(N, R)))
        ok = np.isfinite(values)
        full_score = np.einsum('src,sjrc->sj', vb, P).mean(0)
        refs.append(dict(method=name, interface_lds_nonmember=ratio(values[ok].sum(), ok.sum()),
                         n_valid_outputs=int(ok.sum()), n_total_outputs=len(values),
                         attack_lds=rho(full_score, Ta), mean_pred_score=full_score.mean(), mean_attack_truth=Ta.mean()))
    reference = pd.DataFrame(refs)
    dest = Path(c['out']) / 'prob_diag'
    dest.mkdir(exist_ok=True)
    for name, df in dict(bounds=b, cells=cells, logit_comparison=fix, reference=reference).items():
        df.to_csv(dest / f'{name}.csv', index=False)
    # ---- numbers to compare with the report, then the diagnostics at the primary gamma
    pr = f'IF (gamma={g0})'
    print(f'seed panel {panel}, primary gamma {g0}, sign threshold {SIGN_TOL:g}, logit comparison OOB threshold 1e-6')
    show = lambda df: print(df.to_string(index=False, float_format=lambda x: f'{x:.6g}'), '\n')
    print('--- reference metrics (nonmember interface LDS about 0.227 / 0.228; primary attack LDS about 0.262)')
    show(reference)
    print('--- bounds, all confidence bins (per seed)')
    show(b[(b.conf_bin == 'all') & (b.group != 'all') & b.method.isin([pr, 'true dtheta (linear)'])]
         .drop(columns=['conf_bin']))
    print('--- bounds by confidence bin, IF, vectors outside by more than 1e-3 (per seed)')
    show(b[(b.method == pr) & (b.group != 'all') & (b.conf_bin != 'all')]
         .pivot_table(index=['group', 'conf_bin'], columns='seed', values='oob_gt_0.001').reset_index())
    c0 = cells[cells.method.isin([pr, f'IF logit (gamma={g0})', 'true dtheta (linear)'])]
    print('--- groups (reports: norm ratio 0.00049 / 0.22 / 0.047, within-subset Pearson 0.41 / -0.15 / 0.15)')
    show(c0[c0.conf_bin == 'all'].drop(columns=['conf_bin', 'n_active_rows_per_subset']))
    print('--- IF by confidence bin')
    show(cells[(cells.method == pr) & (cells.group != 'all') & (cells.conf_bin != 'all')]
         .drop(columns=['method']))
    print('--- logit step vs linear')
    show(fix)
    print('Saved:', dest)


if __name__ == '__main__':
    main()
