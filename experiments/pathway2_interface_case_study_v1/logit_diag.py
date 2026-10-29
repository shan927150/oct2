"""Stage-1 diagnosis of the interface IF in logit space; CPU only, nothing is trained.

Why: in probability space the IF predicts about 0.05% of the retrained change of deleted rows. Probability
space mixes two things: the softmax saturation (dp ~ p(1-p) dz) and the size of the logit change itself.
Here the same comparison is made on relative logits u_k = log p_k - log p_y (k != true class y), which are
what the network computes and which do not saturate. Every comparison is between matched pairs of the same
subset j and the same row r (as in the interface LDS); nothing is permuted.

Three rungs, all seed-averaged over the same Stage-1 panel:
  T  retrained change  u(p_sub) - u(p_full)            (read original subset probabilities)
  L  true parameter change linearised  J (theta_j - theta_full)      (dz_true)
  I  IF  J (H + gamma I)^{-1} [(N/K_j) b_j - (N/K_j - 1) sum_i g_i]  (matched, as in the report)
Norm ratios compare amplitudes. Relative errors and Pearson correlations also measure agreement.
Zero probabilities give missing relative logits, never an artificial logit from an epsilon floor.
Rung comparisons share the same finite vectors across the seed panel and all methods.

Tables (OUT/logit_diag/):
  logit_rungs.csv    group (deleted / retained / nonmember) x margin decile: norm ratio, least-squares scale
                     relative error, within-subset Pearson, sign agreement, and missing-vector counts
  own_margin.csv     half-data contrast in logit gap against the full model's fixed competitor class
  own_margin_summary.csv  median absolute contrasts by full-model margin decile
  attack_scale.csv   same-subset diagnostic: scale the IF deleted-row part by alpha, LDS vs attack truth
  fig6_logit.png

    python logit_diag.py --repo ~/oct2-interface-lds-v1r1 --out <interface run dir>
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.dont_write_bytecode = True       # a __pycache__ in the frozen checkout would make it "dirty"
GROUPS = ('deleted', 'retained', 'nonmember')


def rho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def ratio(a, b):
    return float(a / b) if b != 0 else np.nan


def true_class_value(x, y):
    return np.take_along_axis(x, np.broadcast_to(y[:, None], x.shape[:-1] + (1,)), -1)


def rel_logit(p, y):
    """p [..., R, 4], y [R] -> u_k = log p_k - log p_y."""
    lp = np.log(p, out=np.full_like(p, np.nan, dtype=float), where=p > 0)
    return lp - true_class_value(lp, y)


def rel_tangent(dz, y):
    return dz - true_class_value(dz, y)


def drop_true(u, y):
    """[..., R, 4] -> [..., R, 3]: the components of the other classes (the true-class entry is 0 by definition)."""
    other = np.array([[k for k in range(4) if k != c] for c in range(4)])[y]
    return np.take_along_axis(u, np.broadcast_to(other, u.shape[:-1] + (3,)), -1)


def within_subset_pearson(x, t, j):
    r = []
    for k in np.unique(j):
        a, b = x[j == k].ravel(), t[j == k].ravel()
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            r.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(r)) if r else np.nan


def pair_stats(x, t, j):
    """x predicts t; rows are (subset, row) pairs of 3 relative-logit changes; j = subset of each row."""
    nz = t != 0
    return dict(n_vectors=len(t), norm_ratio=ratio(np.linalg.norm(x), np.linalg.norm(t)),
                rel_error=ratio(np.linalg.norm(x - t), np.linalg.norm(t)),
                ls_scale=ratio((x * t).sum(), (x * x).sum()), within_subset_pearson=within_subset_pearson(x, t, j),
                sign_agree=ratio(((np.sign(x) == np.sign(t)) & nz).sum(), nz.sum()),
                median_abs_true=float(np.median(np.abs(t).max(-1))) if len(t) else np.nan,
                median_abs_pred=float(np.median(np.abs(x).max(-1))) if len(t) else np.nan)


def own_effect(M, x):
    """Mean over subsets that delete the image minus mean over subsets that retain it. M [J, N], x [J, N]."""
    D = 1 - M
    return (D * x).sum(0) / D.sum(0) - (M * x).sum(0) / M.sum(0)


def margin_deciles(margin, n=10):
    margin = np.asarray(margin)
    out = np.full(len(margin), np.nan)
    ok = np.isfinite(margin)
    if ok.any():
        out[ok] = pd.qcut(margin[ok], n, labels=False, duplicates='drop')
    return out


def read_logit_truth(data, c, panel):
    """Read p_sub directly: adding p_full to a stored difference can erase tiny probabilities."""
    import analysis_if as ai
    full = np.stack([data['full'][s] for s in panel])
    subset = np.stack([np.stack([ai.v3_interface(c, f'{c["design_name"]}_s{j:03d}_seed{s}',
                                                data['rows']['raw_index'])
                                for j in range(len(data['M']))]) for s in panel])
    # Ordinary mean preserves the same seeds: a missing component in any seed stays missing.
    return full, (rel_logit(subset, data['rows']['y']) - rel_logit(full, data['rows']['y'])[:, None]).mean(0)


def attack_truth(data, panel):
    om = {int(k): v for k, v in data['fr6']['omega'].items()}
    Ta, n = np.zeros(len(data['M'])), np.zeros(len(data['M']))
    for (j, s), rows in data['v6_truth'].items():
        if s in panel:
            Ta[j] += sum(om[r['cls']] * (r['J10'] - r['J00']) for r in rows if r['cls'] in om)
            n[j] += len(data['fr6']['attack_seeds'])
    return Ta / n


def compute(data, c):
    import analysis_if as ai
    M = data['M']
    J, N = M.shape
    K = M.sum(1)
    y = data['rows']['y']
    R = len(y)
    P_all, qual, panel = ai.predictions(data, c)
    si = [c['seeds'].index(s) for s in panel]
    g0 = f"{c['primary_gamma']:g}"
    # ---- relative-logit changes, seed-averaged over the panel
    full, dT = read_logit_truth(data, c, panel)
    arr = [data['solves'][s]['arrays'] for s in panel]
    pred = {}
    for gi, g in enumerate(c['gammas']):
        if qual[f'{g:g}']:
            pred[f'IF (gamma={g:g})'] = np.mean([rel_tangent(ai.matched_dz(a['dz_if'][:, gi], a['dz_gall'][gi], K, N), y)
                                                 for a in arr], 0)
    pred['true dtheta (linear)'] = np.mean([rel_tangent(a['dz_true'], y) for a in arr], 0)
    primary = f'IF (gamma={g0})'
    DT, DP = drop_true(dT, y), {k: drop_true(v, y) for k, v in pred.items()}
    valid = np.isfinite(DT).all(-1)
    for v in DP.values():
        valid &= np.isfinite(v).all(-1)
    # ---- margin of the full model (log p_y - log p of the strongest other class) and its deciles per row type
    p0 = full.mean(0)
    lp0 = np.log(p0, out=np.full_like(p0, -np.inf), where=p0 > 0)
    other = np.where(np.eye(4, dtype=bool)[y], -np.inf, lp0)
    kstar = other.argmax(1)
    margin = lp0[np.arange(R), y] - other.max(1)
    decile = np.r_[margin_deciles(margin[:N]), margin_deciles(margin[N:])]
    # ---- masks [J, R]
    pad = np.zeros((J, R - N), bool)
    masks = dict(deleted=np.c_[M == 0, pad], retained=np.c_[M == 1, pad], nonmember=np.tile(np.arange(R) >= N, (J, 1)))
    # ---- 1. rungs by group and margin decile
    pairs = [(primary, 'truth'), ('true dtheta (linear)', 'truth'), (primary, 'true dtheta (linear)')]
    rows = []
    for g, m in masks.items():
        for d in [-1] + sorted(set(decile[m.any(0) & np.isfinite(decile)])):
            selected = m & ((decile == d)[None, :] if d >= 0 else True)
            sel = selected & valid
            j = np.nonzero(sel)[0]
            ref = {'truth': DT[sel], 'true dtheta (linear)': DP['true dtheta (linear)'][sel]}
            for a, b in pairs:
                rows.append(dict(group=g, margin_decile=d if d >= 0 else 'all', pred=a, ref=b,
                                 n_total_vectors=int(selected.sum()), n_missing_vectors=int((selected & ~valid).sum()),
                                 median_margin=float(np.median(margin[np.nonzero(sel)[1]])) if sel.any() else np.nan,
                                 **pair_stats(DP[a][sel], ref[b], j)))
    for name in DP:                                       # every damping against the truth, all rows of the group
        if name not in (primary, 'true dtheta (linear)'):
            for g, m in masks.items():
                sel = m & valid
                rows.append(dict(group=g, margin_decile='all', pred=name, ref='truth', median_margin=np.nan,
                                 n_total_vectors=int(m.sum()), n_missing_vectors=int((m & ~valid).sum()),
                                 **pair_stats(DP[name][sel], DT[sel], np.nonzero(sel)[0])))
    rungs = pd.DataFrame(rows)
    # ---- 2. own effect of every training image on its margin (deleted minus retained, half-data)
    ar = np.arange(N)
    dm = lambda d: -np.take_along_axis(d[:, :N], np.broadcast_to(kstar[None, :N, None], (J, N, 1)), -1)[..., 0]   # change of the margin
    own = pd.DataFrame(dict(train_position=ar, raw_index=data['rows']['raw_index'][:N], cls=y[:N],
                            competitor_cls=kstar[:N], margin=margin[:N], decile=decile[:N],
                            own_true=own_effect(M, dm(dT)), own_linear=own_effect(M, dm(pred['true dtheta (linear)'])),
                            own_if=own_effect(M, dm(pred[primary]))))
    own['valid_contrast'] = np.isfinite(own[['own_true', 'own_linear', 'own_if']]).all(axis=1)
    vb = np.stack([data['vbar'][s] for s in panel])
    own['attack_row_active'] = (vb[:, :N] != 0).any(-1).any(0)
    # ---- 3. bridge: how far does rescaling the IF deleted-row part go from row 1 to row 3
    Ta = attack_truth(data, panel)
    T = data['truth'][si]
    P = P_all[primary]
    per_vector = lambda X: np.einsum('srk,sjrk->jr', vb, X) / len(panel)         # vbar . dp of every (subset, row)
    tv, pv = per_vector(T), per_vector(P)
    part = lambda v, g: (v * masks[g]).sum(1)
    S_if = {g: part(pv, g) for g in GROUPS}
    S_true = {g: part(tv, g) for g in GROUPS}
    rest = S_if['retained'] + S_if['nonmember']
    alpha_ls = ratio((pv * tv)[masks['deleted']].sum(), (pv * pv)[masks['deleted']].sum())
    alphas = np.logspace(0, 6, 25)
    sweep = [dict(label='IF deleted part x alpha', alpha=a, lds=rho(a * S_if['deleted'] + rest, Ta)) for a in alphas]
    sweep += [dict(label='row 1: all IF', alpha=1., lds=rho(S_if['deleted'] + rest, Ta)),
              dict(label='row 3: deleted rows <- retrained', alpha=np.nan, lds=rho(S_true['deleted'] + rest, Ta)),
              dict(label=f'IF deleted part x alpha_LS = {alpha_ls:.3g}', alpha=alpha_ls, lds=rho(alpha_ls * S_if['deleted'] + rest, Ta)),
              dict(label='IF deleted part alone (scale free)', alpha=np.nan, lds=rho(S_if['deleted'], Ta)),
              dict(label='retrained deleted part alone', alpha=np.nan, lds=rho(S_true['deleted'], Ta)),
              dict(label='IF retained + nonmember part alone', alpha=np.nan, lds=rho(rest, Ta))]
    return rungs, own, pd.DataFrame(sweep).assign(evaluation='same-subset diagnostic'), panel, primary


def own_summary(own):
    """Half-data contrasts against the fixed competitor, summarized on common finite images."""
    out = []
    for d, g in own.groupby('decile'):
        total = len(g)
        g = g[g.valid_contrast]
        out.append(dict(decile=int(d), n=len(g), n_total=total, n_missing=total - len(g), median_margin=g.margin.median(),
                        median_abs_true=g.own_true.abs().median(), median_abs_linear=g.own_linear.abs().median(),
                        median_abs_if=g.own_if.abs().median(),
                        spearman_if_true=rho(g.own_if, g.own_true), spearman_linear_true=rho(g.own_linear, g.own_true)))
    return pd.DataFrame(out, columns=['decile', 'n', 'n_total', 'n_missing', 'median_margin',
                                     'median_abs_true', 'median_abs_linear', 'median_abs_if',
                                     'spearman_if_true', 'spearman_linear_true'])


def figures(rungs, own, sweep, primary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summ = own_summary(own)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for col, lab in (('median_abs_true', 'retrained'), ('median_abs_linear', 'true dtheta, linearised'), ('median_abs_if', 'IF')):
        ax[0].plot(summ.median_margin, summ[col], 'o-', label=lab)
    ax[0].set_yscale('log')
    ax[0].set(xlabel='full-model margin (median per decile)', ylabel='median |deleted-minus-retained contrast|',
              title='Logit gap to the fixed competitor')
    ax[0].legend()
    d = rungs[(rungs.group == 'deleted') & (rungs.margin_decile != 'all')]
    for a, b, lab in ((primary, 'truth', 'IF / retrained'), ('true dtheta (linear)', 'truth', 'linearised / retrained'),
                      (primary, 'true dtheta (linear)', 'IF / linearised')):
        s = d[(d.pred == a) & (d.ref == b)].sort_values('median_margin')
        ax[1].plot(s.median_margin, s.norm_ratio, 'o-', label=lab)
    ax[1].set_yscale('log')
    ax[1].set(xlabel='full-model margin (median per decile)', ylabel='norm ratio', title='Relative-logit amplitudes: deleted rows')
    ax[1].legend()
    s = sweep[sweep.label == 'IF deleted part x alpha']
    ax[2].semilogx(s.alpha, s.lds, 'o-')
    for lab, ls in (('row 1: all IF', ':'), ('row 3: deleted rows <- retrained', '--')):
        ax[2].axhline(float(sweep[sweep.label == lab].lds.iloc[0]), ls=ls, color='gray', label=lab)
    ax[2].set(xlabel='alpha (multiplier of the IF deleted-row part)', ylabel='LDS vs attack truth',
              title='Scaling diagnostic on the same subsets')
    ax[2].legend(fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


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
    rungs, own, sweep, panel, primary = compute(data, c)
    summ = own_summary(own)
    dest = Path(c['out']) / 'logit_diag'
    dest.mkdir(exist_ok=True)
    rungs.to_csv(dest / 'logit_rungs.csv', index=False)
    own.to_csv(dest / 'own_margin.csv', index=False)
    summ.to_csv(dest / 'own_margin_summary.csv', index=False)
    sweep.to_csv(dest / 'attack_scale.csv', index=False)
    figures(rungs, own, sweep, primary, dest / 'fig6_logit.png')
    show = lambda df: print(df.to_string(index=False, float_format=lambda x: f'{x:.6g}'), '\n')
    print(f'seed panel {panel}, primary {primary}')
    print('--- deleted / retained / nonmember, all margins (relative logits)')
    show(rungs[(rungs.margin_decile == 'all')].drop(columns=['margin_decile', 'median_margin']))
    print('--- deleted rows by margin decile')
    show(rungs[(rungs.group == 'deleted') & (rungs.margin_decile != 'all')].drop(columns=['group']))
    print('--- half-data contrast against the fixed competitor, by full-model margin decile')
    show(summ)
    print('--- bridge: same-subset scaling diagnostic (reported: row 1 = 0.262, row 3 = 0.730)')
    show(sweep[sweep.label != 'IF deleted part x alpha'])
    print('Saved:', dest)


if __name__ == '__main__':
    main()
