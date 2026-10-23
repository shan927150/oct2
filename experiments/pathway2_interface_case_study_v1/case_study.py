"""Case study on the finished interface run; CPU only, nothing is trained.

Two bridge scores, each correlated (Spearman, i.e. LDS) with the v6 attack truth over the 32 subsets:
  row 1  L3                         IF-predicted interface change on every row
  row 3  L3, deleted rows <- truth  same, but the deleted training rows use the retrained change
Both scores are sums over rows of  vbar * dp, so row 3 - row 1 is a sum over deleted images. share[j, i] is
what replacing image i's own row adds to subset j's score: the direct explanation of row 1 -> row 3.
Correlation gains are not additive.

Image-level quantities, which answer different questions:
  share / gap_share   score correction from replacing the image's own row (row 1 -> row 3)
  a_if, a_true        deleted-minus-retained change of the image's own row (half-data), predicted vs retrained,
                      weighted by vbar; a descriptive cross-image correlation, not leave-one-out
  tau                 predicted whole-path contribution of the image to row 1 (a prediction, not retraining)
vbar is zero on rows of classes outside the attack objective (only DME and DRUSEN are used), so the own-row
quantities are restricted to active rows. tau, p_true and the confidence bins use all training images.
Everything uses the same Stage-1 seed panel as the report. No permutation test or bootstrap is run.

Keep this script outside the frozen experiment checkout (run.config requires that checkout to stay clean):
    python ~/case_study.py --repo ~/oct2-interface-lds-v1r1 --out <interface run dir> [--gamma 0.03]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.dont_write_bytecode = True       # a __pycache__ in the frozen checkout would make it "dirty"


def rho(a, b):
    return float(spearmanr(a, b)[0])


def own_effect(M, X):
    """Mean change when the image is deleted minus mean change when it is retained.
    M [J, N] (1 = retained), X [S, J, N, C] -> [S, N, C]."""
    D = 1 - M
    return (np.einsum('jn,sjnc->snc', D, X) / D.sum(0)[None, :, None]
            - np.einsum('jn,sjnc->snc', M, X) / M.sum(0)[None, :, None])


def deleted_row_correction(M, T, P, vb):
    """[J, N]: for every subset, each deleted image's share of (row 3 score - row 1 score)."""
    N = M.shape[1]
    return ((T - P)[:, :, :N] * vb[:, None, :N]).sum(-1).mean(0) * (1 - M)


def case_study(M, y, p0, T, P, vb, Ta, tau, k=20):
    """M [J, N] retained mask; y [N]; p0 [N, 4] full-model probabilities of the training rows (seed mean);
    T, P [S, J, R, 4] true / IF-predicted change; vb [S, R, 4] attack adjoint; Ta [J] attack truth;
    tau [N] seed-averaged whole-path UPWEIGHT score. The first N interface rows are the training rows."""
    J, N = M.shape
    # ---- subset level: the two bridge scores and their gap
    score1 = (vb[:, None] * P).sum((2, 3)).mean(0)                         # [J]
    share = deleted_row_correction(M, T, P, vb)                           # [J, N]
    score3 = score1 + share.sum(1)
    # ---- image level
    et, ep = own_effect(M, T[:, :, :N]), own_effect(M, P[:, :, :N])
    a_true = (vb[:, :N] * et).sum(-1).mean(0)                            # own-row attack-weighted contrast
    a_if = (vb[:, :N] * ep).sum(-1).mean(0)
    d = share.sum(0) / (1 - M).sum(0)                                       # mean share of the row1 -> row3 gap
    ar = np.arange(N)
    pt = p0[ar, y]
    logp = np.log(p0)
    other = np.where(np.eye(4, dtype=bool)[y], -np.inf, logp).max(1)
    e_t, e_p = et.mean(0)[ar, y], ep.mean(0)[ar, y]                       # true-class own contrast
    active = np.any(vb[:, :N] != 0, axis=(0, 2))                          # rows the attack objective uses
    img = pd.DataFrame(dict(train_position=ar, cls=y, p_true=pt, loss=-logp[ar, y], margin=logp[ar, y] - other,
                            attack_row_active=active, n_deleted=(1 - M).sum(0).astype(int),
                            a_true=a_true, a_if=a_if, gap_share=d, own_delta_py_true=e_t, own_delta_py_if=e_p,
                            tau_upweight=tau, removal_score=-tau))
    key = dict(lds_row1=rho(score1, Ta), lds_row3=rho(score3, Ta),
               own_effect_spearman_true_class=rho(e_p, e_t),
               own_effect_spearman_attack_active=rho(a_if[active], a_true[active]))
    # ---- groups of images: own-row quantities over active rows, tau over all training images
    act = np.flatnonzero(active)
    order_act = lambda v: act[np.argsort(-np.abs(v[act]))]
    order_all = lambda v: np.argsort(-np.abs(v))
    groups = {'IF own largest absolute': order_act(a_if)[:k], 'IF own smallest absolute': order_act(a_if)[::-1][:k],
              'true own largest absolute': order_act(a_true)[:k], 'true own smallest absolute': order_act(a_true)[::-1][:k],
              'largest absolute mean deleted-row correction': order_act(d)[:k],
              'all attack-active rows': act,
              'L3 whole-path largest absolute': order_all(tau)[:k], 'L3 whole-path smallest absolute': order_all(tau)[::-1][:k],
              'all training images': ar}
    memberships = [[] for _ in ar]
    rows = []
    for name, idx in groups.items():
        for i in idx:
            memberships[i].append(name)
        g = img.iloc[idx]
        rows.append(dict(group=name, n=len(idx), **{f'frac_cls{c}': float((g.cls == c).mean()) for c in range(4)},
                         p_true=g.p_true.mean(), loss=g.loss.mean(), margin=g.margin.mean(),
                         mean_abs_a_true=g.a_true.abs().mean(), mean_abs_a_if=g.a_if.abs().mean(),
                         mean_abs_gap_share=g.gap_share.abs().mean(), mean_abs_removal_score=g.removal_score.abs().mean()))
    img['groups'] = ['; '.join(names) for names in memberships]
    # ---- by confidence of the full model on the true class
    bins = pd.cut(img.p_true, [0, .9, .99, .999, 1.0001], right=False,
                  labels=['<0.9', '0.9 to <0.99', '0.99 to <0.999', '>=0.999'])
    sat = []
    for b, g in img.groupby(bins, observed=True):
        i = g.index.values
        sat.append(dict(p_true_bin=str(b), n=len(i), true_effect=e_t[i].mean(), if_effect=e_p[i].mean(),
                        norm_ratio=np.linalg.norm(e_p[i]) / np.linalg.norm(e_t[i]), spearman=rho(e_p[i], e_t[i])))
    # ---- subset level: high confidence is a measured feature, not proof of memorisation
    sub = pd.DataFrame(dict(subset=np.arange(J), attack_truth=Ta, row1=score1, row3=score3, score_gap=score3 - score1,
                           deleted_high_conf=((1 - M) * (pt > .99)).sum(1),
                           deleted_high_conf_active=((1 - M) * ((pt > .99) & active)).sum(1)))
    for col in ('attack_truth', 'row1', 'row3'):
        sub['rank_' + col] = sub[col].rank(ascending=False)
    for cls in range(4):
        sub[f'deleted_cls{cls}'] = (1 - M[:, y == cls]).sum(1).astype(int)
    for f in ('deleted_high_conf', 'deleted_high_conf_active'):
        key.update({f'rho_{f}_vs_{c}': rho(sub[f], sub[c]) for c in ('attack_truth', 'row1', 'row3')})
    return key, img, pd.DataFrame(rows), pd.DataFrame(sat), sub


def probability_bounds(M, p0, predictions, panel, tol=1e-12):
    """p0 [S, R, 4]; predictions {name: change [S, J, R, 4]}. One point is one four-class vector p0 + dp."""
    J, N = M.shape
    R = p0.shape[1]
    pad = np.zeros((J, R - N), bool)
    masks = dict(deleted=np.c_[M == 0, pad], retained=np.c_[M == 1, pad], nonmember=np.tile(np.arange(R) >= N, (J, 1)))
    rows = []
    for method, dp in predictions.items():
        q = p0[:, None] + dp
        for si, seed in enumerate(panel):
            for group, mask in masks.items():
                values = q[si][mask]
                violation = np.maximum(np.maximum(-values, values - 1), 0).max(1)    # amount outside [0, 1]
                rows.append(dict(method=method, seed=seed, group=group, n_vectors=len(values),
                                 oob_fraction=float((violation > tol).mean()), max_violation=float(violation.max())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='interface run directory')
    ap.add_argument('--gamma', type=float, default=None)
    ap.add_argument('--repo', type=Path, default=Path.cwd(), help='unchanged interface v1r1 repository checkout')
    ap.add_argument('--top-k', type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / 'experiments/pathway2_interface_lds_v1'))
    import analysis_if
    import run
    c = run.config(a.out)
    data = analysis_if.load(Path(c['out']), c)
    preds, qualified, panel = analysis_if.predictions(data, c)
    g = c['primary_gamma'] if a.gamma is None else a.gamma
    if not qualified.get(f'{g:g}', False):
        raise ValueError(f'Gamma {g:g} is unavailable or did not pass the existing solve checks.')
    P = preds[f'IF (gamma={g:g})']
    T = data['truth'][[c['seeds'].index(s) for s in panel]]
    vb = np.stack([data['vbar'][s] for s in panel])
    p0_seeds = np.stack([data['solves'][s]['arrays']['p0'] for s in panel])
    M, N = data['M'], data['M'].shape[1]
    # attack truth over the 32 subsets: same seeds and weights as the bridge table
    om = {int(k): v for k, v in data['fr6']['omega'].items()}
    Ta, n = np.zeros(len(M)), np.zeros(len(M))
    for (j, s), rows in data['v6_truth'].items():
        if s in panel:
            Ta[j] += sum(om[r['cls']] * (r['J10'] - r['J00']) for r in rows if r['cls'] in om)
            n[j] += len(data['fr6']['attack_seeds'])
    gi = c['gammas'].index(g)
    tau = np.mean([data['solves'][s]['arrays']['tau_hbar'][gi] for s in panel], 0)
    key, img, grp, sat, sub = case_study(M, data['rows']['y'][:N], p0_seeds.mean(0)[:N], T, P, vb, Ta / n, tau, a.top_k)
    # v6 identity: row 1 = sum_i ((N / K_j) M_ji - 1) tau_i. A consistency check of the saved tau (sign and scale),
    # not a comparison with retraining; no tolerance is assumed, read the number.
    via_tau = ((N / M.sum(1))[:, None] * M - 1) @ tau
    key['row1_vs_tau_identity_max_rel_err'] = float(np.abs(sub.row1 - via_tau).max() / np.abs(via_tau).max())
    img.insert(0, 'raw_index', data['rows']['raw_index'][:N])
    js, positions = np.nonzero(M == 0)
    contributions = pd.DataFrame(dict(subset=js, train_position=positions, raw_index=data['rows']['raw_index'][positions],
                                      score_correction=deleted_row_correction(M, T, P, vb)[js, positions]))
    bounds = probability_bounds(M, p0_seeds, {'IF': P, 'true_dtheta_linear': preds['true dtheta (linear)']}, panel)
    dest = Path(c['out']) / 'case_study' / f'gamma{g:g}_k{a.top_k}'
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in dict(images=img, groups=grp, p_true_bins=sat, subsets=sub, probability_bounds=bounds,
                         deleted_row_contributions=contributions).items():
        df.to_csv(dest / f'{name}.csv', index=False)
    print(f'gamma = {g:g}, seed panel {panel}')
    for k_, v in key.items():
        print(f'{k_:45s} {v:+.3g}')
    for df in (grp, sat):
        print(df.round(4).to_string(index=False), '\n')
    print('Saved:', dest)


if __name__ == '__main__':
    main()
