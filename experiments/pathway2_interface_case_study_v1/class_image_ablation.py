"""Direct ablations on the finished interface run; CPU only, nothing is trained.

Question: which attack class, and which images, limit the bridge LDS? The bridge score of subset j is a sum over
rows r of  vbar_r . dp_{j,r}.  Because it is a plain sum, any variant (one class, a set of images removed, a set
of images replaced by the retrained change) is computed directly from the stored per-(subset, row) terms.

E1  per attack class      the class's own sensitivity v_c (stored per class in v6's v.npz, mean over attack seeds,
                          weighted by omega_c) gives the class score; it is set against the class truth (mean over
                          Stage-1 panel seeds and attack seeds of J10 - J00 of that class). The combined target is the
                          original one. Rows 1 to 5 of the bridge, plus the deleted part alone and the rest alone.
                          The mixed pairs (class score, other truth) are in the csv only.
E2  image sets            for sets of training images chosen WITHOUT the retrained values (lowest margin, margin
                          below a threshold, largest |a_IF|, smallest gradient norm, a class) and for the oracle
                          set (largest |a_true|, chosen with the truth: optimistic), against random sets of the same size:
                            drop   the images' rows are left out of the score (rows 1, 2, 3), the truth is unchanged
                            fix    only these images' deleted rows take the retrained change (row 1 -> fix)
                            leave  every deleted row but these takes the retrained change (row 3 -> leave)
Random sets: 300 draws, seed 20261030, same size, from the attack-active training images.
Spearman over the 32 subsets (LDS). Every number is a same-subset diagnostic, not a new estimate.

Keep this script outside the frozen experiment checkout (run.config requires that checkout to stay clean):
    python class_image_ablation.py --repo ~/oct2-interface-lds-v1r1 --out <interface run dir> \
        [--gamma 0.03] [--features <out>/case_study/gamma0.03_k100/features_merged.csv]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.dont_write_bytecode = True       # a __pycache__ in the frozen checkout would make it "dirty"
SIZES = (20, 50, 100, 200, 300)
THRESHOLDS = (8, 10, 12)
DRAWS = 300
SEED = 20261030


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def own_effect(M, X):
    """Mean change when the image is deleted minus mean change when it is retained. M [J, N], X [S, J, N, C] -> [S, N, C]."""
    D = 1 - M
    return (np.einsum('jn,sjnc->snc', D, X) / D.sum(0)[None, :, None]
            - np.einsum('jn,sjnc->snc', M, X) / M.sum(0)[None, :, None])


def class_truth(v6_truth, panel, omega):
    """{class: [J]} mean over panel seeds and attack seeds of J10 - J00 of that attack class."""
    J = 1 + max(j for j, _ in v6_truth)
    num = {c: np.zeros(J) for c in omega}
    cnt = {c: np.zeros(J) for c in omega}
    for (j, s), rows in v6_truth.items():
        if s in panel:
            for r in rows:
                if r['cls'] in omega:
                    num[r['cls']][j] += r['J10'] - r['J00']
                    cnt[r['cls']][j] += 1
    return {c: num[c] / cnt[c] for c in omega}


def attack_truth(v6_truth, panel, omega, n_attack_seeds):
    """The combined truth of the bridge table (same expression as in the case study and logit diagnostics)."""
    Ta, n = np.zeros(1 + max(j for j, _ in v6_truth)), np.zeros(1 + max(j for j, _ in v6_truth))
    for (j, s), rows in v6_truth.items():
        if s in panel:
            Ta[j] += sum(omega[r['cls']] * (r['J10'] - r['J00']) for r in rows if r['cls'] in omega)
            n[j] += n_attack_seeds
    return Ta / n


def terms(vb, X):
    """vbar . dp of every (subset, row), averaged over the Stage-1 panel. vb [S, R, 4], X [S, J, R, 4] -> [J, R]."""
    return np.einsum('srk,sjrk->jr', vb, X) / len(vb)


class Bridge:
    """pv, tv [J, R]: predicted and retrained terms; M [J, N] retained mask of the training rows."""

    def __init__(self, pv, tv, M, y):
        self.pv, self.tv = pv, tv
        J, N = M.shape
        R = pv.shape[1]
        self.J, self.N, self.R, self.y = J, N, R, y
        pad = np.zeros((J, R - N), bool)
        self.deleted = np.c_[M == 0, pad]
        self.retained = np.c_[M == 1, pad]
        self.nonmember = np.tile(np.arange(R) >= N, (J, 1))

    def score(self, true_mask=None, keep=None, rows=None):
        """Sum over rows of the retrained term where true_mask, else the predicted term. keep [J, R] drops rows."""
        x = self.pv if true_mask is None else np.where(true_mask, self.tv, self.pv)
        if keep is not None:
            x = x * keep
        if rows is not None:
            x = x[:, rows]
        return x.sum(1)

    def bridge_rows(self, rows=None):
        t = lambda m: self.score(m, rows=rows)
        return {'row 1 all IF': t(None), 'row 2 all retrained': t(np.ones_like(self.deleted)),
                'row 3 deleted rows retrained': t(self.deleted), 'row 4 retained rows retrained': t(self.retained),
                'row 5 nonmember rows retrained': t(self.nonmember),
                'deleted part alone, IF': self.score(None, self.deleted, rows),
                'deleted part alone, retrained': self.score(np.ones_like(self.deleted), self.deleted, rows),
                'retained + nonmember part alone, IF': self.score(None, ~self.deleted, rows)}


CLASS_NAMES = {1: 'DME', 2: 'DRUSEN'}


def class_vbar(analysis_if, c, fr, seed):
    """{class: [R, 4]} sensitivity of that attack class's loss to the Stage-1 probabilities, v6's adjoint averaged
    over attack seeds, not yet weighted. Same file and layout checks as analysis_if.vbar_from_v6."""
    d = analysis_if.v6_dir(c, f'base/seed{seed}', dict(kind='base6', seed=seed))
    with np.load(d / 'v.npz') as z:
        V = z['v']
        if (list(z['attack_seeds']) != list(fr['attack_seeds']) or list(z['classes']) != c['classes']
                or list(z['attack_gammas']) != c['attack_gammas'] or V.ndim != 5 or not np.isfinite(V).all()):
            raise ValueError('v6 adjoint layout or values differ from the frozen panel')
    gi = c['attack_gammas'].index(fr['attack_gamma_star'])
    return {cls: V[:, ci, gi].mean(0) for ci, cls in enumerate(c['classes']) if cls in fr['qualified_classes']}


def e1_classes(bridges, Ta, ta_cls):
    """bridges {'combined': Bridge, 'DME': Bridge of DME's own v, 'DRUSEN': ...}. Every score against every truth."""
    truths = {'combined': Ta, **{CLASS_NAMES[c]: t for c, t in ta_cls.items()}}
    out = []
    for sname, br in bridges.items():
        for label, s in br.bridge_rows().items():
            for tname, t in truths.items():
                out.append(dict(score_of=sname, truth=tname, score=label, lds=rho(s, t)))
    return pd.DataFrame(out)


def image_sets(img, features=None):
    """Sets of attack-active training images, ordered by criteria that do not use the retrained values,
    plus the oracle order. img: DataFrame over the N training images."""
    act = np.flatnonzero(img.active.values)
    by = lambda v, asc: act[np.argsort(v[act] if asc else -v[act], kind='stable')]
    sets = {}
    orders = {'lowest margin': by(img.margin.values, True), 'largest |a_IF|': by(np.abs(img.a_if.values), False)}
    if features is not None:
        orders['smallest gradient norm'] = by(features, True)
    orders['ORACLE largest |a_true|'] = by(np.abs(img.a_true.values), False)
    for name, o in orders.items():
        for n in SIZES:
            sets[(name, n)] = o[:n]
    for th in THRESHOLDS:
        sets[(f'margin < {th}', int((img.margin.values[act] < th).sum()))] = act[img.margin.values[act] < th]
    for name, c in (('class DME', 1), ('class DRUSEN', 2)):
        sets[(name, int((img.cls.values[act] == c).sum()))] = act[img.cls.values[act] == c]
    return {k: v for k, v in sets.items() if 0 < k[1] <= len(act)}, act      # sizes beyond the active set are skipped


def e2_images(br, Ta, sets, act, rng):
    N, R = br.N, br.R

    def variants(idx):
        rowS = np.zeros(R, bool)
        rowS[idx] = True
        keep = ~rowS[None, :] & np.ones((br.J, 1), bool)
        sel = br.deleted & rowS[None, :]
        return {'drop, row 1': br.score(None, keep), 'drop, row 2': br.score(np.ones_like(br.deleted), keep),
                'drop, row 3': br.score(br.deleted, keep), 'fix, row 1 + S deleted rows retrained': br.score(sel),
                'leave, row 3 with S left at IF': br.score(br.deleted & ~rowS[None, :])}

    base = {k: rho(v, Ta) for k, v in variants(np.array([], int)).items()}
    out = []
    draws = {}
    for (name, n), idx in sets.items():
        whole_class = name.startswith('class ')     # random sets mix DME and DRUSEN in other proportions: no like-for-like control
        if not whole_class and n not in draws:
            draws[n] = np.array([[rho(v, Ta) for v in variants(rng.choice(act, n, replace=False)).values()]
                                 for _ in range(DRAWS)])
        for vi, (vname, v) in enumerate(variants(idx).items()):
            lds = rho(v, Ta)
            r = None if whole_class else draws[n][:, vi]
            out.append(dict(set=name, n=n, variant=vname, lds=lds,
                            random_mean=np.nan if r is None else r.mean(),
                            random_p05=np.nan if r is None else np.percentile(r, 5),
                            random_p95=np.nan if r is None else np.percentile(r, 95),
                            share_of_random_below=np.nan if r is None else float((r < lds).mean()),
                            with_retrained_values_in_selection=name.startswith('ORACLE')))
    return pd.DataFrame(out), base


def own_terms(M, T, P, vb, p0, y):
    """Per training image: margin, class, active flag, attack-weighted half-data contrast of the retrained and predicted change."""
    N = M.shape[1]
    et, ep = own_effect(M, T[:, :, :N]), own_effect(M, P[:, :, :N])
    logp = np.log(p0[:N])
    other = np.where(np.eye(4, dtype=bool)[y[:N]], -np.inf, logp).max(1)
    return pd.DataFrame(dict(cls=y[:N], margin=logp[np.arange(N), y[:N]] - other,
                             active=np.any(vb[:, :N] != 0, axis=(0, 2)),
                             a_true=(vb[:, :N] * et).sum(-1).mean(0), a_if=(vb[:, :N] * ep).sum(-1).mean(0)))


def checks(bridges, Ta, ta_cls, omega, vb, vbc, y):
    """Identities and the reference values of the existing bridge table."""
    br = bridges['combined']
    s1, s3 = br.score(None), br.score(br.deleted)
    parts = sum(b.score(None) for k, b in bridges.items() if k != 'combined')
    cls_sum = sum(omega[c] * ta_cls[c] for c in omega)
    out_rows = ~np.isin(y, list(omega))
    chk = dict(row1_lds=rho(s1, Ta), row3_lds=rho(s3, Ta),
               class_scores_sum_to_row1_max_abs=float(np.abs(parts - s1).max()),
               class_truths_recombine_to_attack_truth_max_abs=float(np.abs(cls_sum - Ta).max()),
               class_vbar_sum_vs_mixed_vbar_max_abs=float(np.abs(sum(vbc.values()) - vb).max()),
               vbar_max_abs_on_rows_outside_attack_classes=float(np.abs(vb[:, out_rows]).max()) if out_rows.any() else 0.0)
    for c, v in vbc.items():     # how much of class c's own sensitivity sits on rows of another image class
        other = y != c
        chk[f'vbar_{CLASS_NAMES[c]}_max_abs_on_rows_of_other_classes'] = float(np.abs(v[:, other]).max()) if other.any() else 0.0
    return chk


def align_grad_norm(f, raw_index, y, N):
    """Gradient norm per training image, matched on train_position and verified against raw_index and class."""
    need = {'train_position', 'raw_index', 'cls', 'grad_norm'}
    if not need <= set(f.columns):
        raise ValueError(f'features file lacks columns {sorted(need - set(f.columns))}')
    pos = f.train_position.values
    if len(f) != N or len(np.unique(pos)) != N or pos.min() != 0 or pos.max() != N - 1:
        raise ValueError(f'features file does not hold exactly the {N} training images once each')
    if not (np.array_equal(f.raw_index.values, raw_index[pos]) and np.array_equal(f.cls.values, y[pos])):
        raise ValueError('features file rows differ from the interface run (raw_index or class)')
    if not np.isfinite(f.grad_norm.values).all():
        raise ValueError('features file has missing gradient norms')
    g = np.zeros(N)
    g[pos] = f.grad_norm.values
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='interface run directory')
    ap.add_argument('--gamma', type=float, default=None)
    ap.add_argument('--features', type=Path, default=None, help='features_merged.csv from case_features.py (optional)')
    ap.add_argument('--repo', type=Path, default=Path.cwd(), help='unchanged interface v1r1 repository checkout')
    ap.add_argument('--e1-only', action='store_true', help='per-class table only; skip the image sets')
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
    p0 = np.stack([data['solves'][s]['arrays']['p0'] for s in panel]).mean(0)
    M, y = data['M'], data['rows']['y']
    fr = data['fr6']
    omega = {int(k): v for k, v in fr['omega'].items()}
    per_seed = [class_vbar(analysis_if, c, fr, s) for s in panel]
    vbc = {k: omega[k] * np.stack([d[k] for d in per_seed]) for k in omega}      # weighted class sensitivities
    ta_cls = class_truth(data['v6_truth'], panel, omega)
    Ta = attack_truth(data['v6_truth'], panel, omega, len(fr['attack_seeds']))
    mk = lambda v: Bridge(terms(v, P), terms(v, T), M, y)
    bridges = {'combined': mk(vb), **{CLASS_NAMES[k]: mk(v) for k, v in vbc.items()}}
    br = bridges['combined']
    chk = checks(bridges, Ta, ta_cls, omega, vb, vbc, y)
    e1 = e1_classes(bridges, Ta, ta_cls)
    dest = Path(c['out']) / 'class_image_ablation' / f'gamma{g:g}'
    dest.mkdir(parents=True, exist_ok=True)
    e1.to_csv(dest / 'e1_class_lds.csv', index=False)
    show = lambda df: print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'), '\n')
    print(f'gamma = {g:g}, seed panel {panel}')
    print('--- reference and identity values (row 1 and row 3 of the existing bridge table: 0.262 and 0.730; max_abs near 0)')
    for k, v in chk.items():
        print(f'{k:62s} {v:.6g}')
    print('--- E1: each class score against its own truth, and the combined target')
    diag = e1[(e1.score_of == e1.truth)]
    show(diag.pivot_table(index=['score_of'], columns='score', values='lds', sort=False).reset_index())
    extra = {}
    if not a.e1_only:
        img = own_terms(M, T, P, vb, p0, y)
        feats = None
        if a.features is not None:
            feats = align_grad_norm(pd.read_csv(a.features), data['rows']['raw_index'], y, M.shape[1])
        sets, act = image_sets(img, feats)
        e2, base = e2_images(br, Ta, sets, act, np.random.default_rng(SEED))
        e2.to_csv(dest / 'e2_image_sets.csv', index=False)
        extra = {f'base {k}': v for k, v in base.items()} | dict(draws=DRAWS, seed=SEED, n_active=int(len(act)))
        print(f'--- E2: {len(act)} attack-active training images; each set against 300 random sets of the same size '
              f'(seed {SEED}); whole-class sets have no random control')
        for variant, d in e2.groupby('variant', sort=False):
            print(f'[{variant}]   no image removed or replaced: {base[variant]:.3f}')
            show(d.drop(columns=['variant']))
    pd.Series({**chk, **extra, 'gamma': g, 'panel': str(panel)}).to_csv(dest / 'checks.csv', header=['value'])
    print('Saved:', dest)


if __name__ == '__main__':
    main()
