"""Case-study features: full-model margin, loss and mean per-seed CE gradient norm.

Uses existing images.csv and explicitly selected features_seed*.csv. Features are
aligned by train_position, raw_index and cls before averaging. Default seeds match
the case-study panel 42, 43, 44. No Hessian approximation, training or resampling.

Targets: whole-path removal_score on all training images; a_true, a_if and gap_share
on attack-active rows. Also report DME and DRUSEN separately. gap_share decomposes
the row 3 minus row 1 score difference, not an LDS improvement. The own-row quantities
a_true and a_if are half-data deleted-minus-retained contrasts, not single-image LOO.

Outputs: features_summary.csv, features_groups.csv, features_merged.csv, fig5_features.png
    python case_features.py --dir <OUT>/case_study/gamma0.03_k100
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

KEYS = ['train_position', 'raw_index', 'cls']
CLS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
COLORS = ['#8c8c8c', '#1f77b4', '#d62728', '#2ca02c']


def rho(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y)[0])


def partial_rho(x, y, z):
    """Correlation of residual ranks after linear adjustment for the margin's rank."""
    r = [rankdata(v) for v in (x, y, z)]
    design = np.column_stack([np.ones(len(z)), r[2]])
    res = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in r[:2]]
    if any(np.allclose(v, 0) for v in res):
        return np.nan
    return float(np.corrcoef(*res)[0, 1])


def load(d, seeds=(42, 43, 44)):
    img = pd.read_csv(d / 'images.csv')
    index = img.set_index(KEYS, verify_integrity=True).index
    files = [d.parent / 'features' / f'features_seed{s}.csv' for s in seeds]
    norms = [pd.read_csv(f).set_index(KEYS, verify_integrity=True).loc[index, 'grad_norm'].to_numpy()
             for f in files]
    img['grad_norm'] = np.mean(norms, axis=0)
    return img, ['grad_norm'], [f.name for f in files]


def summary(img, cols):
    populations = {'all training': img, 'attack active': img[img.attack_row_active],
                   'DME': img[img.cls == 1], 'DRUSEN': img[img.cls == 2]}
    rows = []
    for name, data in populations.items():
        targets = ['removal_score'] if name == 'all training' else ['removal_score', 'a_true', 'a_if', 'gap_share']
        for target in targets:
            for feature in ['margin', 'loss'] + cols:
                d = data[np.isfinite(data[[feature, target, 'margin']]).all(axis=1)]
                x, y, z = d[feature].to_numpy(), d[target].abs().to_numpy(), d.margin.to_numpy()
                rows.append(dict(population=name, target=target, feature=feature, n_images=len(d),
                                 spearman_abs_score=rho(x, y),
                                 partial_abs_score_given_margin=partial_rho(x, y, z)
                                 if feature != 'margin' and len(d) >= 3 else np.nan))
    return pd.DataFrame(rows)


def groups(img, cols, k=100, n=20):
    act = img[img.attack_row_active]
    t, f = act.a_true.abs(), act.a_if.abs()
    top = t.nlargest(k).index
    sets = [(f'true top-{k}', 'attack active', top),
            (f'IF top-{k}', 'attack active', f.nlargest(k).index),
            (f'true top-{k}, IF rank lowest {n}', 'attack active', f.rank(ascending=False)[top].nlargest(n).index),
            (f'true smallest {k}', 'attack active', t.nsmallest(k).index),
            (f'correction top-{k}', 'attack active', act.gap_share.abs().nlargest(k).index),
            ('all attack-active', 'attack active', act.index),
            (f'whole-path top-{k}', 'all training', img.removal_score.abs().nlargest(k).index),
            (f'whole-path smallest {k}', 'all training', img.removal_score.abs().nsmallest(k).index),
            ('all training', 'all training', img.index)]
    rows = []
    for name, population, idx in sets:
        d = img.loc[idx]
        rows.append(dict(group=name, population=population, n=len(d),
                         **{f'median_{f}': d[f].median() for f in ['margin', 'loss'] + cols},
                         **{f'frac_cls{c}': (d.cls == c).mean() for c in range(4)}))
    return pd.DataFrame(rows)


def figure(img, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    act = img[img.attack_row_active]
    panels = [('removal_score', img, 'Whole-path score: all training'),
              ('gap_share', act, 'Row 3 - row 1 correction: active rows'),
              ('a_if', act, 'IF own-row contrast: active rows'),
              ('a_true', act, 'Retrained own-row contrast: active rows')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for ax, (target, data, title) in zip(axes.ravel(), panels):
        finite = np.isfinite(data[['grad_norm', target]]).all(axis=1)
        d = data[finite]
        shown = (d.grad_norm > 0) & (d[target].abs() > 0)
        for c in sorted(d.cls.unique()):
            p = d[shown & (d.cls == c)]
            ax.scatter(p.grad_norm, p[target].abs(), s=7, alpha=.45, color=COLORS[c], label=CLS[c])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(xlabel='mean per-seed ||grad CE||', ylabel=f'|{target}|',
               title=f'{title}\nSpearman = {rho(d.grad_norm, d[target].abs()):.3f}; plotted {shown.sum()}/{len(d)}')
        ax.legend(markerscale=2, fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True, help='case_study/gamma0.03_k* directory')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--top-k', type=int, help='defaults to the top-k in the case study groups.csv')
    a = ap.parse_args()
    img, cols, files = load(a.dir, a.seeds)
    k = a.top_k
    if k is None:
        k = int(pd.read_csv(a.dir / 'groups.csv').set_index('group').loc['IF own largest absolute', 'n'])
    s, g = summary(img, cols), groups(img, cols, k)
    s.to_csv(a.dir / 'features_summary.csv', index=False)
    g.to_csv(a.dir / 'features_groups.csv', index=False)
    img.to_csv(a.dir / 'features_merged.csv', index=False)
    figure(img, a.dir / 'fig5_features.png')
    print('feature seeds:', a.seeds, '| files:', files)
    print(s.to_string(index=False, float_format=lambda x: f'{x:.6g}'), '\n')
    print(g.to_string(index=False, float_format=lambda x: f'{x:.6g}'), '\n')
    print('Saved feature tables and fig5_features.png in', a.dir)


if __name__ == '__main__':
    main()
