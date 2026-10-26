"""Figures and example selection for the interface case study. Reads only the CSVs written by case_study.py,
so it runs anywhere (laptop included); no repository or GPU needed.

  fig1_subsets.png   32 subsets, scores vs attack truth in ranks (this is the LDS), next to the count baseline
  fig2_effect.png    deleted-minus-retained contrast: by margin decile, and IF vs truth (log-log)
  fig3_capture.png   cumulative absolute attack-weighted contrast and top-k class composition
  examples.csv       the images for the montage: fixed rules, listed below

Contrasts compare deleted vs retained membership across the existing half-data subsets;
they are not single-image LOO. Capture measures absolute contrast mass, not LDS gain.
The diagonal is the expected random-ranking curve, with no randomization experiment.

    python case_figures.py --dir <OUT>/case_study/gamma0.03_k100 [--examples 8]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

CLS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
COL = ['#8c8c8c', '#1f77b4', '#d62728', '#2ca02c']


def rho(a, b):
    return float(spearmanr(a, b)[0])


def select_examples(img, n=8, seed=0):
    """Fixed rules over DME/DRUSEN; a_true/a_if are attack-weighted contrasts."""
    act = img[img.attack_row_active]
    t, f = act.a_true.abs(), act.a_if.abs()
    top100 = t.nlargest(100).index
    pick = {'largest |true contrast|': t.nlargest(n).index,
            'largest |IF contrast|': f.nlargest(n).index,
            'true top-100, IF rank lowest': f.rank(ascending=False)[top100].nlargest(n).index,
            'random active': act.sample(n, random_state=seed).index,
            'smallest |true contrast|': t.nsmallest(n).index}
    cols = ['raw_index', 'train_position', 'cls', 'p_true', 'margin', 'a_true', 'a_if', 'gap_share']
    return pd.concat([img.loc[idx, cols].assign(example_group=name) for name, idx in pick.items()])


def fig_subsets(sub, path):
    n_act = sub.deleted_cls1 + sub.deleted_cls2
    r = lambda v: rankdata(v)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    panels = (('row 1: L3 (all IF)', sub.row1, 'LDS'),
              ('row 3: deleted rows use truth', sub.row3, 'LDS'),
              ('count baseline: # deleted DME+DRUSEN', n_act, 'Spearman'))
    for a, (title, v, metric) in zip(ax, panels):
        s = a.scatter(r(v), r(sub.attack_truth), c=n_act, cmap='viridis', s=45)
        a.plot([1, len(sub)], [1, len(sub)], color='#bbbbbb', lw=1, zorder=0)
        a.set_title(f'{title}\n{metric} = {rho(v, sub.attack_truth):.3f}', fontsize=10)
        a.set_xlabel(f'ascending rank over {len(sub)} subsets')
        a.set_ylabel('rank of attack truth')
    fig.colorbar(s, ax=ax, label='# deleted DME+DRUSEN images', fraction=0.025)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_effect(img, path):
    d = pd.qcut(img.margin, 10, labels=False)
    g = img.assign(t=img.own_delta_py_true.abs(), p=img.own_delta_py_if.abs()).groupby(d)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(g.margin.median(), g.t.mean(), 'o-', label='retrained contrast')
    ax[0].plot(g.margin.median(), g.p.mean(), 's-', label='IF (gamma = 0.03)')
    ax[0].set_yscale('log')
    ax[0].set(xlabel='full-model margin (median per decile)', ylabel='mean |contrast in p(true class)|',
              title='Deleted-minus-retained contrast vs margin')
    ax[0].legend()
    floor = 1e-12
    for c in range(4):
        m = img.cls == c
        ax[1].scatter(np.maximum(img.own_delta_py_if[m].abs(), floor), np.maximum(img.own_delta_py_true[m].abs(), floor),
                      s=6, alpha=.5, color=COL[c], label=CLS[c])
    ax[1].set_xscale('log'), ax[1].set_yscale('log')
    ax[1].plot([floor, 1], [floor, 1], '--', color='#777777', lw=1, label='equal magnitude')
    ax[1].set(xlabel='|IF contrast|', ylabel='|retrained contrast|',
              title=f'Spearman: absolute = {rho(img.own_delta_py_if.abs(), img.own_delta_py_true.abs()):.3f}; '
                    f'signed = {rho(img.own_delta_py_if, img.own_delta_py_true):.3f}')
    ax[1].legend(markerscale=3)
    n_if = (img.own_delta_py_if.abs() < floor).sum()
    n_true = (img.own_delta_py_true.abs() < floor).sum()
    fig.text(.5, .01, 'Contrast = mean when deleted minus mean when retained across half-data subsets.\n'
             f'Scatter display floor: 1e-12 (IF: {n_if}, truth: {n_true} points); correlations use raw values.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_capture(img, groups, path):
    act = img[img.attack_row_active]
    w = act.a_true.abs().to_numpy()
    x = np.arange(len(w) + 1) / len(w)
    curve = lambda order: np.r_[0, np.cumsum(w[order]) / w.sum()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for name, order in (('truth (oracle)', np.argsort(-w, kind='stable')),
                        ('IF', np.argsort(-act.a_if.abs().to_numpy(), kind='stable')),
                        ('low margin', np.argsort(act.margin.to_numpy(), kind='stable'))):
        ax[0].plot(x, curve(order), label=name)
    ax[0].plot(x, x, '--', color='#999999', label='random expectation')
    ax[0].set(xlabel='fraction of DME+DRUSEN images inspected', ylabel='fraction of total |true weighted contrast|',
              title='Capture of absolute contrast mass')
    ax[0].legend()
    names = ['IF own largest absolute', 'true own largest absolute', 'all attack-active rows',
             'L3 whole-path largest absolute', 'all training images']
    g = groups.set_index('group').loc[names]
    bottom = np.zeros(len(g))
    for c in range(4):
        ax[1].bar(range(len(g)), g[f'frac_cls{c}'], bottom=bottom, color=COL[c], label=CLS[c])
        bottom += g[f'frac_cls{c}'].to_numpy()
    ax[1].set_xticks(range(len(g)), ['IF top\nactive', 'true top\nactive', 'all\nactive',
                                    'tau top\nall train', 'all\ntrain'])
    ax[1].set(ylabel='class fraction', title=f'class mix of the top groups (k = {int(g.n.iloc[0])})')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, .5))
    fig.text(.5, .01, 'Capture uses absolute attack-weighted deleted-minus-retained contrasts, not a share of LDS improvement.\n'
             'IF/true rankings use DME+DRUSEN; whole-path tau uses all training images.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True, help='case_study/gamma0.03_k* directory')
    ap.add_argument('--examples', type=int, default=8)
    a = ap.parse_args()
    img, sub, groups = (pd.read_csv(a.dir / f'{n}.csv') for n in ('images', 'subsets', 'groups'))
    if not 1 <= a.examples <= min(100, int(img.attack_row_active.sum())):
        ap.error('--examples must be between 1 and the smaller of 100 and the active image count')
    fig_subsets(sub, a.dir / 'fig1_subsets.png')
    fig_effect(img, a.dir / 'fig2_effect.png')
    fig_capture(img, groups, a.dir / 'fig3_capture.png')
    select_examples(img, a.examples).to_csv(a.dir / 'examples.csv', index=False)
    print('Saved fig1_subsets.png, fig2_effect.png, fig3_capture.png, examples.csv in', a.dir)


if __name__ == '__main__':
    main()
