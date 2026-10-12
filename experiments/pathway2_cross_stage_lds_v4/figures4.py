"""Figures from summary.json only (matplotlib, Agg)."""
from pathlib import Path

INK, GRID = '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'


def _style(plt):
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                         'legend.frameon': False, 'savefig.dpi': 200})


def make_all(summary, rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    rep = Path(rep)
    out = []
    lad = summary.get('ladder') or {}
    crit = summary.get('critical_spearman')
    if lad:
        gA = summary['frozen']['attack_gamma_star']
        gs = summary.get('ladder_gamma_star')
        names = [f'L1 (gamma_A={gA:g})', 'L2-retrain', 'L2-lin'] + ([f'L3 (gamma={gs:g})'] if gs is not None else []) + ['proxy']
        names = [n for n in names if n in lad and lad[n].get('spearman') is not None]
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        for i, n in enumerate(names):
            b = lad[n]
            lo, hi = b.get('bootstrap_ci_conditional_on_seed_panel') or [b['spearman']] * 2
            col = BLUE if n.startswith('L1') else ORANGE if n.startswith('L2') else GREEN if n.startswith('L3') else GREY
            ax.bar(i, b['spearman'], width=.55, color=col)
            ax.errorbar(i, b['spearman'], yerr=[[b['spearman'] - lo], [hi - b['spearman']]], color=INK, capsize=3, lw=1)
        ax.axhline(0, color=INK, lw=.8)
        if crit:
            ax.axhline(crit, color=INK, lw=1, ls='--', label=f'one-sided 5% level {crit:.2f}')
            ax.legend(loc='upper right', fontsize=7.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=7.5)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('LDS vs Delta_value (32 subsets)')
        ax.set_title('Cross-stage ladder: more truth on the left', fontsize=9, loc='left')
        fig.tight_layout()
        fig.savefig(rep / 'fig1_ladder.png')
        plt.close(fig)
        out.append('fig1_ladder.png')
    g = summary.get('l3_gamma') or {}
    pts = [(float(k), lad[f'L3 (gamma={k})']['spearman']) for k in g if f'L3 (gamma={k})' in lad]
    if pts:
        pts.sort()
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=GREEN, lw=2, label='L3 (full formula)')
        for name, col, ls in (('L2-lin', ORANGE, ':'), ('proxy', GREY, '-.')):
            if name in lad and lad[name].get('spearman') is not None:
                ax.axhline(lad[name]['spearman'], color=col, ls=ls, lw=1.2, label=name)
        if crit:
            ax.axhline(crit, color=INK, lw=1, ls='--')
        ax.axhline(0, color=INK, lw=.8)
        ax.set_xscale('log')
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Stage-1 damping gamma (fixed in advance)')
        ax.set_ylabel('LDS')
        ax.legend(fontsize=7.5, loc='lower left')
        fig.tight_layout()
        fig.savefig(rep / 'fig2_l3_gamma.png')
        plt.close(fig)
        out.append('fig2_l3_gamma.png')
    curves = summary.get('seed_curves') or []
    rel = ((summary.get('truth') or {}).get('seed_reliability') or {}).get('rows') or []
    if curves or rel:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        ax = axes[0]
        r = [x for x in rel if x.get('split_spearman_mean') is not None]
        if r:
            ax.plot([x['k'] for x in r], [x['split_spearman_mean'] for x in r], marker='o', color=BLUE)
        ax.set_ylim(0, 1)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('training seeds per disjoint panel k')
        ax.set_title('Truth rank repeatability', fontsize=9, loc='left')
        ax = axes[1]
        cols = {'L1': BLUE, 'L2-retrain': ORANGE, 'L2-lin': '#b0461d'}
        for name in dict.fromkeys(x['predictor'] for x in curves):
            xs = [x for x in curves if x['predictor'] == name and x['mean'] is not None]
            ax.plot([x['k'] for x in xs], [x['mean'] for x in xs], marker='o', label=name,
                    color=cols.get(name, GREEN))
        if crit:
            ax.axhline(crit, color=INK, lw=1, ls='--')
        ax.axhline(0, color=INK, lw=.8)
        ax.set_ylim(-1, 1)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('training seeds averaged k')
        ax.set_title('LDS vs seeds', fontsize=9, loc='left')
        ax.legend(fontsize=7, loc='lower right')
        fig.tight_layout()
        fig.savefig(rep / 'fig3_truth_and_seeds.png')
        plt.close(fig)
        out.append('fig3_truth_and_seeds.png')
    return out
