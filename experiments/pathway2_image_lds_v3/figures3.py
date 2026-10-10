"""Report figures (matplotlib Agg). Series colours follow the predictor, never the rank."""
from pathlib import Path
import numpy as np

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TCOL = {'ce': (BLUE, 'o', 'CE'), 'brier': (ORANGE, 's', 'Brier'), 'oneminusp': (AQUA, '^', '1 - p_y')}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 9, 'legend.frameon': False, 'lines.linewidth': 2, 'axes.edgecolor': MUTED})
    return plt


def make_all(s, out):
    plt = _plt()
    out = Path(out)
    files = []
    curves = s.get('seed_curves', {})
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
        for t, (col, mk, lab) in TCOL.items():
            cur = curves.get(f'{t}_img')
            if not cur:
                continue
            rows = [r for r in cur['reliability']['rows'] if r.get('split_spearman_mean') is not None]
            axes[0].plot([r['k'] for r in rows], [r['split_spearman_mean'] for r in rows], color=col, marker=mk, label=lab)
        axes[0].set_xscale('log', base=2); axes[0].set_ylim(0, 1.02)
        axes[0].set_xlabel('seeds per disjoint panel k'); axes[0].set_title('Truth rank repeatability (image-mean)')
        axes[0].legend()
        cur = curves.get('ce_img', {})
        colours = {'proxy': BLUE, 'IF gamma=1': ORANGE}
        for j, name in enumerate(dict.fromkeys(r['predictor'] for r in cur.get('lds', []))):
            rows = [r for r in cur['lds'] if r['predictor'] == name and r['mean'] is not None]
            col = next((v for k, v in colours.items() if name.startswith(k)), [AQUA, YELLOW][j % 2])
            axes[1].plot([r['k'] for r in rows], [r['mean'] for r in rows], color=col, marker='o', label=name)
            axes[1].fill_between([r['k'] for r in rows], [r['q025'] for r in rows], [r['q975'] for r in rows],
                                 color=col, alpha=.12, linewidth=0)
        axes[1].axhline(0, color=MUTED, linewidth=.8)
        crit = s.get('critical_spearman')
        if crit:
            axes[1].axhline(crit, color=MUTED, linewidth=1, linestyle='--')
            axes[1].annotate(f'one-sided 5% ({crit:.2f})', (1, crit), textcoords='offset points', xytext=(2, 3),
                             fontsize=8, color=MUTED)
        axes[1].set_xscale('log', base=2); axes[1].set_ylim(-1, 1)
        axes[1].set_xlabel('seeds averaged in truth k'); axes[1].set_title('CE image-mean LDS (deleter) vs seeds')
        axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / 'fig1_truth_and_lds_vs_k.png', dpi=150); plt.close(fig)
        files.append('fig1_truth_and_lds_vs_k.png')
    pq = s.get('per_query', {})
    if pq:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
        for ax, (t, (col, mk, lab)) in zip(axes, TCOL.items()):
            r = pq.get(f'{t}_image')
            if not r:
                continue
            v = np.asarray([x for x in r['per_query'] if x is not None], float)
            v = v[np.isfinite(v)]
            ax.hist(v, bins=np.linspace(-1, 1, 21), color=col, edgecolor='white', linewidth=1.5)
            ax.axvline(r['proxy']['mean'], color=INK, linewidth=1.5, label=f"mean {r['proxy']['mean']:.3f}")
            ax.set_title(f"{lab}: per query image (n={r['proxy']['defined']})"); ax.legend(fontsize=7)
            ax.set_xlabel('proxy LDS across subsets')
        fig.tight_layout(); fig.savefig(out / 'fig2_per_query_image.png', dpi=150); plt.close(fig)
        files.append('fig2_per_query_image.png')
    gam = s.get('gamma', {})
    if any(any('lds_img_raw' in e for e in rows.values()) for rows in gam.values()):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
        for t, (col, mk, lab) in TCOL.items():
            rows = gam.get(t, {})
            gs = sorted(float(g) for g in rows)
            pts = [(g, rows[f'{g:g}']['lds_img_raw']['spearman']) for g in gs if 'lds_img_raw' in rows[f'{g:g}']]
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, label=lab)
            pts = [(g, rows[f'{g:g}']['mean_spearman_if_vs_proxy']) for g in gs
                   if rows[f'{g:g}']['mean_spearman_if_vs_proxy'] is not None]
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, label=lab)
        for ax, title in zip(axes, ('IF LDS (image-mean, raw) vs gamma', 'Spearman(IF, proxy) over 2,045 images')):
            ax.set_xscale('log'); ax.set_ylim(-1, 1.02); ax.axhline(0, color=MUTED, linewidth=.8)
            ax.set_xlabel('gamma (fixed in advance)'); ax.set_title(title); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / 'fig3_gamma.png', dpi=150); plt.close(fig)
        files.append('fig3_gamma.png')
    return files
