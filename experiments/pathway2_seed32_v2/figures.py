"""Static report figures (matplotlib, Agg). Validated categorical slots in fixed
order; every series also has its own marker and a legend (contrast WARN relief)."""
from pathlib import Path
import numpy as np

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'legend.frameon': False,
                         'lines.linewidth': 2, 'lines.markersize': 6})
    return plt


def task1(summary, out):
    plt = _plt()
    files = []
    for design, eps in summary['task1'].items():
        for ep, by_t in eps.items():
            targets = list(by_t)
            fig, axes = plt.subplots(len(targets), 3, figsize=(11, 3.1 * len(targets)), squeeze=False)
            for i, t in enumerate(targets):
                cur = by_t[t]
                ax = axes[i, 0]
                ks = [r['k'] for r in cur['se']]
                ax.plot(ks, [r['analytic_se_mean_over_subsets'] for r in cur['se']], color=SERIES[0], marker=MARKERS[0], label='sd/sqrt(k)')
                rs = [r for r in cur['se'] if r['k'] < max(ks)]  # k = all seeds has a single draw
                ax.plot([r['k'] for r in rs], [r['resampled_sd_of_k_seed_mean'] for r in rs], color=SERIES[1],
                        marker=MARKERS[1], label='resampled k of all seeds (finite pool)')
                ax.set_xscale('log', base=2); ax.set_yscale('log')
                ax.set_title(f'{t}: SE of subset mean'); ax.set_xlabel('seeds k')
                ax = axes[i, 1]
                rows = cur['reliability']['rows']
                emp = [r for r in rows if r.get('split_spearman_mean') is not None]
                ax.errorbar([r['k'] for r in emp], [r['split_spearman_mean'] for r in emp],
                            yerr=[[r['split_spearman_mean'] - r['split_spearman_q025'] for r in emp],
                                  [r['split_spearman_q975'] - r['split_spearman_mean'] for r in emp]],
                            color=SERIES[0], marker=MARKERS[0], capsize=3, label='disjoint split-half Spearman')
                if rows[0].get('spearman_brown_from_k1') is not None:
                    ax.plot([r['k'] for r in rows], [r['spearman_brown_from_k1'] for r in rows], color=SERIES[1],
                            marker=MARKERS[1], linestyle='--', label='Spearman-Brown from k=1')
                icc = [r for r in rows if r.get('icc_reliability') is not None]
                ax.plot([r['k'] for r in icc], [r['icc_reliability'] for r in icc], color=SERIES[2],
                        marker=MARKERS[2], linestyle=':', label='variance components (all seeds)')
                ax.set_xscale('log', base=2); ax.set_ylim(-.2, 1.02)
                ax.set_title(f'{t}: subset-ranking reliability'); ax.set_xlabel('seeds k')
                ax = axes[i, 2]
                preds = list(dict.fromkeys(r['predictor'] for r in cur['lds']))
                for p in preds:
                    j = 0 if p.startswith('IF') else 1 if p.startswith('proxy') else 2  # colour follows the predictor
                    rr = [r for r in cur['lds'] if r['predictor'] == p and r['mean'] is not None]
                    ax.plot([r['k'] for r in rr], [r['mean'] for r in rr], color=SERIES[j], marker=MARKERS[j], label=p)
                    ax.fill_between([r['k'] for r in rr], [r['q025'] for r in rr], [r['q975'] for r in rr],
                                    color=SERIES[j], alpha=.12, linewidth=0)
                crit = summary['critical_spearman'].get(str(summary['designs'][design]['m']))
                if crit:
                    ax.axhline(crit, color=MUTED, linewidth=1, linestyle='--')
                    ax.annotate(f'one-sided 5% ({crit:.2f})', (ks[0], crit), color=MUTED, fontsize=8,
                                textcoords='offset points', xytext=(2, 3))
                ax.axhline(0, color=MUTED, linewidth=.8)
                ax.set_xscale('log', base=2); ax.set_ylim(-1.05, 1.05)
                ax.set_title(f'{t}: LDS (deleter) vs seeds'); ax.set_xlabel('seeds k')
                for a in axes[i]:
                    a.set_xticks(ks); a.set_xticklabels([str(k) for k in ks])
                if i == 0:
                    for a in axes[i]:
                        a.legend(fontsize=7, loc='best')
            fig.suptitle(f'Task 1 - {design}, {ep} endpoint (band = 2.5-97.5% over random seed draws)', color=INK)
            fig.tight_layout()
            path = Path(out) / f'fig_task1_{design}_{ep}.png'
            fig.savefig(path, dpi=150); plt.close(fig)
            files.append(path.name)
    return files


def task3_size(truths, summary, out):
    plt = _plt()
    designs = [d for d in truths if truths[d].get('seeds')]
    if not designs:
        return []
    targets = list(summary['task2_3'][designs[0]]['raw'])
    fig, axes = plt.subplots(1, len(targets), figsize=(3.8 * len(targets), 3.4), squeeze=False)
    for i, t in enumerate(targets):
        ax = axes[0, i]
        for j, d in enumerate(designs):
            D = truths[d]['raw'][t]['D'].mean(1)
            rho = summary['task2_3'][d]['raw'][t]['truth_n_images_spearman']
            ax.scatter(truths[d]['n_images'], D, color=SERIES[j], marker=MARKERS[j], s=28,
                       edgecolor=SURFACE, linewidth=.8, label=f'{d} (Spearman {rho:+.2f})')
        ax.set_xlabel('retained training images N_j'); ax.set_ylabel(f'mean Delta {t} (raw)')
        ax.set_title(t); ax.legend(fontsize=7)
    fig.suptitle('Task 3 - truth vs subset size (size confounding check)', color=INK)
    fig.tight_layout()
    path = Path(out) / 'fig_task3_size.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    return [path.name]


def task4(summary, out):
    plt = _plt()
    targets = list(summary['task4'])
    if not targets:
        return []
    fig, axes = plt.subplots(len(targets), 2, figsize=(9, 3.1 * len(targets)), squeeze=False)
    for i, t in enumerate(targets):
        rows = summary['task4'][t]
        gs = sorted(float(g) for g in rows)
        designs = list(dict.fromkeys(d for g in rows for d in rows[g]['lds']))
        ax = axes[i, 0]
        for j, d in enumerate(designs):
            pts = [(g, rows[f'{g:g}']['lds'][d]['spearman']) for g in gs if d in rows[f'{g:g}']['lds']]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[j], marker=MARKERS[j], label=d)
        for g in gs:
            if rows[f'{g:g}']['status'] != 'solved':
                ax.annotate('unsolvable', (g, 0), color=MUTED, fontsize=7, rotation=90, ha='center')
                ax.axvline(g, color=GRID, linewidth=4, zorder=0)
        ax.axhline(0, color=MUTED, linewidth=.8)
        ax.set_xscale('log'); ax.set_ylim(-1.05, 1.05); ax.set_xlabel('gamma'); ax.set_title(f'{t}: LDS (deleter, raw) vs gamma')
        ax.legend(fontsize=7)
        ax = axes[i, 1]
        ax.plot(gs, [rows[f'{g:g}']['mean_spearman_if_vs_proxy'] for g in gs], color=SERIES[0], marker=MARKERS[0],
                label='mean over score seeds')
        ax.set_xscale('log'); ax.set_ylim(-1, 1.02); ax.set_xlabel('gamma')
        ax.set_title(f'{t}: Spearman(IF, grad inner product), 148 patients'); ax.legend(fontsize=7)
    fig.suptitle('Task 4 - gamma sweep (gammas fixed in advance; not selected by LDS)', color=INK)
    fig.tight_layout()
    path = Path(out) / 'fig_task4_gamma.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    return [path.name]


def task5(summary, out):
    plt = _plt()
    designs = [d for d in summary['task5'] if summary['task5'][d].get('raw')]
    if not designs:
        return []
    targets = list(summary['task5'][designs[0]]['raw'])
    fig, axes = plt.subplots(len(designs), len(targets), figsize=(3.6 * len(targets), 2.8 * len(designs)), squeeze=False)
    for i, d in enumerate(designs):
        for j, t in enumerate(targets):
            r = summary['task5'][d]['raw'][t]
            v = np.asarray(r['per_query'], float)
            v = v[np.isfinite(v)]
            ax = axes[i, j]
            ax.hist(v, bins=np.linspace(-1, 1, 21), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
            ax.axvline(r['proxy']['mean'], color=INK, linewidth=1.5, label=f"mean {r['proxy']['mean']:+.3f}")
            ax.axvline(r['size_baseline']['mean'], color=SERIES[1], linewidth=1.5, linestyle='--',
                       label=f"size baseline {r['size_baseline']['mean']:+.3f}")
            ax.set_title(f"{d} / {t}  (perm p={r['proxy']['permutation_p_one_sided']:.3f})")
            ax.set_xlabel('per-query LDS (proxy)'); ax.legend(fontsize=7)
    nq = summary['task5'][designs[0]]['raw'][targets[0]]['proxy']['n_queries']
    fig.suptitle(f'Task 5 - per-query LDS over {nq} heldout patients (gradient-inner-product preview)', color=INK)
    fig.tight_layout()
    path = Path(out) / 'fig_task5_per_query.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    return [path.name]
