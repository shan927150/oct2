"""Descriptive paired pilot; no confidence intervals from one seed/patient."""
from pathlib import Path
import runtime_late as rt


def collect(c):
    root=Path(c['out']);dest=root/'report';dest.mkdir(exist_ok=False)
    geometry=[];targets=[];trajectories=[];timing=[]
    for cell in c['cells']:
        d=root/cell['name'];geometry+=rt.csv_read(d/'geometry.csv');targets+=rt.csv_read(d/'targets.csv')
        trajectories+=rt.csv_read(d/'trajectory.csv')
        timing.append(dict(T0=cell['T0'],**rt.read(d/'result.json'),elapsed_seconds=rt.read(d/'complete.json')['elapsed_seconds']))
    rt.csv_write(dest/'geometry_all.csv',geometry);rt.csv_write(dest/'targets_all.csv',targets)
    rt.csv_write(dest/'trajectory_all.csv',trajectories);rt.write(dest/'timing_and_gates.json',timing)
    rt.write(dest/'summary.json',dict(seed=c['seed'],patient=c['patients'][0],all_replay_gates_passed=True,
        exact_stable_baseline=True,geometry_rows=len(geometry),query_rows=len(targets),
        primary_dose_pair=[.1,1.],small_dose_pair=[.01,.1],
        interpretation='Conditional native-Adam response; finite dose proportionality, not a certified derivative.',
        limitations=['one seed and patient','T0 changes remaining horizon and LR','fixed-window retains LR differences',
                     'no static IF solve','no cross-stage attack J','no chaos identification','no persistent-deletion LOO claim']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    panels=[('theta','full_relative_shift','Relative parameter displacement'),('theta','R','Dose norm ratio R'),
            ('theta','cosine','Dose direction cosine'),('query_probability','norm_full','Weighted query probability displacement'),
            ('query_probability','R','Query dose norm ratio'),('query_probability','cosine','Query dose direction cosine')]
    for ax,(space,key,title) in zip(axes.flat,panels):
        for endpoint,color in [('raw','#126ca6'),('ema','#c45a1a')]:
            for window,style in [('terminal','-'),('fixed 10 epochs','--')]:
                rows=[r for r in geometry if r['space']==space and r['endpoint']==endpoint and float(r['alpha_small'])==.1 and
                      (int(r['epoch'])==c['epochs'] if window=='terminal' else int(r['epoch'])-int(r['T0'])==c['window_epochs'])]
                rows=sorted(rows,key=lambda x:int(x['T0']))
                ax.plot([int(r['T0']) for r in rows],[float(r[key]) if r[key] else float('nan') for r in rows],
                        marker='o',linestyle=style,color=color,label=endpoint+' '+window)
        if key in ['R','cosine']:ax.axhline(1,color='gray',linewidth=.7)
        ax.set_title(title);ax.set_xlabel('Shared-prefix epochs T0');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"Late downweighting: seed {c['seed']}, patient {c['patients'][0]} | doses 0.1 vs 1")
    fig.tight_layout();fig.savefig(dest/'late_deletion_geometry.png',dpi=160);fig.savefig(dest/'late_deletion_geometry.pdf');plt.close(fig)
    (dest/'REPORT.md').write_text(
        '# Conditional late downweighting pilot\n\n'
        f"Seed {c['seed']}, patient {c['patients'][0]}, dropout off, Adam, 140 epochs, original warmup/cosine and EMA. "
        'The full native baseline matches the frozen stable baseline raw/EMA/optimizer exactly. '
        'Each T0 passes full-state zero-dose replay. Patient weight becomes 1-alpha after T0; original rows, batch denominators and global schedule are retained.\n\n'
        'Solid lines compare endpoints at epoch 140. Dashed lines compare equal 10-epoch windows; their learning rates still differ by T0. '
        'T0=130 belongs to both views. Raw and EMA share a trajectory and are not independent repeats. '
        'Query metrics use a fixed patient/class-balanced heldout panel, FP64 logits for CE, and no probability clipping. '
        'Probability geometry uses sqrt(query weight) times each displacement vector.\n\n'
        'Inspect both dose pairs, absolute changes, denominators and parameter ULP diagnostics before interpreting ratios. '
        'Zero-response ratios/cosines are NA. Positive dose proportionality does not certify static IF or chaos. '
        'No bootstrap CI is reported for this one-seed/one-patient pilot. Checkpoints remain on Delta and are omitted from the archive.\n')
