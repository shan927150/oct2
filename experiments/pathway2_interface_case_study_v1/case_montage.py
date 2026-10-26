"""OCT montage of fixed case-study examples and patient counts in the top groups.
Needs the original images and the frozen data loader, which requires a GPU allocation.
Run case_figures.py first to write examples.csv. No model training is performed.
All displayed contrasts are from half-data subsets, not single-image LOO.

    python case_montage.py --repo ~/oct2-interface-lds-v1r1 --out <interface run dir> --dir <OUT>/case_study/gamma0.03_k100
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True       # a __pycache__ in the frozen checkout would make it "dirty"
CLS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


def montage(X, ex, path):
    """X [raw rows, 1, H, W]; ex: examples.csv (raw_index, cls, margin, a_true, a_if, example_group)."""
    names = list(dict.fromkeys(ex.example_group))
    n = (ex.example_group == names[0]).sum()
    fig, axes = plt.subplots(len(names), n, figsize=(2.1 * n, 2.5 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        for c, (_, e) in enumerate(ex[ex.example_group == name].iterrows()):
            ax = axes[r, c]
            ax.imshow(X[int(e.raw_index), 0], cmap='gray')
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f'{CLS[int(e.cls)]}  raw {int(e.raw_index)}\nmargin {e.margin:.1f}\n'
                         f'true {e.a_true:.2e}  IF {e.a_if:.2e}', fontsize=7)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle('Attack-weighted deleted-minus-retained contrasts from half-data subsets\n'
                 'Fixed example groups can overlap; raw IDs identify repeated images.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def patient_spread(img, groups, k=100):
    """Do a few patients dominate the top-k? groups [raw rows] = patient id."""
    act = img[img.attack_row_active]
    rows = {'true top-%d' % k: act.a_true.abs().nlargest(k).index, 'IF top-%d' % k: act.a_if.abs().nlargest(k).index,
            'all active': act.index}
    summary = []
    for name, idx in rows.items():
        _, counts = np.unique(groups[img.loc[idx, 'raw_index'].to_numpy()], return_counts=True)
        summary.append(dict(group=name, n_images=len(idx), n_patients=len(counts),
                            mean_images_per_patient=counts.mean(), max_images_per_patient=counts.max()))
    return pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='interface run directory')
    ap.add_argument('--dir', type=Path, required=True, help='case_study/gamma0.03_k* directory with examples.csv')
    ap.add_argument('--repo', type=Path, default=Path.cwd(), help='unchanged interface v1r1 repository checkout')
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / 'experiments/pathway2_interface_lds_v1'))
    import runtime as rt
    import run
    ctx = rt.load_context(run.config(a.out))[0]
    img, ex = pd.read_csv(a.dir / 'images.csv'), pd.read_csv(a.dir / 'examples.csv')
    if not (np.array_equal(np.asarray(ctx['train'])[img.train_position.to_numpy()], img.raw_index.to_numpy())
            and np.array_equal(np.asarray(ctx['y'])[img.raw_index.to_numpy()], img.cls.to_numpy())):
        raise ValueError('CSV training rows or labels differ from the frozen run')
    montage(ctx['X'], ex, a.dir / 'fig4_montage.png')
    groups = pd.read_csv(a.dir / 'groups.csv').set_index('group')
    k = int(groups.loc['IF own largest absolute', 'n'])
    patients = np.asarray(ctx['groups'])
    summary = patient_spread(img, patients, k=k)
    summary.to_csv(a.dir / 'patient_spread.csv', index=False)
    ex.assign(patient_id=patients[ex.raw_index.to_numpy()]).to_csv(a.dir / 'examples_with_patients.csv', index=False)
    print(summary.to_string(index=False))
    print('Saved fig4_montage.png, patient_spread.csv, examples_with_patients.csv in', a.dir)


if __name__ == '__main__':
    main()
