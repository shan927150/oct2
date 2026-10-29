"""Per-image CE gradient norms at the frozen full models. GPU; no training or Hessian solves.

The filename is retained for existing submission commands. The earlier approximate
self-influence calculation is removed. grad_norm = ||grad CE_i|| = N ||g_i||,
where the repository returns g_i = grad CE_i / N.

Default seeds 42, 43, 44 match the completed case study. Output:
  OUT/case_study/features/features_seed<seed>.csv

    python self_influence.py --repo <frozen checkout> --out <interface run dir>
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def one_seed(sess, seed):
    model = sess.model(sess.full_name(seed), seed)
    G = sess.image_gradients(model)  # [N, parameters], grad CE_i / N
    norm = sess.N * G.norm(dim=1).cpu().numpy()
    rows = sess.rows3[:sess.N]
    return pd.DataFrame(dict(train_position=np.arange(sess.N), raw_index=rows,
                             cls=sess.ctx['y'][rows], grad_norm=norm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='completed interface run directory')
    ap.add_argument('--repo', type=Path, default=Path.cwd(), help='unchanged interface v1r1 checkout')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / 'experiments/pathway2_interface_lds_v1'))
    import run
    from core_if import ISession
    c = run.config(a.out)
    sess = ISession(c)
    dest = Path(c['out']) / 'case_study' / 'features'
    dest.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        start = time.monotonic()
        feats = one_seed(sess, seed)
        feats.to_csv(dest / f'features_seed{seed}.csv', index=False)
        print(f'seed {seed}: gradient norms for {len(feats)} images, '
              f'{time.monotonic() - start:.1f} seconds', flush=True)
    print('Saved:', dest)


if __name__ == '__main__':
    main()
