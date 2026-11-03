"""Shared loaders for methods A, B and C. Reads the frozen interface run (v1r1) and the v6 run it points to.

Nothing here writes into the frozen checkout or the existing run directories.
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PANEL = (42, 43, 44)
AFFECTED = 3


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def readcsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def writecsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rankdata_rows(X):
    """Average ranks along the last axis (ties averaged), vectorised over leading axes."""
    from scipy.stats import rankdata
    return rankdata(X, axis=-1)


def spearman_rows(X, y):
    """Spearman of every row of X [..., J] with y [J] (or with matching rows of y)."""
    rx = rankdata_rows(np.asarray(X, dtype=float))
    ry = rankdata_rows(np.asarray(y, dtype=float))
    rx = rx - rx.mean(-1, keepdims=True)
    ry = ry - ry.mean(-1, keepdims=True)
    num = (rx * ry).sum(-1)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def spearman(x, y):
    return float(spearman_rows(np.asarray(x)[None], np.asarray(y))[0])


def load_context(repo, out):
    """Interface data (analysis_if.load), the v6 attack table, and per-class pieces on the score panel."""
    sys.path.insert(0, str(Path(repo).resolve() / 'experiments/pathway2_interface_lds_v1'))
    import run
    import analysis_if as ai
    c = run.config(str(out))
    data = ai.load(Path(c['out']), c)
    if tuple(data['frozen']['score_panel']) != PANEL:
        raise ValueError(f"score panel is {data['frozen']['score_panel']}, expected {PANEL}")
    if len(data['M']) != 32:
        raise ValueError('expected 32 subsets')
    with np.load(Path(c['v6_out']) / 'smoke/fixed_rows.npz') as z:
        fixed = {k: z[k] for k in z.files}
    if not np.array_equal(fixed['raw_index'][fixed['shadow_id'] == AFFECTED], data['rows']['raw_index']):
        raise ValueError('attack table and interface rows are not aligned')
    return c, data, ai, fixed


def if_dp(c, data, ai, seed):
    """Matched IF probability change [J, R, 4] at the primary gamma (bridge row 1)."""
    arr = data['solves'][seed]['arrays']
    gi = list(c['gammas']).index(float(c['primary_gamma']))
    dz = ai.matched_dz(arr['dz_if'][:, gi], arr['dz_gall'][gi], data['M'].sum(1), data['M'].shape[1])
    return ai.softmax_jvp(arr['p0'], dz)


def true_dp(c, data, seed):
    """Retrained probability change [J, R, 4] (bridge row 2)."""
    return data['truth'][c['seeds'].index(seed)]


def row3_dp(c, data, seed, dp_if):
    """IF everywhere except the deleted rows, which take the retrained change (bridge row 3)."""
    M = data['M']
    out = dp_if.copy()
    t = true_dp(c, data, seed)
    for j in range(len(M)):
        d = np.flatnonzero(M[j] == 0)
        out[j, d] = t[j, d]
    return out


def class_vbar(c, data, seed):
    """Unweighted attack sensitivity per class at gamma_A*, mean over attack seeds: {cls: [R, 4]}."""
    fr = data['fr6']
    with np.load(Path(c['v6_out']) / f'base/seed{seed}/v.npz') as z:
        V = z['v']
        classes = [int(x) for x in z['classes']]
        gammas = [float(x) for x in z['attack_gammas']]
        if list(z['attack_seeds']) != list(fr['attack_seeds']):
            raise ValueError('v6 attack seeds differ from the frozen panel')
    gi = gammas.index(float(fr['attack_gamma_star']))
    return {cls: V[:, ci, gi].mean(0) for ci, cls in enumerate(classes)}


def class_v_per_seed(c, data, seed):
    """[K, R, 4] per attack seed, per class."""
    fr = data['fr6']
    with np.load(Path(c['v6_out']) / f'base/seed{seed}/v.npz') as z:
        V = z['v']
        classes = [int(x) for x in z['classes']]
        gammas = [float(x) for x in z['attack_gammas']]
    gi = gammas.index(float(fr['attack_gamma_star']))
    return {cls: V[:, ci, gi] for ci, cls in enumerate(classes)}


def class_truth(data, seeds=PANEL):
    """Stored v6 truth per class, mean over the given Stage-1 seeds and all attack seeds: {cls: [J]}."""
    J = len(data['M'])
    out = {}
    for (j, s), rows in data['v6_truth'].items():
        if s not in seeds:
            continue
        for r in rows:
            out.setdefault(int(r['cls']), [[] for _ in range(J)])[j].append(r['J10'] - r['J00'])
    n = {len(v) for vals in out.values() for v in vals}
    if len(n) != 1:
        raise ValueError(f'unequal truth cell counts {n}')
    return {cls: np.array([np.mean(v) for v in vals]) for cls, vals in out.items()}


def omega(data):
    return {int(k): float(v) for k, v in data['fr6']['omega'].items()}
