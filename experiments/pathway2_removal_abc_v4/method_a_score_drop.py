#!/usr/bin/env python3
"""Method A: separate the interface rows in the bridge score (CPU only, no training).

Per-row bridge contributions c_{j,r} = vbar_r . dp_{j,r}, with dp either the IF change (if) or the
Stage-1 retrained change (true; Stage 1 retrained, attack kept at first order). For a row set S_j:

  keep_if / keep_true      sum over S of the if / true contributions          vs the two-stage truth Delta J
  drop_if / drop_true      sum over the other rows                             vs the two-stage truth
  replace_if_by_true       IF everywhere, S retrained (bridge rows 3-5)        vs the two-stage truth
  replace_true_by_if       retrained everywhere, S back to IF (reverse direction)
  within_S                 keep_if vs keep_true: both sums contain the same rows. The latter is a
                           first-order surrogate using retrained Stage-1 changes, not the true two-stage
                           contribution of those rows. Sensitivities are computed using the full table.

The attack is not retrained and the two-stage truth is the stored v6 J10 - J00, so nothing here is a
deletion from the attack training table (method B) or from the target queries (method C). The old E2
"Drop" is drop_if on the image sets.

Row sets: deleted / retained / nonmember rows of each subset; image class (CNV, DME, DRUSEN, NORMAL);
group x class; fixed attack-active image sets (lowest margin, largest |a_IF|, largest |a_true| = Oracle).
Targets: combined (omega-weighted DME + DRUSEN), DME only, DRUSEN only.
Controls: same number of rows per image class and membership (per subset for group sets, fixed across
subsets for image sets), 1000 draws by default. The 2.5-97.5% range is descriptive.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common as C

NAMES = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
SIZES = (20, 50, 100, 200)
ROW1_REFERENCE = 0.26173020527859236      # interface v1r1 bridge row 1, gamma 0.03, panel 42-44


def contributions(c, data, fixed):
    """targets[t] = dict(if=[J, R], true=[J, R], truth=[J]) averaged over the score panel."""
    om = C.omega(data)
    qualified = [int(x) for x in data['fr6']['qualified_classes']]
    contrib, row1 = {'if': {}, 'true': {}}, []
    for s in C.PANEL:
        dps = {'if': C.if_dp(c, data, data['_ai'], s), 'true': C.true_dp(c, data, s)}
        vb = C.class_vbar(c, data, s)
        for v, dp in dps.items():
            for cls in qualified:
                contrib[v].setdefault(cls, []).append(np.einsum('jrk,rk->jr', dp, vb[cls]))
        row1.append(np.einsum('jrk,rk->j', dps['if'], data['vbar'][s]))
    contrib = {v: {cls: np.mean(x, 0) for cls, x in d.items()} for v, d in contrib.items()}
    T = C.class_truth(data)
    targets = {'combined': dict(truth=sum(om[k] * T[k] for k in qualified),
                                **{v: sum(om[k] * contrib[v][k] for k in qualified) for v in contrib})}
    for k in qualified:
        targets[NAMES[k]] = dict(truth=T[k], **{v: contrib[v][k] for v in contrib})
    row1 = np.mean(row1, 0)
    dev = float(np.max(np.abs(targets['combined']['if'].sum(1) - row1)) / max(np.abs(row1).max(), 1e-300))
    return targets, dev


QUANTITIES = ('keep_if', 'keep_true', 'drop_if', 'drop_true', 'replace_if_by_true', 'replace_true_by_if', 'within_S')


def scores(kif, ktrue, tif, ttrue):
    """kif/ktrue: sums over S ([..., J]); tif/ttrue: full sums [J]. Returns (score, reference) per quantity;
    reference None means the two-stage truth."""
    return {'keep_if': (kif, None), 'keep_true': (ktrue, None), 'drop_if': (tif - kif, None),
            'drop_true': (ttrue - ktrue, None), 'replace_if_by_true': (tif - kif + ktrue, None),
            'replace_true_by_if': (ttrue - ktrue + kif, None), 'within_S': (kif, ktrue)}


def row_sets(data, fixed, images):
    """name -> (family, mask [J, R] bool, matching), matching in {'per_subset', 'fixed'}."""
    M = data['M']
    J, N = M.shape
    R = len(data['rows']['raw_index'])
    cls = fixed['classes'][fixed['shadow_id'] == C.AFFECTED]
    deleted = np.zeros((J, R), bool); deleted[:, :N] = M == 0
    retained = np.zeros((J, R), bool); retained[:, :N] = M == 1
    nonmember = np.zeros((J, R), bool); nonmember[:, N:] = True
    sets = {}
    for g, m in [('deleted', deleted), ('retained', retained), ('nonmember', nonmember)]:
        sets[g] = ('group', m, 'per_subset')
    for k, name in NAMES.items():
        sets[f'class_{name}'] = ('class', np.broadcast_to(cls == k, (J, R)).copy(), 'per_subset')
        for g, m in [('deleted', deleted), ('retained', retained), ('nonmember', nonmember)]:
            sets[f'{g}_{name}'] = ('group_x_class', m & (cls == k)[None], 'per_subset')
    active = [r for r in images if str(r['attack_row_active']).lower() == 'true']
    pos = lambda rows: np.array([int(r['train_position']) for r in rows], int)
    orders = {'low_margin': sorted(active, key=lambda r: (float(r['margin']), int(r['train_position']))),
              'high_if': sorted(active, key=lambda r: (-abs(float(r['a_if'])), int(r['train_position']))),
              'oracle_a_true': sorted(active, key=lambda r: (-abs(float(r['a_true'])), int(r['train_position'])))}
    for name, order in orders.items():
        for n in SIZES:
            m = np.zeros(R, bool); m[pos(order[:n])] = True
            sets[f'{name}_{n}'] = ('image_set', np.broadcast_to(m, (J, R)).copy(), 'fixed')
    pool_active = np.zeros(R, bool); pool_active[pos(active)] = True
    return sets, cls, pool_active


def random_sums(mask, matching, cls, pool_active, values, draws, rng, member):
    """Random sums matched by class and member status (fixed images stay fixed across subsets).

    values: list of [J, R] arrays (one per target). Returns list of [draws, J]."""
    J, R = mask.shape
    out = [np.zeros((draws, J)) for _ in values]
    if matching == 'per_subset':
        for j in range(J):
            for k in np.unique(cls[mask[j]]):
                for membership in (False, True):
                    stratum = (cls == k) & (member == membership)
                    pool = np.flatnonzero(stratum)
                    n = int((mask[j] & stratum).sum())
                    if not n:
                        continue
                    if n == len(pool):
                        idx = np.broadcast_to(pool, (draws, n))
                    else:
                        idx = pool[np.argpartition(rng.random((draws, len(pool))), n - 1, axis=1)[:, :n]]
                    for o, v in zip(out, values):
                        o[:, j] += v[j][idx].sum(1)
    else:
        m = mask[0]
        for k in np.unique(cls[m]):
            pool = np.flatnonzero(pool_active & (cls == k))
            n = int((m & (cls == k)).sum())
            idx = pool[np.argpartition(rng.random((draws, len(pool))), n - 1, axis=1)[:, :n]]
            for o, v in zip(out, values):
                o += v[:, idx].sum(-1).T      # [draws, J]
    return out


def analyze(targets, sets, cls, pool_active, draws, seed, member):
    rng = np.random.default_rng(seed)
    names = list(targets)
    values = [targets[t][v] for t in names for v in ('if', 'true')]
    rows, subset_rows = [], []
    for sname, (family, mask, matching) in sets.items():
        rand = random_sums(mask, matching, cls, pool_active, values, draws, rng, member) if draws else None
        for ti, t in enumerate(names):
            tg = targets[t]
            tif, ttrue = tg['if'].sum(1), tg['true'].sum(1)
            cand = scores((tg['if'] * mask).sum(1), (tg['true'] * mask).sum(1), tif, ttrue)
            rq = scores(rand[2 * ti], rand[2 * ti + 1], tif, ttrue) if rand is not None else None
            for q in QUANTITIES:
                x, ref = cand[q]
                y = tg['truth'] if ref is None else ref
                ok = np.ptp(x) > 0 and np.ptp(y) > 0
                row = dict(set=sname, family=family, quantity=q, target=t, rows_per_subset=float(mask.sum(1).mean()),
                           lds=C.spearman(x, y) if ok else float('nan'))
                if rq is not None:
                    xr, refr = rq[q]
                    yr = np.broadcast_to(tg['truth'], xr.shape) if refr is None else refr
                    rl = C.spearman_rows(xr, yr)
                    rl = rl[np.isfinite(rl)]
                    if len(rl):
                        row.update(random_n=len(rl), random_mean=float(rl.mean()),
                                   random_p025=float(np.quantile(rl, .025)), random_p975=float(np.quantile(rl, .975)),
                                   frac_random_ge=float(np.mean(rl >= row['lds'])) if ok else float('nan'))
                rows.append(row)
                subset_rows.extend(dict(set=sname, quantity=q, target=t, subset=j, score=x[j], reference=y[j])
                                   for j in range(len(x)))
    return rows, subset_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', required=True, help='frozen interface checkout')
    p.add_argument('--out', required=True, help='interface v1r1 run directory')
    p.add_argument('--dest', required=True)
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261102)
    a = p.parse_args()
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    c, data, ai, fixed = C.load_context(a.repo, a.out)
    data['_ai'] = ai
    images = C.readcsv(Path(a.out) / 'case_study/gamma0.03_k100/images.csv')
    targets, dev = contributions(c, data, fixed)
    checks = dict(row1_reconstruction_max_rel=dev, row1_reference=ROW1_REFERENCE,
                  lds_all_if={t: C.spearman(v['if'].sum(1), v['truth']) for t, v in targets.items()},
                  lds_all_true={t: C.spearman(v['true'].sum(1), v['truth']) for t, v in targets.items()})
    if dev > 1e-10 or abs(checks['lds_all_if']['combined'] - ROW1_REFERENCE) > 1e-9:
        raise ValueError(f'does not reproduce bridge row 1: {checks}')
    sets, cls, pool_active = row_sets(data, fixed, images)
    member = np.arange(len(cls)) < data['M'].shape[1]
    rows, subset_rows = analyze(targets, sets, cls, pool_active, a.draws, a.seed, member)
    get = lambda s_, q: next(r['lds'] for r in rows if r['set'] == s_ and r['quantity'] == q and r['target'] == 'combined')
    # Store useful reconstruction diagnostics without embedding prior result claims.
    checks['bridge_rows'] = dict(row2_all_true=checks['lds_all_true']['combined'], row3_deleted=get('deleted', 'replace_if_by_true'),
                                 row4_retained=get('retained', 'replace_if_by_true'), row5_nonmember=get('nonmember', 'replace_if_by_true'),
                                 keep_if_deleted=get('deleted', 'keep_if'), keep_true_deleted=get('deleted', 'keep_true'))
    (dest / 'a_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    C.writecsv(dest / 'a_results.csv', rows, ['set', 'family', 'quantity', 'target', 'rows_per_subset', 'lds',
                                              'random_n', 'random_mean', 'random_p025', 'random_p975', 'frac_random_ge'])
    C.writecsv(dest / 'a_subset_values.csv', subset_rows)
    print(json.dumps(checks, indent=2))
    for r in rows:
        if r['target'] == 'combined' and r['family'] in ('group', 'group_x_class'):
            rr = f"  random {r['random_mean']:.3f} [{r['random_p025']:.3f}, {r['random_p975']:.3f}]" if 'random_mean' in r else ''
            print(f"{r['set']:20s} {r['quantity']:19s} LDS={r['lds']:.3f}{rr}")


if __name__ == '__main__':
    main()
