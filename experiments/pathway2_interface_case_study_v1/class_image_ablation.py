"""Retrain Stage-2 attacks after removing selected shadow-3 training-table rows.

One worker per (selection, Stage-1 seed); collect only after all workers finish.
Uses the original fixed target queries, saved Stage-1 IF and retrained probabilities.
Group 'none' removes nothing: it re-runs the original attacks on this hardware and is compared with the stored
v6 truth (replay) and with the removal groups, so a change in LDS can be told apart from run-to-run drift.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
GROUPS = ('none', 'low_margin', 'low_margin_random', 'high_if', 'high_if_random')
PANEL = (42, 43, 44)
N_SELECTED = 50          # default size of every removed set; --n-selected changes it
RANDOM_SEED = 20261030


def selections(images, n=N_SELECTED):
    """Return positions in the affected shadow-3 block, with class-matched controls."""
    active = [r for r in images if r['attack_row_active'].lower() == 'true']
    low = sorted(active, key=lambda r: (float(r['margin']), int(r['train_position'])))[:n]
    high = sorted(active, key=lambda r: (-abs(float(r['a_if'])), int(r['train_position'])))[:n]
    rng = np.random.default_rng(RANDOM_SEED)

    def matched(chosen):
        result = []
        for cls in sorted({int(r['cls']) for r in chosen}):
            pool = [r for r in active if int(r['cls']) == cls]
            n = sum(int(r['cls']) == cls for r in chosen)
            result.extend(pool[i] for i in rng.choice(len(pool), size=n, replace=False))
        return result

    return dict(none=[], low_margin=low, low_margin_random=matched(low), high_if=high, high_if_random=matched(high))


def kept_training_indices(fixed, selected, cls):
    """Only the selected member rows of affected shadow 3 are removed."""
    aff = np.flatnonzero(fixed['shadow_id'] == 3)
    positions = np.array([int(r['train_position']) for r in selected], dtype=int)
    raws = np.array([int(r['raw_index']) for r in selected], dtype=int)
    if not np.array_equal(fixed['raw_index'][aff[positions]], raws):
        raise ValueError('Selected image IDs do not match the affected shadow-3 rows')
    if not np.array_equal(fixed['classes'][aff[positions]], np.array([int(r['cls']) for r in selected])):
        raise ValueError('Selected image classes do not match the affected shadow-3 rows')
    if np.any(fixed['membership'][aff[positions]] != 1):
        raise ValueError('Selected rows must be shadow-3 member rows')
    keep = np.ones(len(fixed['classes']), dtype=bool)
    keep[aff[positions]] = False
    return np.flatnonzero((fixed['classes'] == cls) & keep)


def train_condition(A, pilot, hp, fixed, x, cls, seed, selected, device, gamma=None):
    """Rebuild the attack from the filtered table; keep target query rows fixed."""
    train_idx = kept_training_indices(fixed, selected, cls)
    query = fixed['q_classes'] == cls
    model = A.train(pilot, hp, x[train_idx], fixed['membership'][train_idx], seed)
    metric = A.evaluate(pilot, model, fixed['q_x'][query], fixed['q_membership'][query])
    v = None
    if gamma is not None:
        adj = A.adjoint(model, x[train_idx], fixed['membership'][train_idx],
                        fixed['q_x'][query], fixed['q_membership'][query], [gamma], device)
        entry = adj['per_gamma'][float(gamma)]
        if not entry['solved'] or not np.isfinite(entry['solve_rel_residual']):
            raise RuntimeError('Attack adjoint solve failed')
        affected = fixed['shadow_id'][train_idx] == 3
        block = np.flatnonzero(fixed['shadow_id'] == 3)
        positions = np.searchsorted(block, train_idx[affected])
        v = np.zeros((len(block), x.shape[1]), dtype=float)
        v[positions] = entry['v'][affected]
    return float(metric['cross_entropy']), v


def load_interface(repo, out):
    sys.path.insert(0, str(Path(repo).resolve() / 'experiments/pathway2_interface_lds_v1'))
    import run
    import analysis_if
    c = run.config(str(out))
    data = analysis_if.load(Path(c['out']), c)
    return c, data, analysis_if


def image_rows(out):
    path = Path(out) / 'case_study/gamma0.03_k100/images.csv'
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def worker(repo, out, dest, task, n=N_SELECTED):
    c, data, ai = load_interface(repo, out)
    import attack4 as A
    from core_if import ISession
    panel = data['frozen']['score_panel']
    if list(panel) != list(PANEL):
        raise ValueError(f'This run expects the existing panel {PANEL}; got {panel}')
    group, seed = GROUPS[task // len(PANEL)], PANEL[task % len(PANEL)]
    images = image_rows(out)
    selected = selections(images, n)[group]
    if len(selected) != (0 if group == 'none' else n):
        raise ValueError('Not enough attack-active training images')
    session = ISession(c)
    with np.load(Path(c['v6_out']) / 'smoke/fixed_rows.npz') as z:
        fixed = {k: z[k] for k in z.files}
    if not np.array_equal(fixed['raw_index'][fixed['shadow_id'] == 3], data['rows']['raw_index']):
        raise ValueError('Attack table and interface output rows are not aligned')
    fr = data['fr6']
    classes = fr['qualified_classes']
    attack_seeds = fr['attack_seeds']
    gamma_a = float(fr['attack_gamma_star'])
    pfull = data['full'][seed]
    N = data['M'].shape[1]
    arrays = data['solves'][seed]['arrays']
    gamma = float(c['primary_gamma'])
    gi = list(c['gammas']).index(gamma)
    dz = ai.matched_dz(arrays['dz_if'][:, gi], arrays['dz_gall'][gi], data['M'].sum(1), N)
    pred = ai.softmax_jvp(arrays['p0'], dz)
    J = len(data['M'])
    score = np.zeros(J)
    vbar = np.zeros_like(pfull, dtype=float)
    truth = np.zeros(J)
    rows = []
    full_x = A.assemble(fixed, pfull)
    for cls in classes:
        omega = float(fr['omega'][str(cls)])
        for k in attack_seeds:
            j00, v = train_condition(A, session.pilot, session.hp, fixed, full_x, cls, k,
                                     selected, session.dev, gamma_a)
            score += omega * np.einsum('jrc,rc->j', pred, v) / len(attack_seeds)
            vbar += omega * v / len(attack_seeds)
            for j in range(J):
                psub = pfull + data['truth'][c['seeds'].index(seed), j]
                j10, _ = train_condition(A, session.pilot, session.hp, fixed, A.assemble(fixed, psub),
                                         cls, k, selected, session.dev)
                truth[j] += omega * (j10 - j00) / len(attack_seeds)
                rows.append(dict(subset=j, cls=cls, attack_seed=k, J00=j00, J10=j10))
            print(f'{group} seed={seed} class={cls} attack_seed={k} finished', flush=True)
    d = Path(dest) / group
    d.mkdir(parents=True, exist_ok=True)
    if group == 'none':     # same attacks as v6 r1, rerun here: how far are they from the stored truth?
        stored = data['v6_truth']
        gap = dict(J10_minus_J00_max_abs=0.0, J00_max_abs=0.0, J10_max_abs=0.0)
        for j in range(J):
            old_by_key = {(r['cls'], r['attack_seed']): r for r in stored[(j, seed)]}
            for new in (r for r in rows if r['subset'] == j):
                old = old_by_key[(new['cls'], new['attack_seed'])]
                gap['J00_max_abs'] = max(gap['J00_max_abs'], abs(new['J00'] - old['J00']))
                gap['J10_max_abs'] = max(gap['J10_max_abs'], abs(new['J10'] - old['J10']))
                gap['J10_minus_J00_max_abs'] = max(gap['J10_minus_J00_max_abs'],
                    abs((new['J10'] - new['J00']) - (old['J10'] - old['J00'])))
        original_v = data['vbar'][seed]
        old_score = np.einsum('jrc,rc->j', pred, original_v)
        gap['vbar_max_abs'] = float(np.max(np.abs(vbar - original_v)))
        gap['vbar_relative_l2'] = float(np.linalg.norm(vbar - original_v) / max(np.linalg.norm(original_v), 1e-300))
        gap['score_max_abs'] = float(np.max(np.abs(score - old_score)))
        (d / f'seed{seed}_replay.json').write_text(json.dumps(gap))
        print(f'replay of stored v6 truth, seed {seed}: max abs {gap}', flush=True)
    fields = list(images[0])
    with (d / f'seed{seed}_selected.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(selected)
    with (d / f'seed{seed}_truth.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(d / f'seed{seed}.npz', score=score, truth=truth, group=group,
                        n_removed=len(selected), stage1_seed=seed)


def collect(repo, out, dest, n=N_SELECTED):
    from scipy.stats import spearmanr
    c, data, ai = load_interface(repo, out)
    fr = data['fr6']
    J = len(data['M'])
    base_score, base_truth = [], []
    for seed in PANEL:
        arr = data['solves'][seed]['arrays']
        gi = list(c['gammas']).index(float(c['primary_gamma']))
        dz = ai.matched_dz(arr['dz_if'][:, gi], arr['dz_gall'][gi], data['M'].sum(1), data['M'].shape[1])
        dp = ai.softmax_jvp(arr['p0'], dz)
        base_score.append(np.einsum('jrc,rc->j', dp, data['vbar'][seed]))
        t = np.zeros(J)
        for j in range(J):
            for cls in fr['qualified_classes']:
                vals = [r['J10'] - r['J00'] for r in data['v6_truth'][(j, seed)] if r['cls'] == cls]
                t[j] += float(fr['omega'][str(cls)]) * np.mean(vals)
        base_truth.append(t)
    methods = [('stored baseline (v6 r1)', np.mean(base_score, axis=0), np.mean(base_truth, axis=0))]
    for group in GROUPS:
        parts = []
        for seed in PANEL:
            with np.load(Path(dest) / group / f'seed{seed}.npz') as z:
                expected_n = 0 if group == 'none' else n
                if str(z['group']) != group or int(z['stage1_seed']) != seed or int(z['n_removed']) != expected_n:
                    raise ValueError(f'Mismatched output for {group}, seed {seed}, n={n}')
                parts.append((z['score'], z['truth']))
        methods.append((group, np.mean([x[0] for x in parts], axis=0), np.mean([x[1] for x in parts], axis=0)))
    with (Path(dest) / 'results.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['group', 'removed_shadow3_member_rows', 'lds'])
        for group, score, truth in methods:
            value = float(spearmanr(score, truth).statistic)
            w.writerow([group, n if group in ('low_margin', 'low_margin_random', 'high_if', 'high_if_random') else 0, value])
            print(f'{group:26s} LDS={value:.6f}', flush=True)
    for seed in PANEL:
        gap = json.loads((Path(dest) / 'none' / f'seed{seed}_replay.json').read_text())
        print(f'replay of the stored truth, Stage-1 seed {seed}: max abs {gap}', flush=True)
    with (Path(dest) / 'subset_values.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['group', 'subset', 'predicted', 'truth'])
        for group, score, truth in methods:
            w.writerows((group, j, score[j], truth[j]) for j in range(J))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--task', type=int)
    p.add_argument('--n-selected', type=int, default=N_SELECTED, help='rows removed per selection')
    p.add_argument('--collect', action='store_true')
    a = p.parse_args()
    if a.n_selected < 1:
        p.error('--n-selected must be positive')
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    if a.collect:
        collect(a.repo, a.out, a.dest, a.n_selected)
    else:
        if a.task is None or not 0 <= a.task < len(GROUPS) * len(PANEL):
            p.error(f'task must be 0..{len(GROUPS) * len(PANEL) - 1}')
        worker(a.repo, a.out, a.dest, a.task, a.n_selected)


if __name__ == '__main__':
    main()
