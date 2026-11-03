#!/usr/bin/env python3
"""Method B: summarise true deletions from the attack training table (attack retrained, v and truth recomputed).

Reads the existing runs without changing them:
  --v2-dest   attack_prune_supplement_v2 directory (class_member groups, other-shadow controls, n50/n200
              random2..5, Oracle and Oracle random1..5); groups that are not complete for all three
              Stage-1 seeds are listed and skipped, so the v2 package's own collect is not needed
  manifest old50 / old200   the first runs (none, low_margin, high_if, random1; combined values only)

For every group and target: LDS, and how far deletion moved each side relative to none:
  truth_vs_none / score_vs_none          Spearman of the new truth / prediction with none's
  old_score_x_new_truth                  none's prediction against the new truth (truth side only)
  new_score_x_old_truth                  new prediction against none's truth (prediction side only)
  truth_mean_over_none, truth_sd_over_none, score_sd_over_none
Summaries compare each candidate with its matched controls (5 random sets, or 5 other-shadow sets).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common as C

TNAMES = ('combined', 'DME', 'DRUSEN')


def verify(m):
    bad = [p for p, h in m['pins'].items() if not Path(p).is_file() or C.sha(p) != h]
    if bad:
        raise ValueError(f'pinned inputs changed: {bad[:5]}')


def baseline(c, data, ai):
    """none, per class: score [2, J] and truth [2, J] averaged over the panel (bitwise the v6 values)."""
    qualified = [int(x) for x in data['fr6']['qualified_classes']]
    sc = []
    for s in C.PANEL:
        dp = C.if_dp(c, data, ai, s)
        vb = C.class_vbar(c, data, s)
        sc.append(np.array([np.einsum('jrk,rk->j', dp, vb[k]) for k in qualified]))
    T = C.class_truth(data)
    return np.mean(sc, 0), np.array([T[k] for k in qualified])


def targets(score, truth, w):
    """score/truth either [2, J] (per class) or [J] (combined only)."""
    if np.ndim(score) == 1:
        return {'combined': (score, truth)}
    return {'combined': (w @ score, w @ truth), 'DME': (score[0], truth[0]), 'DRUSEN': (score[1], truth[1])}


def describe(name, kind, n, t, x, y, bx, by):
    sd = lambda v: float(np.std(v, ddof=1))
    return dict(group=name, kind=kind, n_removed=n, target=t, lds=C.spearman(x, y),
                truth_vs_none=C.spearman(y, by), score_vs_none=C.spearman(x, bx),
                old_score_x_new_truth=C.spearman(bx, y), new_score_x_old_truth=C.spearman(x, by),
                truth_mean_over_none=float(np.mean(y) / np.mean(by)), truth_sd_over_none=sd(y) / sd(by),
                score_sd_over_none=sd(x) / sd(bx))


def load_group(d, name, n_expected=None, *, manifest_sha256=None, weights=None, expected_j=32):
    """Load one complete group. New v2 files must match the current manifest and weights.

    Legacy files predate the manifest and have a different seed field; they are
    independently pinned in the v2 manifest and validated using their own metadata.
    """
    paths = [Path(d) / name / f'seed{s}.npz' for s in C.PANEL]
    if any(not f.exists() for f in paths):
        return None
    vals = []
    for s, f in zip(C.PANEL, paths):
        with np.load(f) as z:
            if n_expected is not None and int(z['n_removed']) != n_expected:
                raise ValueError(f'{f}: removed {int(z["n_removed"])}, expected {n_expected}')
            if str(z['group']) != name:
                raise ValueError(f'{f}: group does not match {name}')
            if manifest_sha256 is not None:
                if int(z['seed']) != s or str(z['manifest_sha256']) != manifest_sha256:
                    raise ValueError(f'{f}: seed or manifest mismatch')
                if weights is None or not np.array_equal(z['weights'], weights):
                    raise ValueError(f'{f}: class weights mismatch')
                score, truth = z['score_by_class'], z['truth_by_class']
                shape = (len(weights), expected_j)
            else:
                if int(z['stage1_seed']) != s:
                    raise ValueError(f'{f}: Stage-1 seed mismatch')
                score, truth = z['score'], z['truth']
                shape = (expected_j,)
            if score.shape != shape or truth.shape != shape or not np.isfinite(score).all() or not np.isfinite(truth).all():
                raise ValueError(f'{f}: invalid score or truth array')
            vals.append((score.copy(), truth.copy()))
    return np.mean([v[0] for v in vals], 0), np.mean([v[1] for v in vals], 0)


def summarise(rows, candidate, controls, target):
    by = {(r['group'], r['target']): r for r in rows}
    if (candidate, target) not in by:
        return None
    got = [by[(x, target)] for x in controls if (x, target) in by]
    cand = by[(candidate, target)]
    out = dict(candidate=candidate, target=target, n_removed=cand['n_removed'], candidate_lds=cand['lds'],
               candidate_truth_mean_over_none=cand['truth_mean_over_none'], candidate_truth_vs_none=cand['truth_vs_none'],
               controls_found=len(got), controls_expected=len(controls))
    if got:
        l = np.array([g['lds'] for g in got]); m = np.array([g['truth_mean_over_none'] for g in got])
        out.update(control_lds_mean=float(l.mean()), control_lds_min=float(l.min()), control_lds_max=float(l.max()),
                   candidate_outside_control_lds_range=bool(cand['lds'] < l.min() or cand['lds'] > l.max()),
                   control_truth_mean_over_none_mean=float(m.mean()), control_truth_mean_over_none_min=float(m.min()),
                   control_truth_mean_over_none_max=float(m.max()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v2-dest', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    dest = Path(a.v2_dest)
    m = json.loads((dest / 'manifest.json').read_text())
    verify(m)
    c, data, ai, fixed = C.load_context(m['repo'], m['out'])
    om = C.omega(data)
    w = np.array([om[k] for k in (1, 2)])
    bs, bt = baseline(c, data, ai)
    base = targets(bs, bt, w)
    # none of the first run must equal the stored baseline (it replayed v6 exactly)
    old_none = load_group(m['old50'], 'none', 0, expected_j=len(data['M']))
    if old_none is None:
        raise FileNotFoundError('first-run n50 none is incomplete')
    if not (np.allclose(old_none[0], w @ bs, rtol=1e-10, atol=1e-14) and np.allclose(old_none[1], w @ bt, rtol=1e-10, atol=1e-14)):
        raise ValueError('first-run none differs from the stored v6 baseline')
    rows, missing = [], []
    for t, (x, y) in base.items():
        rows.append(describe('none', 'baseline', 0, t, x, y, x, y))
    groups = list(m['groups']) + [dict(name=al, kind='alias_of_' + tg, n=next(g['n'] for g in m['groups'] if g['name'] == tg),
                                       source=tg) for al, tg in m.get('aliases', {}).items()]
    for g in groups:
        got = load_group(dest / 'results', g.get('source', g['name']), g['n'],
                         manifest_sha256=C.sha(dest / 'manifest.json'), weights=w, expected_j=len(data['M']))
        if got is None:
            missing.append(g['name'])
            continue
        for t, (x, y) in targets(got[0], got[1], w).items():
            rows.append(describe(g['name'], g['kind'], g['n'], t, x, y, *base[t]))
    sizes = [n for n, key in ((50, 'old50'), (200, 'old200')) if m.get(key)]
    for n in sizes:
        for name in ('low_margin', 'low_margin_random', 'high_if', 'high_if_random'):
            got = load_group(m['old50' if n == 50 else 'old200'], name, n, expected_j=len(data['M']))
            label = f'n{n}_{name}' + ('1' if name.endswith('_random') else '')
            if got is None:
                missing.append(label)
                continue
            for t, (x, y) in targets(got[0], got[1], w).items():
                rows.append(describe(label, 'first_run', n, t, x, y, *base[t]))
    summary = []
    for g in m['groups']:
        if g['kind'] == 'class_member':
            ctrl = [f"{g['name']}_other_random{k}" for k in range(1, 6)]
            for t in TNAMES:
                s = summarise(rows, g['name'], ctrl, t)
                if s:
                    summary.append(dict(s, control='other_shadow'))
    for n in sizes:
        for name in ('low_margin', 'high_if', 'oracle'):
            cand = f'n{n}_{name}'
            ctrl = [f'{cand}_random{k}' for k in range(1, 6)]
            for t in TNAMES:
                s = summarise(rows, cand, ctrl, t)
                if s:
                    summary.append(dict(s, control='matched_random'))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    C.writecsv(out / 'b_results.csv', rows)
    C.writecsv(out / 'b_summary.csv', summary, sorted({k for r in summary for k in r}, key=lambda k: (k != 'candidate', k)))
    (out / 'b_missing_groups.json').write_text(json.dumps(missing, indent=2) + '\n')
    print(f'{len(missing)} groups not complete (skipped): {missing}')
    for s in summary:
        rng = (f"controls {s['control_lds_mean']:.3f} [{s['control_lds_min']:.3f}, {s['control_lds_max']:.3f}] "
               f"truth/none {s['candidate_truth_mean_over_none']:.2f} vs {s['control_truth_mean_over_none_mean']:.2f}"
               if 'control_lds_mean' in s else 'no controls yet')
        print(f"{s['candidate']:28s} {s['target']:8s} LDS={s['candidate_lds']:.3f}  {rng}  ({s['controls_found']}/5)")


if __name__ == '__main__':
    main()
