"""Numerical checks of class_image_ablation.py on synthetic data (run locally; not a Slurm step)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import class_image_ablation as ab

rng = np.random.default_rng(3)
S, J, N, Rn = 3, 32, 120, 80
R = N + Rn
M = np.zeros((J, N))
for j in range(J):
    M[j, rng.permutation(N)[:N // 2]] = 1
y = rng.integers(0, 4, R)
omega = {1: .6, 2: .4}
V = {c: rng.normal(size=(S, R, 4)) * (y == c)[None, :, None] for c in (1, 2)}      # class sensitivity: only rows of that class
vbc = {c: omega[c] * V[c] for c in omega}
vb = vbc[1] + vbc[2]
T = rng.normal(size=(S, J, R, 4)) * .05
P = .2 * T + rng.normal(size=(S, J, R, 4)) * .01
v6 = {(j, s): [dict(cls=c, J10=float(rng.normal()), J00=0.) for _ in range(2) for c in (1, 2)] for j in range(J) for s in (42, 43, 44)}
panel = [42, 43, 44]

mk = lambda v: ab.Bridge(ab.terms(v, P), ab.terms(v, T), M, y)
bridges = {'combined': mk(vb), 'DME': mk(vbc[1]), 'DRUSEN': mk(vbc[2])}
br = bridges['combined']
Ta_c = ab.class_truth(v6, panel, omega)
Ta = ab.attack_truth(v6, panel, omega, 2)
chk = ab.checks(bridges, Ta, Ta_c, omega, vb, vbc, y)
print({k: round(v, 12) for k, v in chk.items()})
assert chk['class_scores_sum_to_row1_max_abs'] < 1e-12 and chk['class_truths_recombine_to_attack_truth_max_abs'] < 1e-12
assert chk['class_vbar_sum_vs_mixed_vbar_max_abs'] < 1e-12 and chk['vbar_max_abs_on_rows_outside_attack_classes'] == 0
assert chk['vbar_DME_max_abs_on_rows_of_other_classes'] == 0 and chk['vbar_DRUSEN_max_abs_on_rows_of_other_classes'] == 0
# a class sensitivity that leaks onto another class's rows must show up in the check
leak = {1: vbc[1] + 0.01 * (y == 2)[None, :, None], 2: vbc[2]}
assert ab.checks(bridges, Ta, Ta_c, omega, vb, leak, y)['vbar_DME_max_abs_on_rows_of_other_classes'] > 0

# E1: diagonal and mixed pairs
e1 = ab.e1_classes(bridges, Ta, Ta_c)
assert set(e1.score_of) == {'combined', 'DME', 'DRUSEN'} and set(e1.truth) == {'combined', 'DME', 'DRUSEN'}
r1 = e1[(e1.score == 'row 1 all IF') & (e1.score_of == 'DME') & (e1.truth == 'DME')].lds.iloc[0]
assert abs(r1 - ab.rho(bridges['DME'].score(None), Ta_c[1])) < 1e-12

# gradient-norm alignment
import pandas as pd
raw = np.arange(1000, 1000 + R)
f = pd.DataFrame(dict(train_position=np.arange(N)[::-1], raw_index=raw[:N][::-1], cls=y[:N][::-1], grad_norm=np.arange(N, dtype=float)))
gn = ab.align_grad_norm(f, raw, y, N)
assert gn[N - 1] == 0 and gn[0] == N - 1
for bad in (f.assign(raw_index=f.raw_index + 1), f.assign(cls=(f.cls + 1) % 4), f.iloc[:-1], f.assign(grad_norm=np.nan)):
    try:
        ab.align_grad_norm(bad, raw, y, N)
        raise SystemExit('misaligned features file was accepted')
    except ValueError:
        pass

# loop reference for rows 1 and 3
ref1 = np.mean([[(vb[k] * P[k, j]).sum() for j in range(J)] for k in range(S)], 0)
ref3 = []
for k in range(S):
    X = P[k].copy()
    for j in range(J):
        rr = np.flatnonzero(M[j] == 0)
        X[j, rr] = T[k, j, rr]
    ref3.append((vb[k][None] * X).sum((1, 2)))
ref3 = np.mean(ref3, 0)
print('row 1 vs loop', np.abs(br.score(None) - ref1).max(), '| row 3 vs loop', np.abs(br.score(br.deleted) - ref3).max())
assert np.abs(br.score(None) - ref1).max() < 1e-12 and np.abs(br.score(br.deleted) - ref3).max() < 1e-12

# image sets: the image-level variants reduce to the bridge rows at the extremes
img = ab.own_terms(M, T, P, vb, np.exp(rng.normal(size=(R, 4))) / 4, y)
sets, act = ab.image_sets(img)
assert ('class DME', int(((y[:N] == 1) & img.active.values).sum())) in sets
assert set(act) == set(np.flatnonzero(np.isin(y[:N], [1, 2])))
e2, base = ab.e2_images(br, Ta, {('all active', len(act)): act, ('none', 0): np.array([], int)}, act, np.random.default_rng(0))
g = lambda s_, v: float(e2[(e2.set == s_) & (e2.variant == v)].lds.iloc[0])
row1, row3 = ab.rho(br.score(None), Ta), ab.rho(br.score(br.deleted), Ta)
print('fix(all active) = row 3:', g('all active', 'fix, row 1 + S deleted rows retrained'), row3)
print('leave(all active) = row 1:', g('all active', 'leave, row 3 with S left at IF'), row1)
print('drop(none) = rows 1 / 3 base:', base['drop, row 1'], base['drop, row 3'])
assert abs(g('all active', 'fix, row 1 + S deleted rows retrained') - row3) < 1e-12
assert abs(g('all active', 'leave, row 3 with S left at IF') - row1) < 1e-12
assert abs(base['drop, row 1'] - row1) < 1e-12 and abs(base['drop, row 3'] - row3) < 1e-12
# dropping all active images leaves nothing in the score (every active row is out), a constant: undefined LDS
# whole-class sets carry no random control; other sets do
cs = {('class DME', 10): np.flatnonzero(img.active.values & (img.cls.values == 1))[:10], ('lowest margin', 10): act[:10]}
e2b, _ = ab.e2_images(br, Ta, cs, act, np.random.default_rng(0))
assert e2b[e2b.set == 'class DME'].random_mean.isna().all() and e2b[e2b.set == 'lowest margin'].random_mean.notna().all()
print('all synthetic checks passed')
