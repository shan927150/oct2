"""Checks that an actual training-row deletion reaches both model fits and adjoints."""
import numpy as np
import class_image_ablation as a


def test_training_and_adjoint_use_identical_filtered_table():
    fixed = dict(x=np.arange(24, dtype=float).reshape(6, 4),
                 shadow_id=np.array([0, 0, 3, 3, 3, 3]),
                 classes=np.array([1, 1, 1, 1, 1, 1]),
                 raw_index=np.array([10, 11, 20, 21, 22, 23]),
                 membership=np.array([1, 0, 1, 1, 0, 0]),
                 q_classes=np.array([1, 1]), q_x=np.ones((2, 4)), q_membership=np.array([0, 1]))
    selected = [dict(train_position='1', raw_index='21', cls='1')]
    seen = {}

    class Attack:
        @staticmethod
        def train(pilot, hp, x, y, seed):
            seen['train'] = x.copy()
            return object()

        @staticmethod
        def evaluate(pilot, model, qx, qm):
            return dict(cross_entropy=float(len(seen['train'])))

        @staticmethod
        def adjoint(model, x, y, qx, qm, gammas, device):
            seen['adjoint'] = x.copy()
            return dict(per_gamma={0.3: dict(solved=True, solve_rel_residual=0., v=np.ones_like(x))})

    assert np.array_equal(a.kept_training_indices(fixed, [], 1), np.arange(6))      # nothing selected: the full table
    j00, v = a.train_condition(Attack, None, None, fixed, fixed['x'], 1, 5, selected, 'cpu', 0.3)
    j10, _ = a.train_condition(Attack, None, None, fixed, fixed['x'] + 1, 1, 5, selected, 'cpu')
    assert j00 == j10 == 5
    assert np.array_equal(seen['adjoint'], fixed['x'][[0, 1, 2, 4, 5]])
    assert np.array_equal(seen['train'], (fixed['x'] + 1)[[0, 1, 2, 4, 5]])
    assert np.array_equal(v[:, 0], [1., 0., 1., 1.])


def test_controls_match_class_and_are_reproducible():
    rows = [dict(attack_row_active='True', cls=str(i % 2), train_position=str(i), raw_index=str(i),
                 margin=str(i), a_if=str(i)) for i in range(120)]
    x, y = a.selections(rows), a.selections(rows)
    for name in a.GROUPS:
        assert len(x[name]) == (0 if name == 'none' else 50) and x[name] == y[name]
    assert all(len(v) == (0 if k == 'none' else 20) for k, v in a.selections(rows, 20).items())
    for candidate, control in [('low_margin', 'low_margin_random'), ('high_if', 'high_if_random')]:
        assert sorted(r['cls'] for r in x[candidate]) == sorted(r['cls'] for r in x[control])


def test_worker_retrains_each_truth_on_filtered_rows():
    import csv
    import sys
    import tempfile
    import types
    from pathlib import Path

    n, jcount = 120, 3
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'v6/smoke').mkdir(parents=True)
        imgdir = root / 'out/case_study/gamma0.03_k100'
        imgdir.mkdir(parents=True)
        fixed = dict(x=np.zeros((n, 4)), shadow_id=np.full(n, 3), classes=np.arange(n) % 2 + 1,
                     raw_index=np.arange(n), membership=np.ones(n, dtype=int),
                     q_classes=np.array([1, 2]), q_x=np.ones((2, 4)), q_membership=np.array([1, 1]))
        np.savez(root / 'v6/smoke/fixed_rows.npz', **fixed)
        img = [dict(train_position=i, raw_index=i, cls=i % 2 + 1, attack_row_active='True',
                    margin=i, a_if=i) for i in range(n)]
        with (imgdir / 'images.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(img[0])); w.writeheader(); w.writerows(img)
        cfg = dict(v6_out=str(root / 'v6'), seeds=[42, 43, 44], primary_gamma=0.03, gammas=[0.03])
        change = np.empty((3, jcount, n, 4))
        for j in range(jcount):
            change[:, j] = j + 1
        arrays = dict(dz_if=np.ones((jcount, 1, n, 4)), dz_gall=np.zeros((1, n, 4)), p0=np.zeros((n, 4)))
        stored = {(j, 42): [dict(cls=c_, attack_seed=10, J10=j + 1., J00=0.) for c_ in (1, 2)] for j in range(jcount)}
        data = dict(frozen=dict(score_panel=[42, 43, 44]), rows=dict(raw_index=np.arange(n)), v6_truth=stored,
                    fr6=dict(qualified_classes=[1, 2], attack_seeds=[10], attack_gamma_star=0.3,
                             omega={'1': 0.5, '2': 0.5}), M=np.ones((jcount, n)),
                    full={42: np.zeros((n, 4))}, truth=change, solves={42: dict(arrays=arrays)},
                    vbar={42: np.full((n, 4), 0.5)})
        ai = types.SimpleNamespace(matched_dz=lambda dz, *_: dz, softmax_jvp=lambda p, dz: dz)
        attack = types.ModuleType('attack4')
        attack.assemble = lambda f, p: p.copy()
        attack.train = lambda pilot, hp, x, y, seed: x.copy()
        attack.evaluate = lambda pilot, model, qx, qm: dict(cross_entropy=float(model.mean()))
        attack.adjoint = lambda model, x, y, qx, qm, gamma, device: dict(per_gamma={0.3:
            dict(solved=True, solve_rel_residual=0., v=np.ones_like(x))})
        core = types.ModuleType('core_if')
        core.ISession = lambda c: types.SimpleNamespace(pilot=None, hp=None, dev='cpu')
        old_load, old_attack, old_core = a.load_interface, sys.modules.get('attack4'), sys.modules.get('core_if')
        try:
            a.load_interface = lambda *args: (cfg, data, ai)
            sys.modules['attack4'], sys.modules['core_if'] = attack, core
            a.worker('unused', root / 'out', root / 'result', 0)            # group 'none', Stage-1 seed 42
            with np.load(root / 'result/none/seed42.npz') as result:
                assert np.allclose(result['truth'], [1, 2, 3]) and np.allclose(result['score'], 240)
            import json
            gap = json.loads((root / 'result/none/seed42_replay.json').read_text())
            assert all(value == 0.0 for value in gap.values())
            a.worker('unused', root / 'out', root / 'result', 3)            # group 'low_margin', Stage-1 seed 42
            with np.load(root / 'result/low_margin/seed42.npz') as result:
                assert np.allclose(result['truth'], [1, 2, 3])
                assert np.allclose(result['score'], 140)
            with (root / 'result/low_margin/seed42_truth.csv').open() as f:
                assert len(list(csv.DictReader(f))) == 6
        finally:
            a.load_interface = old_load
            for key, value in [('attack4', old_attack), ('core_if', old_core)]:
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


def test_collect_keeps_n_and_seed_outputs_aligned():
    import csv
    import json
    import tempfile
    import types
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        panel, J = a.PANEL, 4
        vec = np.array([1., 4., 2., 3.])
        arr = dict(dz_if=vec.reshape(J, 1, 1, 1), dz_gall=np.zeros((1, 1, 1)), p0=np.zeros((1, 1)))
        cfg = dict(primary_gamma=0.03, gammas=[0.03])
        fr = dict(qualified_classes=[1], omega={'1': 1.})
        truth = {(j, s): [dict(cls=1, J00=0., J10=float(vec[j]))] for s in panel for j in range(J)}
        data = dict(fr6=fr, M=np.ones((J, 1)), solves={s: {'arrays': arr} for s in panel},
                    vbar={s: np.ones((1, 1)) for s in panel}, v6_truth=truth)
        ai = types.SimpleNamespace(matched_dz=lambda dz, *_: dz, softmax_jvp=lambda p, dz: dz)
        for group in a.GROUPS:
            d = dest / group
            d.mkdir()
            for seed in panel:
                np.savez(d / f'seed{seed}.npz', score=vec, truth=vec, group=group,
                         n_removed=0 if group == 'none' else 50, stage1_seed=seed)
                if group == 'none':
                    (d / f'seed{seed}_replay.json').write_text(json.dumps({'score_max_abs': 0.0}))
        old = a.load_interface
        try:
            a.load_interface = lambda *args: (cfg, data, ai)
            a.collect('unused', 'unused', dest, 50)
            with (dest / 'results.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 6 and all(abs(float(r['lds']) - 1.0) < 1e-12 for r in rows)
            assert int(rows[1]['removed_shadow3_member_rows']) == 0
            assert all(int(r['removed_shadow3_member_rows']) == 50 for r in rows[2:])
            try:
                a.collect('unused', 'unused', dest, 200)
            except ValueError as e:
                assert 'Mismatched output' in str(e)
            else:
                raise AssertionError('An incorrect n must not be reported as the actual removal count')
        finally:
            a.load_interface = old
