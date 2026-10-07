"""CPU tests: formulas, designs, statistics, Slurm wiring and a synthetic
end-to-end collect. No claim about OCT/GPU numerics (that is the smoke's job)."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt  # noqa: E402  (adds LDS v1 to sys.path)
import targets as T  # noqa: E402
import designs as DS  # noqa: E402
import stats2 as S  # noqa: E402
import design as v1  # noqa: E402  LDS v1
from analysis_stats import spearman  # noqa: E402


def synthetic_base(n_train=16, n_query=12, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 9, n_train)
    qsizes = rng.integers(1, 5, n_query)
    groups = np.concatenate([np.repeat(np.arange(n_train), sizes), np.repeat(1000 + np.arange(n_query), qsizes)])
    labels = np.concatenate([np.repeat(np.arange(n_train) % 4, sizes), np.repeat(np.arange(n_query) % 4, qsizes)])
    train = np.arange(sizes.sum())
    test = np.arange(sizes.sum(), len(groups))
    return v1.make_design(train, test, groups, labels, 8, .5, 20260929), labels


class TargetTests(unittest.TestCase):
    def test_derivatives(self):
        from formula_tests import target_fd_checks
        self.assertTrue(all(v < 1e-6 for v in target_fd_checks().values()))

    def test_ce_matches_lds_v1(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(4), 9)
        y, w = rng.integers(0, 4, 9), rng.random(9)
        a, va = T.value_and_v('ce', p, y, w)
        b, vb = v1.output_and_v(p, y, w)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(va, vb)

    def test_query_units_recombine_macro(self):
        base, labels = synthetic_base()
        test = np.asarray(base['query_rows'])
        ids, units, coef = T.query_units(base)
        p = np.random.default_rng(2).dirichlet(np.ones(4), len(test))
        for t in T.TARGETS:
            Y, _ = T.value_and_v(t, p, labels[test], np.asarray(base['query_weights']))
            Yq = T.per_query_values(t, p, labels[test], units)
            self.assertAlmostEqual(float(coef @ Yq), Y, places=12)


class SolverTests(unittest.TestCase):
    def test_multishift_equals_dense_and_cg(self):
        from formula_tests import torch_solver_checks
        out = torch_solver_checks()
        self.assertLess(out['equals_cg_after_6_steps_rel_error'], 1e-10)

    def test_indefinite_gamma_flagged(self):
        import solver
        A = np.diag(np.r_[-.05, np.linspace(.1, 4, 30)])
        b = np.ones(31)
        ms = solver.multishift_lanczos(lambda v: A @ v, b, [1., .03], 31, 1e-12)
        self.assertFalse(solver.qualify(.03, 1e-9, ms['ritz_min'], None, 1e-3)['qualified'])
        self.assertTrue(solver.qualify(1., 1e-9, ms['ritz_min'], None, 1e-3)['qualified'])


class DeleterTests(unittest.TestCase):
    def test_deleter_equivalence_and_sign(self):
        rng = np.random.default_rng(3)
        masks = rng.integers(0, 2, (8, 20))
        tau, truth = rng.normal(size=20), rng.normal(size=8)
        dele = S.deleter_prediction(masks, tau)
        np.testing.assert_allclose(dele, S.retained_sum(masks, tau) - tau.sum(), atol=1e-12)
        self.assertAlmostEqual(spearman(dele, truth), spearman(S.retained_sum(masks, tau), truth), places=12)
        self.assertAlmostEqual(spearman(-dele, truth), -spearman(dele, truth), places=12)

    def test_matched_equals_lds_v1_intervention(self):
        base, _ = synthetic_base()
        tau = np.random.default_rng(4).normal(size=len(base['train_patients']))
        n = np.asarray([len(p['rows']) for p in base['train_patients']])
        mine = S.matched_prediction(base['masks'], tau, n)
        ref = [v1.intervention(base, j, 'subset', 1., n.sum())[1] @ tau for j in range(8)]
        np.testing.assert_allclose(mine, ref, rtol=1e-12)


class DesignTests(unittest.TestCase):
    def patients(self):
        rng = np.random.default_rng(9)
        sizes = np.maximum(1, rng.lognormal(2.2, .8, 148).astype(int))
        start = np.r_[0, np.cumsum(sizes)]
        return [dict(patient=i, oct_class=i % 4, rows=list(range(start[i], start[i + 1]))) for i in range(148)]

    def test_equal_size_stratified(self):
        P = self.patients()
        d = DS.equal_size_stratified(P, 32, .5, 20261006, .02, .10, diagnostic_pool=200)
        self.assertEqual(len({tuple(m) for m in d['masks']}), 32)
        M = np.asarray(d['masks'])
        cls = np.asarray([p['oct_class'] for p in P])
        imgs = np.asarray([len(p['rows']) for p in P])
        for c in range(4):
            self.assertTrue(np.all(M[:, cls == c].sum(1) == d['stratified_counts'][str(c)]))
            self.assertTrue(np.all(np.abs(M[:, cls == c] @ imgs[cls == c] - d['target_class_images'][str(c)])
                                   <= d['class_tolerance'][str(c)]))
        self.assertTrue(np.all(np.abs(M @ imgs - d['target_total_images']) <= d['total_tolerance']))
        self.assertEqual(M.sum(1).tolist(), [74] * 32)
        again = DS.equal_size_stratified(P, 32, .5, 20261006, .02, .10, diagnostic_pool=200)
        self.assertEqual(again['masks'], d['masks'])

    def test_infeasible_fails_loudly(self):
        with self.assertRaises(ValueError):
            DS.equal_size_stratified(self.patients(), 4, .5, 1, 0., 0., batch=1000, max_draws=3000)

    def test_small_deletion(self):
        d = DS.small_deletion(self.patients(), 32, 10, 5)
        self.assertTrue(all(sum(m) == 138 for m in d['masks']))


class StatsTests(unittest.TestCase):
    def test_residual_spearman_removes_size(self):
        rng = np.random.default_rng(5)
        n = rng.uniform(700, 1300, 32)
        noise = rng.normal(size=32)
        truth, pred = -n / 100 + noise, -n
        self.assertGreater(spearman(pred, truth), .8)
        self.assertIsNone(S.residual_spearman(pred, truth, n))  # pure size predictor: undefined after control
        mixed = -n + 50 * rng.normal(size=32)  # size-driven predictor with an unrelated extra part
        self.assertGreater(spearman(mixed, truth), .5)
        self.assertLess(abs(S.residual_spearman(mixed, truth, n)), .45)
        self.assertGreater(S.residual_spearman(noise + .01 * rng.normal(size=32), truth, n), .9)

    def test_per_query_vectorized_matches_loop(self):
        rng = np.random.default_rng(6)
        pred, truth = rng.normal(size=(8, 30)), rng.normal(size=(8, 30))
        truth[:, 3] = 1.  # constant -> undefined
        r = S.per_query_lds(pred, truth, draws=50)
        loop = [spearman(pred[:, q], truth[:, q]) for q in range(30)]
        np.testing.assert_allclose([x for x in r['per_query'] if np.isfinite(x)], [x for x in loop if x is not None])
        self.assertEqual(r['defined'], 29)

    def test_seed_curves(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(8, 1))
        Y = signal + .5 * rng.normal(size=(8, 32))
        se = S.se_curve(Y, [1, 2, 4, 8, 16, 32], draws=50)
        self.assertTrue(all(a['analytic_se_mean_over_subsets'] > b['analytic_se_mean_over_subsets'] for a, b in zip(se, se[1:])))
        self.assertAlmostEqual(se[-1]['resampled_sd_of_k_seed_mean'], 0.)
        rel = S.reliability_curve(Y, [1, 2, 4, 8, 16, 32], draws=50)['rows']
        self.assertIsNone(rel[-1].get('split_spearman_mean'))
        self.assertGreater(rel[-1]['icc_reliability'], rel[0]['icc_reliability'])
        curve = S.lds_curve(Y, {'oracle': lambda s: signal[:, 0]}, [1, 32], draws=20)
        self.assertLess(curve[0]['mean'], curve[-1]['mean'] + 1e-12)


class ControllerTests(unittest.TestCase):
    def config(self, tmp):
        import run
        c = dict(out=tmp, account='acct', seeds=list(range(42, 74)), concurrency=8, score_time='04:00:00',
                 designs=dict(pilot8=dict(kind='pilot', m=8, retain=.5), eq32=dict(kind='eq', m=32, retain=.5)),
                 design_seeds=dict(pilot8=list(range(42, 74)), eq32=list(range(42, 50))))
        c['cells'] = run.make_cells(c)
        c['packs'] = run.make_packs(c['cells'], 22.)
        c['score_tasks'] = [dict(seed=s, target=t) for t in ('oneminusp', 'ce') for s in (42, 43, 44)]
        c['proxy_tasks'] = [c['seeds'][i:i + 4] for i in range(0, 32, 4)]
        return c

    def test_cells_and_packs(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.config(tmp)
            cells = {x['name']: x for x in c['cells']}
            self.assertEqual(len(c['cells']), 32 + 256 + 256)
            self.assertEqual(sum(x['source'] == 'harvest' for x in c['cells']), 3 + 24)
            for packs in c['packs'].values():
                for p in packs:
                    self.assertLessEqual(sum(cells[n]['minutes'] for n in p), 22.)
            order = [cells[n]['seed'] for p in c['packs']['pilot8'] for n in p]
            self.assertEqual(order, sorted(order))  # seed-major: whole seeds finish early

    def test_submit_and_resubmit(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            c = self.config(tmp)
            rt.write(Path(tmp) / 'config.json', c)
            calls = []
            def fake(cmd, **kw):
                calls.append(cmd)
                return str(500 + len(calls)) + '\n'
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output', side_effect=fake):
                run.submit(SimpleNamespace(out=tmp))
                self.assertIn('--dependency=afterok:501', calls[1])
                proxy = calls[-1]
                self.assertIn('--dependency=afterany:502', proxy)
                self.assertTrue(any(x.startswith('--array=0-') and x.endswith('%8') for x in calls[1]))
                with self.assertRaises(FileExistsError):
                    run.submit(SimpleNamespace(out=tmp))
                todo = run.incomplete(c)
                self.assertEqual(todo['proxy'], list(range(8)))
                self.assertEqual(run.array([1, 4, 7], 8), '--array=1,4,7%8')


class PrepareTests(unittest.TestCase):
    def test_prepare_assembles_frozen_config(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'exp'
            base = dict(out=str(out), input_sha256={}, source_sha256={}, full=tmp, account='acct')
            args = run.main.__code__  # noqa: F841  (parser lives in main)
            argv = ['run.py', 'prepare', '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch.object(run.rt, 'prepare_base', return_value=base), \
                    patch.object(run, 'check_baseline'), patch.object(run, 'check_pilot'), \
                    patch.object(run.rt, 'verify_hashes'), patch.object(run.rt, 'sha', return_value='x'), \
                    patch.object(run.rt, 'save_config') as save:
                run.main()
            c = save.call_args[0][0]
            self.assertEqual(c['seeds'], list(range(42, 74)))
            self.assertEqual(set(c['designs']), {'pilot8', 'eq32'})
            self.assertEqual(len(c['score_tasks']), 9)
            self.assertEqual(c['score_tasks'][0], dict(seed=42, target='oneminusp'))
            self.assertEqual(len(c['proxy_tasks']), 8)
            self.assertIn(1., c['gammas'])
            with patch.object(sys, 'argv', argv + ['--smalldel']), patch.object(run.rt, 'prepare_base', return_value=base):
                with self.assertRaises(ValueError):
                    run.main()  # smalldel requires a mentor note


class EndToEndTests(unittest.TestCase):
    """Synthetic additive world: true deletion effect = sum_{p not in S}(-tau_p).
    IF/proxy equal to tau must give LDS ~ 1; a size-only truth must vanish after control."""

    def test_collect(self):
        import run
        rng = np.random.default_rng(11)
        base, labels = synthetic_base()
        spec = dict(pilot8=dict(kind='pilot', m=8, retain=.5),
                    eq32=dict(kind='eq', m=32, retain=.5, seed=3, total_tol_frac=.3, class_tol_frac=.6))
        designs = json.loads(json.dumps(DS.build_all(base, spec)))
        P = len(base['train_patients'])
        N = sum(len(p['rows']) for p in base['train_patients'])
        test = np.asarray(base['query_rows'])
        ids, units, coef = T.query_units(base)
        Q = len(units)
        tau = rng.normal(size=P) * 1e-3
        aq = rng.uniform(.5, 1.5, Q)
        seeds = list(range(42, 50))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'exp'
            root.mkdir()
            c = dict(out=str(root), seeds=seeds, design_seeds=dict(pilot8=seeds, eq32=seeds[:4]), designs=spec,
                     score_seeds=[42, 43, 44], score_targets=['oneminusp', 'ce'], gammas=[1., .3, .03])
            c['cells'] = run.make_cells(c)
            rt.write(root / 'config.json', c)
            rt.write(root / 'designs.json', dict(base=base, designs=designs))
            (root / 'smoke').mkdir()
            rt.write(root / 'smoke' / 'designs.json', dict(base=base, designs=designs))
            rt.write(root / 'smoke' / 'result.json', {})
            rt.finish(root / 'smoke', c, dict(kind='smoke'))

            def interface(delta):
                p = np.full((N + len(test), 4), .1)
                py = .6 - delta * aq[np.searchsorted(np.cumsum([len(u) for u in units]), np.arange(len(test)), side='right')]
                rows = np.arange(N, N + len(test))
                p[rows] = ((1 - py) / 3)[:, None]
                p[rows, labels[test]] = py
                return p

            for cell in c['cells']:
                d = root / 'cells' / cell['name']
                d.mkdir(parents=True)
                noise = 1e-5 * rng.normal()
                if cell['kind'] == 'full':
                    delta, n_img = noise, N
                else:
                    mask = np.asarray(designs[cell['design']]['masks'][cell['subset']])
                    delta = float(-(1 - mask) @ tau) + noise
                    n_img = int(mask @ [len(p['rows']) for p in base['train_patients']])
                Pm = interface(delta)
                np.savez_compressed(d / 'interface.npz', raw=Pm, ema=Pm, raw_index=np.arange(len(Pm)),
                                    classes=np.r_[labels[:N], labels[test]])
                vals = T.all_values(Pm[N:], labels[test], np.asarray(base['query_weights']), units)
                rt.write(d / 'result.json', dict(cell=cell, Y={ep: {t: v['Y'] for t, v in vals.items()} for ep in ('raw', 'ema')},
                                                n_train_images=n_img))
                rt.finish(d, c, cell)
            for s in seeds:
                d = root / 'proxy' / f'seed{s}'
                d.mkdir(parents=True)
                arrays = dict(train_patient_ids=np.asarray([p['patient'] for p in base['train_patients']]), query_patient_ids=ids)
                for t in T.TARGETS:
                    arrays[f'proxy_macro_{t}'] = tau + 1e-5 * rng.normal(size=P)
                    arrays[f'proxy_query_{t}'] = aq[:, None] * tau[None, :]
                np.savez_compressed(d / 'proxy.npz', **arrays)
                rt.write(d / 'result.json', {})
                rt.finish(d, c, dict(kind='proxy', seed=s))
            for s in c['score_seeds']:
                for t in c['score_targets']:
                    d = root / 'score' / f'seed{s}_{t}'
                    d.mkdir(parents=True)
                    np.savez_compressed(d / 'scores.npz', **{f'tau_gamma{g:g}': tau for g in c['gammas'] if g != .03},
                                        proxy=tau, train_patient_ids=np.asarray([p['patient'] for p in base['train_patients']]))
                    gam = [dict(gamma=g, qualified=g != .03, reasons=[] if g != .03 else ['non-positive curvature'],
                                true_rel_residual=1e-6, damped_krylov_ritz_min=g - .05,
                                spearman_if_vs_proxy_148_patients=1. - g / 10) for g in c['gammas']]
                    rt.write(d / 'result.json', dict(gammas=gam, lanczos=dict(steps=20)))
                    rt.finish(d, c, dict(kind='score', seed=s, target=t))
            with patch.object(run, 'config', return_value=c):
                run.collect(SimpleNamespace(out=str(root), allow_partial=False))
            s = rt.read(root / 'report' / 'summary.json')
            for design in ('pilot8', 'eq32'):
                blk = s['task2_3'][design]['raw']['oneminusp']
                self.assertGreater(blk['predictors']['IF_gamma1_deleter']['spearman'], .95)
                eq = blk['deleter_equivalence']
                self.assertAlmostEqual(eq['deleter_spearman'], eq['retained_sum_spearman'])
                self.assertAlmostEqual(eq['sign_error_spearman_if_minus_dropped'], -eq['deleter_spearman'])
                self.assertGreater(s['task5'][design]['raw']['oneminusp']['proxy']['mean'], .9)
            self.assertTrue(s['task4']['oneminusp']['0.03']['status'].startswith('unqualified'))
            self.assertEqual(s['task4']['oneminusp']['0.3']['status'], 'solved')
            self.assertIsNone(s['task2_3']['pilot8']['raw']['brier']['deleter_equivalence'])
            for name in ('fig_task1_pilot8_raw.png', 'fig_task3_size.png', 'fig_task4_gamma.png', 'fig_task5_per_query.png',
                         'REPORT_CN.md', 'table_lds.csv'):
                self.assertTrue((root / 'report' / name).exists(), name)
            self.assertTrue(Path(str(root) + '.report.tar.gz').exists())
            self.assertEqual(s['same_seed_design_comparison']['seeds'], seeds[:4])
            self.assertTrue(s['task5']['eq32']['raw']['oneminusp']['proxy']['bootstrap_ci_conditional_on_seed_panel'])
            import tarfile
            with tarfile.open(Path(str(root) + '.report.tar.gz')) as archive:
                self.assertIn('exp/proxy/seed42/proxy.npz', archive.getnames())
                self.assertIn('exp/cells/full_seed42/interface.npz', archive.getnames())


if __name__ == '__main__':
    unittest.main()
