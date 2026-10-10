"""CPU unit tests: targets, design, score identities, Slurm wiring, prepare, and a
synthetic end-to-end collect with a known additive truth."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt  # noqa: E402
import qtargets as Q  # noqa: E402
import imgdesign as ID  # noqa: E402
from analysis_stats import spearman  # noqa: E402


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class TargetTests(unittest.TestCase):
    def test_logit_derivatives_match_finite_differences(self):
        rng = np.random.default_rng(1)
        z, y = rng.normal(size=(9, 4)) * 2, rng.integers(0, 4, 9)
        d = rng.normal(size=z.shape)
        for t in Q.TARGETS:
            u = Q.logit_derivative(t, log_softmax(z), y)
            eps = 1e-6
            fd = (Q.per_image_loss(t, log_softmax(z + eps * d), y) - Q.per_image_loss(t, log_softmax(z - eps * d), y)) / (2 * eps)
            np.testing.assert_allclose(fd, (u * d).sum(1), rtol=1e-6, atol=1e-9, err_msg=t)

    def test_extreme_logits_stay_finite(self):
        z = np.array([[400., -400., 0., 0.]])
        lp = log_softmax(z)
        for t in Q.TARGETS:
            self.assertTrue(np.isfinite(Q.per_image_loss(t, lp, np.array([1]))).all())
            self.assertTrue(np.isfinite(Q.logit_derivative(t, lp, np.array([1]))).all())
        self.assertAlmostEqual(float(Q.per_image_loss('ce', lp, np.array([1]))[0]), 800., places=6)

    def test_values_weightings(self):
        rng = np.random.default_rng(2)
        lp = log_softmax(rng.normal(size=(10, 4)))
        y = rng.integers(0, 4, 10)
        units = [np.arange(0, 4), np.arange(4, 10)]
        w = np.r_[np.full(4, .5 / 4), np.full(6, .5 / 6)]
        v = Q.values(lp, y, w, units)
        for t in Q.TARGETS:
            loss = Q.per_image_loss(t, lp, y)
            self.assertAlmostEqual(v[t]['Y_img'], loss.mean())
            self.assertAlmostEqual(v[t]['Y_macro'], .5 * loss[:4].mean() + .5 * loss[4:].mean())


class DesignTests(unittest.TestCase):
    def test_fixed_size_distinct_and_deterministic(self):
        d = ID.image_design(2045, 32, .5, 20261009)
        M = np.asarray(d['masks'])
        self.assertEqual(d['retained_images'], 1022)
        self.assertTrue(np.all(M.sum(1) == 1022))
        self.assertEqual(len({m.tobytes() for m in M}), 32)
        self.assertEqual(ID.image_design(2045, 32, .5, 20261009)['masks'], d['masks'])
        self.assertLess(abs(M.mean() - 1022 / 2045), 1e-12)
        self.assertLess(np.abs(M.mean(0) - .5).max(), .4)  # every image appears in some subsets

    def test_identities(self):
        rng = np.random.default_rng(3)
        M = np.asarray(ID.image_design(40, 10, .5, 1)['masks'], float)
        tau, truth = rng.normal(size=40), rng.normal(size=10)
        dele = ID.deleter_prediction(M, tau)
        np.testing.assert_allclose(dele, M @ tau - tau.sum(), atol=1e-12)
        self.assertAlmostEqual(spearman(dele, truth), spearman(ID.matched_prediction(M, tau), truth))
        self.assertAlmostEqual(spearman(-dele, truth), -spearman(dele, truth))
        groups = np.repeat([7, 8, 9, 10], 10)
        A = ID.aggregation(groups, [7, 8, 9, 10])
        np.testing.assert_allclose(A @ tau, [tau[i * 10:(i + 1) * 10].sum() for i in range(4)])

    def test_features(self):
        M = np.asarray(ID.image_design(20, 8, .5, 2)['masks'])
        f = ID.features(M, np.arange(20) % 4, np.repeat(np.arange(5), 4))
        self.assertTrue(np.all(f['n_images'] == 10))
        self.assertTrue(np.all(f['patients_fully_removed'] + f['patients_with_any_retained_image'] == 5))


class ControllerTests(unittest.TestCase):
    def config(self, tmp):
        import run
        c = dict(out=tmp, account='acct', seeds=list(range(42, 50)), concurrency=8, design_name='img32',
                 design=dict(kind='image_random', m=32, retain=.5, seed=1), score_seeds=[42, 43, 44],
                 score_targets=['ce', 'oneminusp', 'brier'], train_partition='gpuA40x4', fp64_partition='gpuA100x4',
                 score_time='04:00:00')
        c['cells'] = run.make_cells(c)
        c['packs'] = run.make_packs(c['cells'], 20.)
        c['proxy_tasks'], c['score_tasks'] = c['seeds'], c['score_seeds']
        return c

    def test_cells_and_packs(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.config(tmp)
            self.assertEqual(len(c['cells']), 8 + 256)
            self.assertTrue(all(x['source'] == 'harvest' for x in c['cells'] if x['kind'] == 'full'))
            cells = {x['name']: x for x in c['cells']}
            for packs in c['packs'].values():
                for p in packs:
                    self.assertLessEqual(len(p), 8)
                    self.assertLessEqual(sum(cells[n]['minutes'] for n in p), 20.)
            order = [cells[n]['seed'] for p in c['packs']['img32'] for n in p]
            self.assertEqual(order, sorted(order))

    def test_submit_partitions_and_dependencies(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            c = self.config(tmp)
            calls = []
            def fake(cmd, **kw):
                calls.append(cmd)
                return str(700 + len(calls)) + '\n'
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output', side_effect=fake):
                run.submit(SimpleNamespace(out=tmp, defer_score=False, score_time=None, fp64_partition=None))
                by = {cmd[-1]: cmd for cmd in calls}
                self.assertIn('--partition=gpuA40x4', by['smoke'])
                self.assertIn('--partition=gpuA40x4', by['pack:img32'])
                self.assertIn('--partition=gpuA100x4', by['proxy'])
                self.assertIn('--partition=gpuA100x4', by['score'])
                self.assertIn('--dependency=afterok:701', by['pack:img32'])
                self.assertIn('--dependency=afterany:702', by['proxy'])
                self.assertIn('--time=04:00:00', by['score'])
                with self.assertRaises(FileExistsError):
                    run.submit(SimpleNamespace(out=tmp, defer_score=False, score_time=None, fp64_partition=None))
            rt.write(Path(tmp) / 'config.json', {})
            self.assertEqual(run.incomplete(c)['score'], [0, 1, 2])

    def test_prepare_assembles_frozen_config(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'exp'
            base = dict(out=str(out), input_sha256={}, source_sha256={}, full=tmp, data_dir=tmp, account='acct',
                        dataset_fingerprint='x')
            cfg = dict(baseline_out='/b', lds_pilot_out='/p')
            with patch.object(sys, 'argv', ['run.py', 'prepare', '--out', str(out)]), \
                    patch.object(run.rt, 'prepare_base', return_value=base), \
                    patch.object(run, 'check_seed32', return_value=cfg), \
                    patch.object(run.rt, 'save_config') as save:
                run.main()
            c = save.call_args[0][0]
            self.assertEqual(c['seeds'], list(range(42, 50)))
            self.assertEqual(c['design'], dict(kind='image_random', m=32, retain=.5, seed=20261009))
            self.assertEqual(c['score_tasks'], [42, 43, 44])
            self.assertEqual((c['train_partition'], c['fp64_partition']), ('gpuA40x4', 'gpuA100x4'))
            self.assertEqual(len(c['cells']), 264)
            for rel in rt.REUSED:
                self.assertIn(str(rt.REPO / rel), c['source_sha256'])


class SyntheticCollect(unittest.TestCase):
    """Additive world: subset effect = sum over deleted images of -tau_i (+ seed noise).
    Proxy/IF equal to tau must give LDS near 1; per-query predictions likewise."""

    def test_missing_patient_gate_withholds_if(self):
        import analysis3
        data = dict(checks={}, scores={(42, 'ce'): dict(
            result=dict(gammas=[dict(gamma=1., qualified=True)]),
            arrays={'tau_gamma1': np.ones(4)})})
        tau, why = analysis3.if_scores(data, dict(score_seeds=[42]), 'ce', 1.)
        self.assertIsNone(tau)
        self.assertIn('gate missing', why)

    def test_collect(self):
        import run
        import targets as T32
        from design import make_design
        rng = np.random.default_rng(5)
        ntr, nq = 60, 12
        g = np.r_[np.repeat(np.arange(12), 5), 100 + np.arange(nq)]
        y = np.r_[np.repeat(np.arange(12) % 4, 5), np.arange(nq) % 4]
        train, test = np.arange(ntr), np.arange(ntr, ntr + nq)
        base = make_design(train, test, g, y, 8, .5, 20260929)
        design = ID.image_design(ntr, 16, .5, 4)
        M = np.asarray(design['masks'], float)
        _, units, _ = T32.query_units(base)
        tau = rng.normal(size=ntr) * 1e-2
        aq = rng.uniform(.5, 1.5, nq)
        seeds = [42, 43, 44, 45]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'exp'; root.mkdir()
            c = dict(out=str(root), seeds=seeds, design_name='img16', design=dict(kind='image_random', m=16, retain=.5, seed=4),
                     score_seeds=[42, 43], score_targets=['ce', 'oneminusp'], gammas=[1., .1])
            c['cells'] = run.make_cells(c)
            c['proxy_tasks'], c['score_tasks'] = seeds, c['score_seeds']
            rt.write(root / 'config.json', c)
            frozen = dict(base=base, design=design)
            (root / 'smoke').mkdir()
            rt.write(root / 'smoke/designs.json', frozen); rt.write(root / 'smoke/result.json', {})
            rt.finish(root / 'smoke', c, dict(kind='smoke'))
            rt.write(root / 'designs.json', frozen)

            # Partial collection is valid immediately after smoke, before any cell exists.
            with patch.object(run, 'config', return_value=c):
                run.collect(SimpleNamespace(out=str(root), allow_partial=True))
            partial = next(root.glob('report_partial_*/summary.json'))
            self.assertEqual(rt.read(partial)['design']['n_query'], nq)

            def logs(delta):
                z = np.zeros((ntr + nq, 4))
                z[np.arange(ntr + nq), y] = 1.
                z[ntr + np.arange(nq), y[test]] = 1. - delta * aq   # logit of the true class
                return log_softmax(z)
            for cell in c['cells']:
                d = root / 'cells' / cell['name']; d.mkdir(parents=True)
                delta = 1e-4 * rng.normal() + (0. if cell['kind'] == 'full' else float(ID.deleter_prediction(M[[cell['subset']]], tau)[0]))
                Lg = logs(delta)
                np.savez_compressed(d / 'interface.npz', raw=np.exp(Lg), ema=np.exp(Lg), log_raw=Lg, log_ema=Lg,
                                    raw_index=np.r_[train, test], classes=y)
                v = Q.values(Lg[ntr:], y[test], np.asarray(base['query_weights']), units)
                rt.write(d / 'result.json', dict(cell=cell, n_train_images=ntr if cell['kind'] == 'full' else int(M[cell['subset']].sum()),
                                                Y={ep: {t: dict(img=v[t]['Y_img'], macro=v[t]['Y_macro']) for t in Q.TARGETS}
                                                   for ep in ('raw', 'ema')}))
                rt.finish(d, c, cell)
            # delta lowers the true-class logit, so every loss rises with delta = deleter(tau); scores = tau
            for s in seeds:
                d = root / 'proxy' / f'seed{s}'; d.mkdir(parents=True)
                arr = {}
                for t in Q.TARGETS:
                    for w in ('img', 'macro'):
                        arr[f'tau_{w}_{t}'] = tau + 1e-4 * rng.normal(size=ntr)
                    Tq = aq[:, None] * tau[None, :]
                    arr[f'pred_qimage_{t}'] = -(1. - M) @ Tq.T
                    arr[f'pred_qpatient_{t}'] = -(1. - M) @ np.stack([Tq[u].mean(0) for u in units]).T
                np.savez_compressed(d / 'proxy.npz', **arr)
                rt.write(d / 'result.json', {}); rt.finish(d, c, dict(kind='proxy', seed=s))
            for s in c['score_seeds']:
                for t in c['score_targets']:
                    d = root / 'score' / f'seed{s}_{t}'; d.mkdir(parents=True)
                    np.savez_compressed(d / 'scores.npz', **{'tau_gamma1': tau}, proxy=tau)
                    gam = [dict(gamma=1., qualified=True, reasons=[], true_rel_residual=1e-9, spearman_if_vs_proxy_images=1.),
                           dict(gamma=.1, qualified=False, reasons=['non-positive curvature'], true_rel_residual=None,
                                spearman_if_vs_proxy_images=None)]
                    rt.write(d / 'result.json', dict(gammas=gam, lanczos=dict(steps=9)))
                    rt.finish(d, c, dict(kind='score', seed=s, target=t))
                if s in (42, 43, 44):
                    d = root / 'score' / f'seed{s}_check'; d.mkdir(parents=True)
                    np.savez_compressed(d / 'scores.npz', patient_tau_gamma1=np.zeros(3))
                    rt.write(d / 'result.json', dict(gate=dict(passed=True, rel_l2_difference=1e-9, spearman=1., true_rel_residual=1e-9)))
                    rt.finish(d, c, dict(kind='score', seed=s, target='check'))
            with patch.object(run, 'config', return_value=c):
                run.collect(SimpleNamespace(out=str(root), allow_partial=False))
            s = rt.read(root / 'report/summary.json')
            blk = s['macro']['raw']['ce_img']['predictors']
            self.assertGreater(blk['proxy_deleter']['spearman'], .9)
            self.assertGreater(blk['proxy_deleter_score_seeds']['spearman'], .9)
            self.assertGreater(blk['IF_gamma1_deleter']['spearman'], .9)
            self.assertGreater(s['macro']['raw']['brier_macro']['predictors']['proxy_deleter']['spearman'], .9)
            eq = s['equivalence']['ce']
            self.assertAlmostEqual(eq['deleter'], eq['retained_sum']); self.assertAlmostEqual(eq['sign_flip'], -eq['deleter'])
            self.assertTrue(s['gamma']['ce']['0.1']['status'].startswith('unqualified'))
            self.assertIsNone(s['gamma']['ce']['0.1']['mean_spearman_if_vs_proxy'])
            self.assertGreater(s['per_query']['ce_image']['proxy']['mean'], .9)
            self.assertGreater(s['per_query']['oneminusp_patient']['proxy']['mean'], .9)
            self.assertEqual(s['design']['n_images_range'][0], s['design']['n_images_range'][1])
            self.assertTrue((root / 'report' / 'fig1_truth_and_lds_vs_k.png').exists())
            self.assertTrue(Path(str(root) + '.report.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
