"""CPU unit tests: the shared Galerkin solver against dense solves and multi-shift Lanczos, the per-output
LDS statistics (including the exact permutation reduction), the matched-normalisation and softmax algebra,
the panel rule, prepare pins, repository rules and job dependencies."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime as rt  # noqa: E402
import subspace  # noqa: E402
import analysis_if as AI  # noqa: E402
import core_if  # noqa: E402
from analysis_stats import spearman  # noqa: E402


def own_run(name):
    spec = importlib.util.spec_from_file_location(name, HERE / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


def spectrum_matrix(D, rng, negative=0.):
    lam = np.r_[np.exp(rng.uniform(np.log(.05), np.log(5), 25)), np.exp(rng.uniform(np.log(1e-5), np.log(1e-3), D - 26)), [negative]]
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    return (U * lam) @ U.T, U


class SolverTests(unittest.TestCase):
    def test_matches_dense_and_reuses_directions(self):
        import torch
        rng = np.random.default_rng(0)
        D = 300
        H, U = spectrum_matrix(D, rng, negative=-1e-3)
        Ht = torch.tensor(H)
        G = torch.tensor(rng.normal(size=(60, 25)) @ U[:, :25].T + .02 * rng.normal(size=(60, D)))
        masks = rng.random((12, 60)) < .5
        gam, tols = [1., .1, .03, .003], [1e-10, 1e-10, 1e-10, 1e-8]
        sg = subspace.SharedGalerkin(lambda v: Ht @ v, D, D)
        news = []
        for j in range(12):
            b = torch.tensor(masks[j].astype(float)) @ G
            out = sg.solve(b, gam, tols, 300)
            news.append(out['new_vectors'])
            self.assertTrue(out['converged'])
            for g in gam:
                x = np.linalg.solve(H + g * np.eye(D), b.numpy())
                self.assertLess(np.linalg.norm(out['x'][g].numpy() - x) / np.linalg.norm(x), 1e-5)
                r = np.linalg.norm(H @ out['x'][g].numpy() + g * out['x'][g].numpy() - b.numpy()) / np.linalg.norm(b.numpy())
                self.assertAlmostEqual(r, out['rel_residual'][g], delta=1e-10)   # stored HQ residual is exact
        self.assertLess(np.mean(news[3:]), news[0] / 2)                          # later right-hand sides reuse the basis
        self.assertAlmostEqual(sg.ritz_min(), -1e-3, places=6)

    def test_first_rhs_is_multishift_lanczos(self):
        import torch
        sys.path.insert(0, str(rt.SEED32))
        import solver
        rng = np.random.default_rng(1)
        H, _ = spectrum_matrix(200, rng)
        Ht = torch.tensor(H)
        b = torch.tensor(rng.normal(size=200))
        gam = [1., .1, .03]
        sg = subspace.SharedGalerkin(lambda v: Ht @ v, 200, 200)
        out = sg.solve(b, gam, [1e-6] * 3, 200)
        ms = solver.multishift_lanczos(lambda v: Ht @ v, b, gam, 200, 1e-6, dot=lambda a, c: float(a @ c))
        self.assertEqual(out['new_vectors'], ms['steps'])
        for g in gam:
            np.testing.assert_allclose(out['x'][g].numpy(), ms['x'][g].numpy(), rtol=1e-6, atol=1e-9)

    def test_cap_and_stall_are_reported(self):
        import torch
        rng = np.random.default_rng(2)
        H, _ = spectrum_matrix(100, rng)
        Ht = torch.tensor(H)
        sg = subspace.SharedGalerkin(lambda v: Ht @ v, 100, 5)
        out = sg.solve(torch.tensor(rng.normal(size=100)), [.003], [1e-12], 50)
        self.assertTrue(out['capped'])
        self.assertFalse(out['converged'])
        sg2 = subspace.SharedGalerkin(lambda v: 2 * v, 10, 10)                       # H = 2I: one direction suffices
        out = sg2.solve(torch.ones(10, dtype=torch.float64), [1.], [1e-12], 5)
        self.assertTrue(out['converged'])
        self.assertEqual(out['new_vectors'], 1)


class StatisticsTests(unittest.TestCase):
    def test_per_output_lds_equals_loop(self):
        rng = np.random.default_rng(3)
        P, T = rng.normal(size=(32, 50)), rng.normal(size=(32, 50))
        T[:, :10] = P[:, :10] ** 3
        loop = [spearman(P[:, k], T[:, k]) for k in range(50)]
        np.testing.assert_allclose(AI.per_output_lds(P, T), loop, atol=1e-12)

    def test_permutation_reduction_is_exact(self):
        rng = np.random.default_rng(4)
        P, T = rng.normal(size=(16, 40)), rng.normal(size=(16, 40))
        A, B = AI._centered_unit_ranks(P), AI._centered_unit_ranks(T)
        C = A @ B.T / 40
        for _ in range(5):
            pi = rng.permutation(16)
            direct = np.mean(AI.per_output_lds(P, T[pi]))
            self.assertAlmostEqual(C[np.arange(16), pi].sum(), direct, places=12)   # T rows permuted
        r = AI.mean_lds_inference(P, P ** 3, perms=200, boots=200)
        self.assertAlmostEqual(r['mean'], 1.)
        self.assertLess(r['perm_p_one_sided'], .01)
        null = AI.mean_lds_inference(P, T, perms=500, boots=100)
        self.assertGreater(null['perm_p_one_sided'], .01)

    def test_split_half_ceiling(self):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=(32, 30))
        Ts = signal[None] + .5 * rng.normal(size=(8, 32, 30))
        out = AI.split_half(Ts, np.arange(30))
        self.assertGreater(out['spearman_brown'], out['half'])
        self.assertLess(out['ceiling'], 1.)


class AlgebraTests(unittest.TestCase):
    def test_matched_normalisation(self):
        """matched dz == J_z (A)^{-1} [ -sum_i ((N/K_j) M_ji - 1) g_i ] for a dense linear model."""
        rng = np.random.default_rng(6)
        N, D, R = 20, 15, 7
        A = np.eye(D) + .1 * rng.normal(size=(D, D))
        A = A @ A.T
        G = rng.normal(size=(N, D))
        Jz = rng.normal(size=(R * 4, D))
        M = (rng.random((5, N)) < .5).astype(float)
        M[:, 0] = 1
        K = M.sum(1)
        dz_del = np.stack([(Jz @ np.linalg.solve(A, (1 - M[j]) @ G)).reshape(R, 4) for j in range(5)])
        dz_all = (Jz @ np.linalg.solve(A, G.sum(0))).reshape(R, 4)
        got = AI.matched_dz(dz_del, dz_all, K, N)
        for j in range(5):
            want = (Jz @ np.linalg.solve(A, -(((N / K[j]) * M[j] - 1) @ G))).reshape(R, 4)
            np.testing.assert_allclose(got[j], want, atol=1e-10)

    def test_softmax_jvp_and_step(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=(5, 4))
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        dz = rng.normal(size=(5, 4))
        eps = 1e-6
        sm = lambda v: np.exp(v) / np.exp(v).sum(1, keepdims=True)
        fd = (sm(z + eps * dz) - sm(z - eps * dz)) / (2 * eps)
        np.testing.assert_allclose(AI.softmax_jvp(p, dz), fd, atol=1e-8)
        np.testing.assert_allclose(core_if.softmax_jvp(p, dz), fd, atol=1e-8)
        np.testing.assert_allclose(AI.softmax_step(p, dz), sm(z + dz) - p, atol=1e-12)


class PanelTests(unittest.TestCase):
    def test_panel_rule(self):
        seeds = list(range(42, 50))
        self.assertEqual(core_if.choose_panel(seeds, 2., 24., 12.)[0], seeds)
        self.assertEqual(core_if.choose_panel(seeds, 6., 24., 12.)[0], seeds[:3])
        self.assertIsNone(core_if.choose_panel(seeds, 9., 24., 12.)[0])
        self.assertIsNone(core_if.choose_panel(seeds, 13., 240., 12.)[0])       # walltime
        h, hvps = core_if.projected_hours(20., 60, [10, 12, 8], 33, 24, 300., 4., 262)
        self.assertAlmostEqual(hvps, 60 + 33 * 10 + 24 + 12)
        self.assertAlmostEqual(h, (hvps * 20 + 300 + 262 * 4) / 3600)


class PrepareTests(unittest.TestCase):
    def build_v6(self, tmp, commit=None):
        v6 = tmp / 'v6'
        cells = [dict(name=f'truth_s{j:03d}_seed{s}', kind='truth', subset=j, seed=s, minutes=1.) for s in (42, 43) for j in range(2)]
        cfg = dict(out=str(v6), schema='OCT_cross_stage_lds_v6', git_commit=commit or rt.V6_COMMIT, full='/full', data_dir='d',
                   dataset_fingerprint='fp', v3_out='/v3', v3_config_sha256='x', v3_cells=[], seed32_out='/s', baseline_out='/b',
                   lds_pilot_out='/p', seeds=[42, 43], epochs=140, recipe={}, design_name='img32', design=dict(m=2),
                   v3_prediction_tol=1e-10, harvest_tol=1e-10, attack=dict(epochs=50), classes=[0, 1, 2, 3], n_shadow=5,
                   primary_classes=[1, 2], attack_gammas=[1., .3], attack_seeds=[5101, 5104], gammas=[1., .3, .03, .003],
                   cg_fail_tol=1e-3, probe_steps=24, batch=64, weight_decay=1e-5, cells=cells, adjoint_identity_rel_tol=1e-9,
                   score_seeds=[42, 43], input_sha256={})
        v6.mkdir()
        rt.write(v6 / 'config.json', cfg)
        (v6 / 'config.sha256').write_text(rt.sha(v6 / 'config.json') + '\n')

        def done(d, cell, files):
            d.mkdir(parents=True)
            for f in files:
                (d / f).write_bytes(f.encode() + str(d).encode())
            if 'frozen.json' in files:
                (d / 'frozen.json').unlink()
                rt.write(d / 'frozen.json', dict(attack_seeds=[5101, 5104], attack_gamma_star=.3, omega={'1': .6, '2': .4},
                                                 qualified_classes=[1, 2]))
            rt.finish(d, cfg, cell)
        done(v6 / 'smoke', dict(kind='xsmoke'), ['result.json', 'designs.json', 'frozen.json', 'fixed_rows.npz'])
        for s in cfg['seeds']:
            done(v6 / 'base' / f'seed{s}', dict(kind='base6', seed=s), ['result.json', 'v.npz', 'h.npz'])
            done(v6 / 'score' / f'seed{s}', dict(kind='score6', seed=s), ['result.json', 'scores.npz'])
        for cell in cells:
            done(v6 / 'truth' / cell['name'], dict(cell, kind='truth6'), ['result.json'])
        (v6 / 'report').mkdir()
        rt.write(v6 / 'report/complete.json', dict(status='complete', config_sha256=rt.sha(v6 / 'config.json'), missing=[]))
        return v6

    def args(self, out, v6, **kw):
        a = dict(out=str(out), v6_out=str(v6), audit='/a', gammas='', primary_gamma=.03, tol_main=1e-6, tol_small=1e-4, max_dim=1400,
                 max_new=200, pilot_rhs=4, budget_gpu_hours=24., solve_hours=12., partition='gpuA100x4',
                 cpu_account='cpu-acct', cpu_partition='cpu', concurrency=8)
        a.update(kw)
        return SimpleNamespace(**a)

    def test_prepare_pins_v6(self):
        run = own_run('run_prep_if')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v6 = self.build_v6(tmp)
            base = lambda a, schema: dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()),
                                          full='/full', data_dir='d', dataset_fingerprint='fp', account='acct',
                                          input_sha256={}, source_sha256={})
            with patch.object(rt, 'prepare_base', side_effect=base):
                run.prepare(self.args(tmp / 'out', v6))
                c = rt.read(tmp / 'out/config.json')
                self.assertEqual(c['schema'], 'OCT_interface_lds_v1')
                self.assertEqual(c['v6_config_sha256'], rt.sha(v6 / 'config.json'))
                for rel in ('base/seed43/v.npz', 'score/seed42/scores.npz', 'truth/truth_s001_seed43/result.json', 'smoke/frozen.json'):
                    self.assertIn(str(v6 / rel), c['input_sha256'])
                self.assertNotIn(str(v6 / 'base/seed42/h.npz'), c['input_sha256'])
                self.assertEqual(c['attack_seeds'], [5101, 5104])
                with self.assertRaisesRegex(ValueError, 'Primary gamma'):
                    run.prepare(self.args(tmp / 'out2', v6, primary_gamma=.05))
                with self.assertRaisesRegex(ValueError, 'Primary gamma'):
                    run.prepare(self.args(tmp / 'out2b', v6, gammas='1,0.3'))
                run.prepare(self.args(tmp / 'out2c', v6, gammas='0.03,1'))
                self.assertEqual(rt.read(tmp / 'out2c/config.json')['gammas'], [1., .03])
                (v6 / 'score/seed43/scores.npz').write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    run.prepare(self.args(tmp / 'out3', v6))
            (tmp / 'other').mkdir()
            v6b = self.build_v6(tmp / 'other', commit='0' * 40)
            with patch.object(rt, 'prepare_base', side_effect=base):
                with self.assertRaisesRegex(ValueError, 'v6 r1 run'):
                    run.prepare(self.args(tmp / 'out4', v6b))

    def test_mismatched_inputs_rejected_before_copy(self):
        run = own_run('run_prep_if2')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v6 = self.build_v6(tmp)
            for key in ('full', 'data_dir', 'dataset_fingerprint'):
                def base(a, schema, key=key):
                    c = dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()), full='/full',
                             data_dir='d', dataset_fingerprint='fp', account='acct', input_sha256={}, source_sha256={})
                    c[key] = 'different'
                    return c
                with self.subTest(key=key), patch.object(rt, 'prepare_base', side_effect=base):
                    with self.assertRaisesRegex(ValueError, key + ' differs'):
                        run.prepare(self.args(tmp / ('bad_' + key), v6))


class RepoAndSubmitTests(unittest.TestCase):
    def test_check_repo_rules(self):
        ours = 'A\texperiments/pathway2_interface_lds_v1/core_if.py'
        for row, diffs, ok in ((ours, ('', '', ''), True),
                               ('A\texperiments/pathway2_cross_stage_lds_v6/core6.py', ('', '', ''), True),
                               ('M\texperiments/pathway2_cross_stage_lds_v6/core6.py', ('', '', ''), False),
                               ('A\tscripts/cross_stage/new.py', ('', '', ''), False),
                               (ours, ('', '', 'experiments/pathway2_cross_stage_lds_v6/core6.py'), False)):
            with self.subTest(row=row, diffs=diffs):
                calls = iter(['', row, *diffs, 'head'])
                with patch.object(rt, 'git', side_effect=lambda *a: next(calls)), patch.object(rt.subprocess, 'run'):
                    if ok:
                        self.assertEqual(rt.check_repo(), 'head')
                    else:
                        with self.assertRaises(ValueError):
                            rt.check_repo()

    def test_submit_dependencies(self):
        run = own_run('run_submit_if')
        with tempfile.TemporaryDirectory() as tmp:
            c = dict(out=tmp, account='gpu-acct', cpu_account='cpu-acct', cpu_partition='cpu', partition='gpuA100x4',
                     seeds=list(range(42, 50)), concurrency=8, solve_hours=12.)
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output',
                  side_effect=['101', '102', '103']) as submit:
                run.submit(SimpleNamespace(out=tmp))
            jobs = rt.read(Path(tmp) / 'jobs.json')
            calls = {call.args[0][-1]: call.args[0] for call in submit.call_args_list}
            self.assertIn('--array=0-7%8', calls['solve'])
            self.assertIn('--time=12:00:00', calls['solve'])
            self.assertIn('--dependency=afterok:' + jobs['smoke'], calls['solve'])
            self.assertIn('--dependency=afterok:' + jobs['solve'], calls['collect'])
            self.assertIn('--account=cpu-acct', calls['collect'])
            self.assertTrue(all(any(x.startswith('--job-name=oct_if_') for x in v) for v in calls.values()))


if __name__ == '__main__':
    unittest.main()
