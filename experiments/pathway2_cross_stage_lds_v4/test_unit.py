"""CPU unit tests: the damped attack adjoint against re-optimisation, attack-table assembly,
class weights, pack layout, and the ladder analysis on synthetic results with a known answer."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt  # noqa: E402
import attack4 as A  # noqa: E402
import analysis4  # noqa: E402
from analysis_stats import spearman  # noqa: E402


def pilot():
    import importlib.util
    cross = rt.REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(rt.REPO / 'src')]
    spec = importlib.util.spec_from_file_location('u_score4', cross / '07_cross_stage_score_ladder.py')
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score, score.import_pilot_module()


def simplex(rng, n):
    z = rng.normal(size=(n, 4)) * 1.5
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class AdjointTests(unittest.TestCase):
    def test_damped_adjoint_equals_ridge_reoptimisation(self):
        """With phi* stationary for CE + gamma/2 |phi|^2, v = dJ/dp exactly (Eq. 11-21 with damping as ridge)."""
        import torch
        import torch.nn.functional as F
        _, pl = pilot()
        rng = np.random.default_rng(0)
        P, M = simplex(rng, 40), rng.integers(0, 2, 40)
        Pq, Mq = simplex(rng, 25), rng.integers(0, 2, 25)
        gamma = .3
        dev = pl.DEVICE
        torch.manual_seed(0)
        model = pl.build_attack_model('nn', 4, 64).double()

        def fit(m, Pin):
            x = torch.as_tensor(Pin, device=dev)
            y = torch.as_tensor(M, device=dev)
            opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=500, tolerance_grad=1e-14,
                                    tolerance_change=1e-16, history_size=50, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(m(x), y) + .5 * gamma * sum((p ** 2).sum() for p in m.parameters())
                loss.backward()
                return loss
            for _ in range(100):
                opt.step(closure)
            g = torch.autograd.grad(F.cross_entropy(m(x), y) + .5 * gamma * sum((p ** 2).sum() for p in m.parameters()),
                                    list(m.parameters()))
            return float(torch.cat([t.reshape(-1) for t in g]).norm())

        self.assertLess(fit(model, P), 1e-8)
        adj = A.adjoint(model, P, M, Pq, Mq, [gamma], dev)
        e = adj['per_gamma'][gamma]
        self.assertTrue(e['solved'] and e['damped_eig_min'] > 0)

        def J(Pin):
            m = copy.deepcopy(model)
            self.assertLess(fit(m, Pin), 1e-8)
            with torch.no_grad():
                return float(F.cross_entropy(m(torch.as_tensor(Pq, device=dev)), torch.as_tensor(Mq, device=dev)))
        eps = 1e-3
        for r in (0, 7, 31):
            d = rng.normal(size=4)
            Pp, Pm = P.copy(), P.copy()
            Pp[r] += eps * d
            Pm[r] -= eps * d
            fd = (J(Pp) - J(Pm)) / (2 * eps)
            self.assertAlmostEqual(fd, float(e['v'][r] @ d), delta=1e-4 * max(abs(fd), 1e-2))

    def test_adjoint_matches_legacy_implicit_v(self):
        score, pl = pilot()
        rng = np.random.default_rng(1)
        P, M = simplex(rng, 30), rng.integers(0, 2, 30)
        Pq, Mq = simplex(rng, 12), rng.integers(0, 2, 12)
        import torch
        torch.manual_seed(1)
        model = pl.build_attack_model('nn', 4, 64)
        mine = A.adjoint(model, P, M, Pq, Mq, [1., .1], pl.DEVICE)
        for g in (1., .1):
            ref = score.attack_implicit_v(copy.deepcopy(model).double(), P, M, Pq, Mq, g, pl.DEVICE)
            np.testing.assert_allclose(mine['per_gamma'][g]['v'], ref['v'], rtol=1e-9, atol=1e-13)
            self.assertAlmostEqual(mine['per_gamma'][g]['damped_eig_min'], ref['damped_eig_min'], places=10)


class TableTests(unittest.TestCase):
    def fixed(self):
        sid = np.r_[np.zeros(6), np.full(5, 3), np.full(4, 1)].astype(np.int64)
        cls = np.array([0, 1, 0, 1, 2, 3, 1, 0, 0, 2, 1, 3, 3, 0, 1])
        x = np.random.default_rng(0).random((15, 4)).astype(np.float32)
        x[sid == 3] = np.nan
        return dict(x=x, shadow_id=sid, classes=cls, membership=np.r_[np.ones(8), np.zeros(7)].astype(np.int64),
                    q_classes=np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3]))

    def test_assemble_and_class_rows(self):
        fx = self.fixed()
        p3 = np.arange(20, dtype=float).reshape(5, 4)
        x = A.assemble(fx, p3)
        np.testing.assert_array_equal(x[6:11], p3.astype(np.float32))
        self.assertFalse(np.isnan(x).any())
        tr, aff, where = A.class_rows(fx, 0)
        np.testing.assert_array_equal(tr, [0, 2, 7, 8, 13])
        np.testing.assert_array_equal(where, [1, 2])   # positions inside the shadow-3 block
        with self.assertRaises(ValueError):
            A.assemble(fx, np.zeros((4, 4)))

    def test_weights_follow_query_counts(self):
        w = A.weights(self.fixed()['q_classes'], [1, 3])
        self.assertAlmostEqual(w[1], 3 / 7)
        self.assertAlmostEqual(w[3], 4 / 7)
        with self.assertRaises(ValueError):
            A.weights(self.fixed()['q_classes'], [])


class PackTests(unittest.TestCase):
    def test_packs_never_cross_seeds(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('run_x4', Path(__file__).resolve().parent / 'run.py')
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        c = dict(seeds=[42, 43, 44], design=dict(m=11))
        cells = run.make_cells(c)
        packs = run.make_packs(cells, 20.)
        by = {x['name']: x for x in cells}
        self.assertEqual(sorted(n for p in packs for n in p), sorted(by))
        for p in packs:
            self.assertEqual(len({by[n]['seed'] for n in p}), 1)
            self.assertLessEqual(len(p), 8)


class PrepareTests(unittest.TestCase):
    def build(self, tmp):
        v3, full = tmp / 'v3', tmp / 'full'
        for d in (v3 / 'smoke', full / 'splits', full / 'checkpoints'):
            d.mkdir(parents=True)
        rt.write(full / 'experiment_config.json', dict(args=dict(n_shadow=4, affected_shadow=3, attack_epochs=50, attack_lr=.01,
                                                                 attack_batch_size=128, deterministic=True,
                                                                 attack_seeds=[5101, 5102, 5103, 5104, 5105])))
        rt.write(full / 'splits/fresh_patient_split.json', dict(shadow_models=[]))
        for n in ('target_fixed', 'shadow_0_fixed', 'shadow_1_fixed', 'shadow_2_fixed'):
            (full / 'checkpoints' / f'{n}.pt').write_bytes(n.encode())
        mods = [r for r in rt.REUSED]
        cells = []
        for s in (42, 43):
            cells.append(dict(name=f'full_seed{s}', group='full', kind='full', seed=s, source='harvest'))
            cells += [dict(name=f'img32_s{j:03d}_seed{s}', group='img32', kind='subset', subset=j, seed=s, source='train')
                      for j in range(3)]
        cv3 = dict(out=str(v3), schema='OCT_image_lds_v3', git_commit=rt.V3_COMMIT, design_name='img32',
                   dataset_fingerprint='fp', full=str(full), data_dir='d', repo=str(rt.REPO),
                   source_sha256={str(rt.REPO / r): rt.sha(rt.REPO / r) for r in mods}, cells=cells, seeds=[42, 43],
                   design=dict(kind='image_random', m=3, retain=.5, seed=1), input_sha256={}, harvest_tol=1e-10,
                   seed32_out='/s32', baseline_out='/b', lds_pilot_out='/p', epochs=140, recipe={})
        rt.write(v3 / 'config.json', cv3)
        (v3 / 'config.sha256').write_text(rt.sha(v3 / 'config.json') + '\n')
        rt.write(v3 / 'designs.json', dict(base={}, design={}))
        for f in ('result.json', 'designs.json'):
            rt.write(v3 / 'smoke' / f, {})
        rt.finish(v3 / 'smoke', cv3, dict(kind='smoke'))
        for cell in cells:
            d = v3 / 'cells' / cell['name']
            d.mkdir(parents=True)
            rt.write(d / 'result.json', dict(cell=cell))
            np.savez(d / 'interface.npz', raw=np.zeros(2))
            if cell['kind'] == 'subset':
                (d / 'final.pt').write_bytes(b'weights' + cell['name'].encode())
            rt.finish(d, cv3, cell)
        return v3, full

    def args(self, out, v3):
        from types import SimpleNamespace
        return SimpleNamespace(out=str(out), v3_out=str(v3), gammas='1,0.3,0.1,0.03,0.01,0.003', attack_gammas='1,0.1',
                               k_attack=3, attack_seeds='', score_seeds='', j00_tol=1e-6, lanczos_max=300, probe_steps=24,
                               partition='gpuA100x4', fp64_partition='gpuA100x4', score_time='02:00:00', pack_minutes=20.,
                               concurrency=8, audit='/a', primary_classes='1,2', primary_gamma=.03, background_recipe='stable',
                               cpu_account='acct-cpu', cpu_partition='cpu')

    def test_prepare_pins_v3_and_legacy(self):
        import importlib.util
        import tempfile
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('run_p4', Path(__file__).resolve().parent / 'run.py')
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v3, full = self.build(tmp)
            base = lambda a, schema: dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()),
                                          full=str(full), data_dir='d', dataset_fingerprint='fp', account='acct',
                                          input_sha256={}, source_sha256={})
            with patch.object(rt, 'prepare_base', side_effect=base):
                run.prepare(self.args(tmp / 'out', v3))
                c = rt.read(tmp / 'out/config.json')
                self.assertEqual(c['attack_seeds'], [5101, 5102, 5103])
                self.assertEqual(c['attack'], dict(epochs=50, lr=.01, batch=128, hidden=64, deterministic=True))
                self.assertEqual(c['score_seeds'], [42, 43])
                self.assertEqual(len(c['cells']), 6)
                self.assertTrue(all(str(v3 / 'cells' / x['name'] / 'complete.json') in c['input_sha256'] for x in c['v3_cells']))
                self.assertIn(str(full / 'checkpoints/shadow_2_fixed.pt'), c['input_sha256'])
                self.assertNotIn(str(full / 'checkpoints/shadow_3_fixed.pt'), c['input_sha256'])
                # a modified v3 subset model must stop prepare
                (v3 / 'cells/img32_s001_seed43/final.pt').write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    run.prepare(self.args(tmp / 'out2', v3))


class AnalysisTests(unittest.TestCase):
    def test_known_ladder(self):
        """L1 exact, L2-lin = truth + noise, L3 = monotone transform: LDS 1 / <1 / 1; unqualified class ignored."""
        rng = np.random.default_rng(3)
        m, seeds, K, C = 12, [42, 43], [5101, 5102], [0, 1, 2, 3]
        omega = {'0': .25, '1': .75}
        M = (rng.random((m, 20)) < .5).astype(float)
        effect = rng.normal(size=(m, len(seeds), len(K), len(C)))
        truth_rows = {}
        for j in range(m):
            for si, s in enumerate(seeds):
                rows = []
                for ki, k in enumerate(K):
                    for ci, cls in enumerate(C):
                        e = effect[j, si, ki, ci]
                        rows.append(dict(attack_seed=k, cls=cls, J00=1., J10=1. + e, J_L2_retrain=1. + .5 * e,
                                         L1={'1': e, '0.1': -e}, L2_lin=e + rng.normal() * 3))
                truth_rows[(j, s)] = dict(rows=rows, dtheta_norm=1., jvp=dict(row_clip_rate=0.),
                                          drift={g: dict(js_true=0., js_linearized=0., js_true_vs_linearized=0.)
                                                 for g in ('deleted', 'retained', 'nonmember')})
        w = np.array([.25, .75, 0, 0])
        truth = (effect * w).sum(3).mean((1, 2))
        pred = np.exp(truth)                    # monotone: LDS = 1
        scores = {s: dict(arrays={'pred_gamma1': pred, 'pred_proxy': -pred}, result=dict(
            gammas=[dict(gamma=1., qualified=True, reasons=[], true_rel_residual=1e-6, spearman_if_vs_proxy_images=-1.)],
            lanczos=dict(steps=5))) for s in seeds}
        base_rows = [dict(seed=s, attack_seed=k, cls=cls, metrics=dict(auc=.6, balanced_accuracy=.55, cross_entropy=.69),
                          hessian_eig_min=-.01, train_grad_norm=1e-3) for s in seeds for k in K for cls in C]
        data = dict(frozen=dict(omega=omega, qualified_classes=[0, 1], attack_gamma_star=1.), attack_base=base_rows,
                    smoke=dict(checks={}), M=M, base={}, truth=truth_rows, scores=scores, missing=[])
        c = dict(seeds=seeds, attack_seeds=K, classes=C, attack_gammas=[1., .1], gammas=[1.], primary_gamma=1., score_seeds=seeds)
        summary, tables = analysis4.analyze(data, c)
        lad = summary['ladder']
        self.assertAlmostEqual(lad['L1 (gamma_A=1)']['spearman'], 1.)
        self.assertAlmostEqual(lad['L1 (gamma_A=0.1)']['spearman'], -1.)
        self.assertAlmostEqual(lad['L2-retrain']['spearman'], 1.)
        self.assertAlmostEqual(lad['L3 (gamma=1)']['spearman'], 1.)
        self.assertAlmostEqual(lad['proxy']['spearman'], -1.)
        self.assertLess(lad['L2-lin']['spearman'], 1.)
        self.assertAlmostEqual(lad['L1 (gamma_A=1)']['slope_truth_on_prediction'], 1.)
        self.assertEqual(summary['ladder_gamma_star'], 1.)
        self.assertAlmostEqual(spearman(truth, truth), 1.)
        self.assertIn('L2-lin - L3 (gamma=1)', summary['differences'])
        self.assertFalse(summary['attack_quality']['2']['qualified'])


class SubmitTests(unittest.TestCase):
    def test_background_dependencies_and_cpu_collection(self):
        import importlib.util
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('run_submit4', Path(__file__).resolve().parent / 'run.py')
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        for background in ([0, 1, 2, 4], []):
            with self.subTest(background=background), tempfile.TemporaryDirectory() as tmp:
                c = dict(out=tmp, account='gpu-acct', cpu_account='cpu-acct', cpu_partition='cpu',
                         partition='gpuA100x4', fp64_partition='gpuA100x4', score_time='02:00:00',
                         background_shadows=background, seeds=list(range(42, 50)), score_seeds=list(range(42, 50)),
                         packs=list(range(32)), concurrency=8)
                with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output',
                      side_effect=[str(i) for i in range(100, 106)]) as submit:
                    run.submit(SimpleNamespace(out=tmp, score_time=None, fp64_partition=None))
                jobs = rt.read(Path(tmp) / 'jobs.json')
                calls = {call.args[0][-1]: call.args[0] for call in submit.call_args_list}
                self.assertEqual(set(jobs), {'smoke', 'base', 'pack:truth', 'score', 'collect'} | ({'background'} if background else set()))
                if background:
                    self.assertIn('--dependency=afterok:' + jobs['background'], calls['smoke'])
                for mode in ('pack:truth', 'score'):
                    self.assertIn('--dependency=afterok:' + jobs['base'], calls[mode])
                self.assertIn('--dependency=afterok:' + jobs['pack:truth'] + ':' + jobs['score'], calls['collect'])
                self.assertIn('--account=cpu-acct', calls['collect'])
                self.assertIn('--partition=cpu', calls['collect'])
                self.assertTrue(any(x.endswith('/collect.slurm') for x in calls['collect']))


if __name__ == '__main__':
    unittest.main()
