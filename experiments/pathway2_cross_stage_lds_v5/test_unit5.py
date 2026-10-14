"""CPU unit tests for v5: calibration metrics, the J00 gate, prepare (stable target, gate config),
submission (target in the background array), and the hybrid analysis on synthetic results."""
from pathlib import Path
import importlib.util
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime as rt  # noqa: E402
import core5  # noqa: E402
import analysis5  # noqa: E402


def own_run(name):
    spec = importlib.util.spec_from_file_location(name, HERE / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


class CalibrationTests(unittest.TestCase):
    def test_perfect_and_overconfident(self):
        y = np.r_[np.ones(50), np.zeros(50)].astype(int)
        good = core5.calibration(np.r_[np.full(50, .9), np.full(50, .1)], y)
        self.assertAlmostEqual(good['ece'], .1)
        self.assertEqual(good['confident_wrong_rate'], 0.)
        # 10 members predicted 0.001: confidently wrong, as for v4's unsaturated target members
        bad = core5.calibration(np.r_[np.full(40, .999), np.full(10, .001), np.full(50, .001)], y)
        self.assertAlmostEqual(bad['confident_wrong_rate'], .1)
        self.assertGreater(bad['ece'], good['ece'] - .1)

    def test_p_true_profile(self):
        x = np.array([[.995, .005, 0, 0], [.5, .5, 0, 0], [0, 1., 0, 0], [0, .98, .02, 0]])
        prof = core5.p_true_profile(x, np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1]))
        self.assertAlmostEqual(prof['member']['frac_below_0_99'], 1 / 3)
        self.assertEqual(prof['nonmember']['n'], 1)


class GateTests(unittest.TestCase):
    def rows(self, ce):
        return [dict(seed=s, attack_seed=k, cls=cls, metrics=dict(cross_entropy=ce[cls] + .01 * k, auc=.9, ece=.05,
                                                                  confident_wrong_rate=0.))
                for s in (42, 43) for k in (0, 1) for cls in (0, 1, 2, 3)]

    def test_gate_uses_every_primary_baseline(self):
        ok = core5.gate_decision(self.rows({0: 2., 1: .4, 2: .5, 3: 3.}), [1, 2], math.log(2))
        self.assertTrue(ok['passed'])                       # unqualified classes 0 and 3 do not count
        self.assertAlmostEqual(ok['per_class']['2']['J00_max'], .51)
        bad = core5.gate_decision(self.rows({0: .4, 1: .4, 2: .687, 3: .4}), [1, 2], math.log(2))
        self.assertFalse(bad['passed'])                     # max .697 > ln 2 although the mean is below
        self.assertFalse(bad['per_class']['2']['passed'])
        with self.assertRaises(ValueError):
            core5.gate_decision(self.rows({0: .4, 1: .4, 2: .4, 3: .4}), [5], math.log(2))


class PrepareTests(unittest.TestCase):
    def build(self, tmp):
        v3, full = tmp / 'v3', tmp / 'full'
        for d in (v3 / 'smoke', full / 'splits', full / 'checkpoints'):
            d.mkdir(parents=True)
        rt.write(full / 'experiment_config.json', dict(args=dict(n_shadow=5, affected_shadow=3, attack_epochs=50, attack_lr=.01,
                                                                 attack_batch_size=128, deterministic=True, target_seed=42007,
                                                                 attack_seeds=[5101, 5102, 5103, 5104, 5105])))
        rt.write(full / 'splits/fresh_patient_split.json', dict(shadow_models=[]))
        for n in ('target_fixed', 'shadow_0_fixed', 'shadow_1_fixed', 'shadow_2_fixed', 'shadow_4_fixed'):
            (full / 'checkpoints' / f'{n}.pt').write_bytes(n.encode())
        cells = []
        for s in (42, 43):
            cells.append(dict(name=f'full_seed{s}', group='full', kind='full', seed=s, source='harvest'))
            cells += [dict(name=f'img32_s{j:03d}_seed{s}', group='img32', kind='subset', subset=j, seed=s, source='train')
                      for j in range(3)]
        mods = [r for r in rt.REUSED if not r.startswith(rt.V4_PREFIX)]
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

    def args(self, out, v3, gate=math.log(2)):
        return SimpleNamespace(out=str(out), v3_out=str(v3), gammas='1,0.3,0.1,0.03,0.01,0.003', attack_gammas='1,0.1',
                               k_attack=3, attack_seeds='', score_seeds='', j00_tol=1e-6, lanczos_max=300, probe_steps=24,
                               partition='gpuA100x4', fp64_partition='gpuA100x4', score_time='02:00:00', pack_minutes=20.,
                               concurrency=8, audit='/a', primary_classes='1,2', primary_gamma=.03, j00_gate=gate,
                               v4_out='/v4', cpu_account='acct-cpu', cpu_partition='cpu')

    def test_prepare_stable_target_and_gate(self):
        run = own_run('run_p5')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v3, full = self.build(tmp)
            base = lambda a, schema: dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()),
                                          full=str(full), data_dir='d', dataset_fingerprint='fp', account='acct',
                                          input_sha256={}, source_sha256={})
            with patch.object(rt, 'prepare_base', side_effect=base):
                run.prepare(self.args(tmp / 'out', v3))
                c = rt.read(tmp / 'out/config.json')
                self.assertEqual(c['schema'], 'OCT_cross_stage_lds_v5')
                self.assertEqual(c['target_recipe'], 'stable')
                self.assertEqual(c['target_seed'], 42007)
                self.assertEqual(c['background_shadows'], [0, 1, 2, 4])
                self.assertEqual(c['background_jobs'], [0, 1, 2, 4, 'target'])
                self.assertAlmostEqual(c['j00_gate'], math.log(2))
                self.assertEqual(c['adjoint_identity_rel_tol'], 1e-9)
                self.assertEqual(c['primary_classes'], [1, 2])
                self.assertEqual(c['attack_seeds'], [5101, 5102, 5103])
                self.assertIn(str(full / 'checkpoints/target_fixed.pt'), c['input_sha256'])
                for rel in ('core4.py', 'attack4.py', 'analysis4.py'):
                    self.assertIn(str(rt.V4 / rel), c['source_sha256'])
                with self.assertRaises(ValueError):
                    run.prepare(self.args(tmp / 'out2', v3, gate=2.))

    def test_check_repo_allows_only_v4_and_v5_additions(self):
        rows = {'A\texperiments/pathway2_cross_stage_lds_v5/core5.py': True,
                'A\texperiments/pathway2_cross_stage_lds_v4/core4.py': True,
                'M\texperiments/pathway2_cross_stage_lds_v4/core4.py': False,
                'A\tscripts/cross_stage/new.py': False}
        for row, ok in rows.items():
            with self.subTest(row=row):
                calls = iter(['', row, ''])
                with patch.object(rt, 'git', side_effect=lambda *a: next(calls) if a[0] != 'rev-parse' else 'head'), \
                        patch.object(rt.subprocess, 'run'):
                    if ok:
                        self.assertEqual(rt.check_repo(), 'head')
                    else:
                        with self.assertRaises(ValueError):
                            rt.check_repo()
        calls = iter(['', 'A\texperiments/pathway2_cross_stage_lds_v5/core5.py', 'experiments/pathway2_cross_stage_lds_v4/core4.py'])
        with patch.object(rt, 'git', side_effect=lambda *a: next(calls)), patch.object(rt.subprocess, 'run'):
            with self.assertRaises(ValueError):          # v4 package edited after cc0cad6
                rt.check_repo()


class SubmitTests(unittest.TestCase):
    def test_target_in_background_array(self):
        run = own_run('run_s5')
        with tempfile.TemporaryDirectory() as tmp:
            c = dict(out=tmp, account='gpu-acct', cpu_account='cpu-acct', cpu_partition='cpu', partition='gpuA100x4',
                     fp64_partition='gpuA100x4', score_time='02:00:00', background_shadows=[0, 1, 2, 4],
                     background_jobs=[0, 1, 2, 4, 'target'], seeds=list(range(42, 50)), score_seeds=list(range(42, 50)),
                     packs=list(range(40)), concurrency=8)
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output',
                  side_effect=[str(i) for i in range(100, 106)]) as submit:
                run.submit(SimpleNamespace(out=tmp, score_time=None, fp64_partition=None))
            jobs = rt.read(Path(tmp) / 'jobs.json')
            calls = {call.args[0][-1]: call.args[0] for call in submit.call_args_list}
            self.assertIn('--array=0-4%8', calls['background'])
            self.assertIn('--dependency=afterok:' + jobs['background'], calls['smoke'])
            self.assertTrue(all(any(x.startswith('--job-name=oct_x5_') for x in v) for v in calls.values()))


class HybridAnalysisTests(unittest.TestCase):
    def test_known_hybrids(self):
        """Deleted rows carry the whole effect: Spearman 1 with the truth, nonmember 0 effect, additivity exact."""
        rng = np.random.default_rng(4)
        m, seeds, K, C, gA = 10, [42, 43], [5101, 5102], [0, 1, 2, 3], [1., .1]
        omega = {'1': .6, '2': .4}
        M = (rng.random((m, 20)) < .5).astype(float)
        effect = rng.normal(size=(m, len(seeds), len(K), len(C)))
        truth_rows, hyb = {}, {}
        for j in range(m):
            for si, s in enumerate(seeds):
                rows, hrows = [], []
                for ki, k in enumerate(K):
                    for cls in C:
                        e = effect[j, si, ki, cls]
                        rows.append(dict(attack_seed=k, cls=cls, J00=1., J10=1. + e, J_L2_retrain=1. + e, auc10=.9 + .01 * e,
                                         L1={'1': e, '0.1': e}, L2_lin=e))
                        hrows.append(dict(attack_seed=k, cls=cls, J00=1., J_deleted=1. + e, J_nonmember=1., auc_deleted=.9,
                                          auc_nonmember=.9, L1_deleted={'1': e, '0.1': e}, L1_nonmember={'1': 0., '0.1': 0.},
                                          L1_retained={'1': 0., '0.1': 0.}))
                truth_rows[(j, s)] = dict(rows=rows, dtheta_norm=1., jvp=dict(row_clip_rate=0.),
                                          drift={g: dict(js_true=0., js_linearized=0., js_true_vs_linearized=0.)
                                                 for g in ('deleted', 'retained', 'nonmember')})
                hyb[(j, s)] = dict(rows=hrows)
        base_rows = [dict(seed=s, attack_seed=k, cls=cls, metrics=dict(auc=.9, balanced_accuracy=.8, cross_entropy=.4),
                          hessian_eig_min=-.01, train_grad_norm=1e-3) for s in seeds for k in K for cls in C]
        data = dict(frozen=dict(omega=omega, qualified_classes=[1, 2], attack_gamma_star=1.), attack_base=base_rows,
                    smoke=dict(checks={}), M=M, base={}, truth=truth_rows, scores={}, missing=[], hybrid=hyb,
                    gate=dict(passed=True), target=dict(seed=42007))
        c = dict(seeds=seeds, attack_seeds=K, classes=C, attack_gammas=gA, gammas=[1.], primary_gamma=1., score_seeds=seeds)
        summary, tables = analysis5.analyze(data, c)
        h = summary['hybrid']
        self.assertAlmostEqual(h['groups']['deleted']['spearman_with_full_truth'], 1.)
        self.assertAlmostEqual(h['groups']['deleted']['fraction_of_full_shift'], 1.)
        self.assertAlmostEqual(h['groups']['deleted']['l1_group_lds_on_group_truth']['spearman'], 1.)
        self.assertAlmostEqual(h['groups']['nonmember']['truth_mean'], 0.)
        self.assertAlmostEqual(h['additivity']['interaction_mean'], 0.)
        self.assertLess(h['l1_partition_max_abs_error'], 1e-12)
        self.assertAlmostEqual(summary['truth_shape']['spearman_delta_auc_vs_delta_ce'], 1.)
        self.assertEqual(len(tables['hybrid']), 2)


if __name__ == '__main__':
    unittest.main()
