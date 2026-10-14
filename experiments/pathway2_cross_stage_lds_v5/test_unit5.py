"""CPU unit tests for v5 r1: quality metrics and the CE gate, prepare against a v4 run (reuse pins),
submission (interp job), the hybrid residual/interaction names (including the retained-only
counterexample from the 2026-10-14 review), the v4 pairing with its replay check, the projection
helpers and the interpolation summary."""
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


# ------------------------------------------------------------------ helpers / metrics
class MetricTests(unittest.TestCase):
    def test_calibration_and_prior(self):
        y = np.r_[np.ones(50), np.zeros(50)].astype(int)
        good = core5.calibration(np.r_[np.full(50, .9), np.full(50, .1)], y)
        self.assertAlmostEqual(good['ece'], .1)
        self.assertEqual(good['confident_wrong_rate'], 0.)
        bad = core5.calibration(np.r_[np.full(40, .999), np.full(10, .001), np.full(50, .001)], y)
        self.assertAlmostEqual(bad['confident_wrong_rate'], .1)
        self.assertAlmostEqual(core5.prior_ce(y), math.log(2))
        self.assertAlmostEqual(core5.prior_ce(np.r_[np.ones(25), np.zeros(75)]), -(.25 * math.log(.25) + .75 * math.log(.75)))

    def test_p_true_profile_by_class(self):
        x = np.array([[.995, .005, 0, 0], [.5, .5, 0, 0], [0, 1., 0, 0], [0, .98, .02, 0]])
        prof = core5.p_true_profile(x, np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1]))
        self.assertAlmostEqual(prof['all']['member']['frac_below_0_99'], 1 / 3)
        self.assertAlmostEqual(prof['by_class']['1']['member']['frac_below_0_99'], .5)
        self.assertEqual(prof['by_class']['0']['nonmember']['n'], 1)

    def test_projection_is_not_identity_at_saturation(self):
        p = np.array([[1., 0, 0, 0], [.4, .3, .2, .1]])
        q = core5.project(p)
        self.assertGreater(np.abs(q - p)[0].sum(), 0)          # saturated row is modified
        np.testing.assert_allclose(q[1], p[1])                  # interior row untouched
        st = core5.projection_stats(np.array([[1.1, -.1, 0, 0], [.4, .3, .2, .1]]))
        self.assertEqual(st['rows_negative'], .5)
        self.assertEqual(st['rows_modified'], .5)


class GateTests(unittest.TestCase):
    def rows(self, ce):
        return [dict(seed=s, attack_seed=k, cls=cls, metrics=dict(cross_entropy=ce[cls] + .01 * k, auc=.9, ece=.05, brier=.1,
                                                                  confident_wrong_rate=0., prior_ce=.69,
                                                                  cross_entropy_legacy=1.2, cross_entropy_legacy_fp64=1.2))
                for s in (42, 43) for k in (0, 1) for cls in (0, 1, 2, 3)]

    def test_gate_uses_every_primary_baseline(self):
        ok = core5.gate_decision(self.rows({0: 2., 1: .4, 2: .5, 3: 3.}), [1, 2], math.log(2))
        self.assertTrue(ok['passed'])
        self.assertEqual(ok['name'], 'CE quality gate')
        self.assertAlmostEqual(ok['per_class']['2']['J00_max'], .51)
        self.assertAlmostEqual(ok['per_class']['1']['legacy_J00_mean'], 1.2)
        bad = core5.gate_decision(self.rows({0: .4, 1: .4, 2: .687, 3: .4}), [1, 2], math.log(2))
        self.assertFalse(bad['passed'])                     # max .697 > ln 2 although the mean is below
        with self.assertRaises(ValueError):
            core5.gate_decision(self.rows({0: .4, 1: .4, 2: .4, 3: .4}), [5], math.log(2))


# ------------------------------------------------------------------ prepare / submit
class PrepareTests(unittest.TestCase):
    def build_v3(self, tmp):
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

    def build_v4(self, tmp, c0):
        """A completed v4-style run with the same design fields as the prepared v5 config c0."""
        v4 = tmp / 'v4'
        same = ('full', 'data_dir', 'dataset_fingerprint', 'v3_out', 'v3_config_sha256', 'seeds', 'epochs', 'recipe', 'design',
                'attack', 'attack_seeds', 'classes', 'primary_classes', 'attack_gammas', 'gammas', 'score_seeds',
                'fixed_shadow_seed', 'background_shadows', 'batch', 'weight_decay', 'cells')
        c4 = {k: c0[k] for k in same}
        c4.update(out=str(v4), schema='OCT_cross_stage_lds_v4', git_commit=rt.V4_COMMIT, background_recipe='stable')
        c4['cells'] = [dict(x, minutes=2.5) for x in c4['cells']]
        v4.mkdir()
        rt.write(v4 / 'config.json', c4)
        (v4 / 'config.sha256').write_text(rt.sha(v4 / 'config.json') + '\n')

        def done(d, cell, files):
            d.mkdir(parents=True)
            for f in files:
                (d / f).write_bytes(f.encode() + str(d).encode())
            rt.finish(d, c4, cell)
        done(v4 / 'smoke', dict(kind='xsmoke'), ['result.json', 'designs.json', 'frozen.json', 'fixed_rows.npz'])
        for sid in c4['background_shadows']:
            done(v4 / 'background' / f'shadow{sid}', dict(kind='background', shadow=sid), ['result.json', 'final.pt', 'interface.npz'])
        for s in c4['seeds']:
            done(v4 / 'base' / f'seed{s}', dict(kind='base', seed=s), ['result.json', 'v.npz', 'h.npz'])
        for cell in c4['cells']:
            done(v4 / 'truth' / cell['name'], cell, ['result.json'])
        for s in c4['score_seeds']:
            done(v4 / 'score' / f'seed{s}', dict(kind='xscore', seed=s), ['result.json', 'scores.npz'])
        (v4 / 'report').mkdir()
        rt.write(v4 / 'report/complete.json', dict(status='complete', config_sha256=rt.sha(v4 / 'config.json'), missing=[]))
        return v4

    def args(self, out, v3, v4, gate=math.log(2)):
        return SimpleNamespace(out=str(out), v3_out=str(v3), gammas='1,0.3,0.1,0.03,0.01,0.003', attack_gammas='1,0.1',
                               k_attack=3, attack_seeds='', score_seeds='', j00_tol=1e-6, lanczos_max=300, probe_steps=24,
                               partition='gpuA100x4', fp64_partition='gpuA100x4', score_time='02:00:00', pack_minutes=20.,
                               concurrency=8, audit='/a', primary_classes='1,2', primary_gamma=.03, j00_gate=gate,
                               v4_out=str(v4), cpu_account='acct-cpu', cpu_partition='cpu')

    def test_prepare_pins_v4_and_reuses_backgrounds(self):
        run = own_run('run_p5')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v3, full = self.build_v3(tmp)
            base = lambda a, schema: dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()),
                                          full=str(full), data_dir='d', dataset_fingerprint='fp', account='acct',
                                          input_sha256={}, source_sha256={}, batch=64, weight_decay=1e-5)
            with patch.object(rt, 'prepare_base', side_effect=base):
                with patch.object(run, 'check_v4', return_value=('h', {})):
                    run.prepare(self.args(tmp / 'probe', v3, tmp / 'nov4'))
                v4 = self.build_v4(tmp, rt.read(tmp / 'probe/config.json'))
                run.prepare(self.args(tmp / 'out', v3, v4))
                c = rt.read(tmp / 'out/config.json')
                self.assertEqual(c['schema'], 'OCT_cross_stage_lds_v5r1')
                self.assertEqual(c['target_seed'], 42007)
                self.assertEqual(c['background_jobs'], [0, 1, 2, 4, 'target'])
                self.assertEqual(sorted(c['background_source']), ['0', '1', '2', '4'])
                self.assertEqual(c['v4_config_sha256'], rt.sha(v4 / 'config.json'))
                self.assertIn(str(v4 / 'background/shadow2/final.pt'), c['input_sha256'])
                self.assertIn(str(v4 / 'smoke/fixed_rows.npz'), c['input_sha256'])
                self.assertEqual(c['interp']['subsets'], [0, 1, 2, 3])
                self.assertEqual(c['interp']['seeds'], [42, 43])
                self.assertIn(str(rt.V4 / 'core4.py'), c['source_sha256'])
                with self.assertRaises(ValueError):                      # gate outside (0, 1]
                    run.prepare(self.args(tmp / 'out2', v3, v4, gate=2.))
                (v4 / 'background/shadow1/final.pt').write_bytes(b'tampered')
                with self.assertRaises(ValueError):                      # a modified v4 background is refused
                    run.prepare(self.args(tmp / 'out3', v3, v4))

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
            with self.assertRaises(ValueError):
                rt.check_repo()


class SubmitTests(unittest.TestCase):
    def test_target_and_interp_jobs(self):
        run = own_run('run_s5')
        with tempfile.TemporaryDirectory() as tmp:
            c = dict(out=tmp, account='gpu-acct', cpu_account='cpu-acct', cpu_partition='cpu', partition='gpuA100x4',
                     fp64_partition='gpuA100x4', score_time='02:00:00', background_shadows=[0, 1, 2, 4],
                     background_jobs=[0, 1, 2, 4, 'target'], seeds=list(range(42, 50)), score_seeds=list(range(42, 50)),
                     packs=list(range(32)), concurrency=8)
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output',
                  side_effect=[str(i) for i in range(100, 107)]) as submit:
                run.submit(SimpleNamespace(out=tmp, score_time=None, fp64_partition=None))
            jobs = rt.read(Path(tmp) / 'jobs.json')
            calls = {call.args[0][-1]: call.args[0] for call in submit.call_args_list}
            self.assertIn('--array=0-4%8', calls['background'])
            self.assertIn('--dependency=afterok:' + jobs['base'], calls['interp'])
            self.assertFalse(any(x.startswith('--array') for x in calls['interp']))
            self.assertIn('--dependency=afterok:' + ':'.join([jobs['pack:truth'], jobs['score'], jobs['interp']]), calls['collect'])
            self.assertTrue(all(any(x.startswith('--job-name=oct_x5_') for x in v) for v in calls.values()))


# ------------------------------------------------------------------ analysis
def synthetic(effect, rng, m, seeds, K, C, legacy_shift=0., hybrid=None):
    """v5-style truth rows; effect[j, si, ki, c] is the full-truth Delta for the new target."""
    truth, hyb = {}, {}
    for j in range(m):
        for si, s in enumerate(seeds):
            rows, hrows = [], []
            for ki, k in enumerate(K):
                for cls in C:
                    e = effect[j, si, ki, cls]
                    rows.append(dict(attack_seed=k, cls=cls, J00=1., J10=1. + e, J_L2_retrain=1. + e, auc10=.9, brier00=.2,
                                     brier10=.2 + e, L1={'1': e, '0.1': e}, L2_lin=e, J_proj0=1.01, J_L2_logit=1. + e,
                                     J00_legacy=1.2, J10_legacy=1.2 + e + legacy_shift, J_L2_retrain_legacy=1.2 + e,
                                     J_L2_logit_legacy=1.2 + e))
                    if hybrid is not None:
                        parts = {g: hybrid[g][j, si, ki, cls] for g in ('deleted', 'retained', 'nonmember', 'deleted_nonmember')}
                        hrows.append(dict(attack_seed=k, cls=cls, J00=1., J00_legacy=1.2,
                                          L1_by_group={g: {'1': parts[g], '0.1': parts[g]} for g in ('deleted', 'retained', 'nonmember')},
                                          J={g: 1. + v for g, v in parts.items()}, J_legacy={g: 1.2 + v for g, v in parts.items()}))
            truth[(j, s)] = dict(rows=rows, dtheta_norm=1., jvp=dict(row_clip_rate=0.),
                                 drift={g: dict(js_true=0., js_linearized=0., js_true_vs_linearized=0.)
                                        for g in ('deleted', 'retained', 'nonmember')})
            hyb[(j, s)] = dict(rows=hrows)
    return truth, hyb


class AnalysisTests(unittest.TestCase):
    m, seeds, K, C = 12, [42, 43], [5101, 5102], [0, 1, 2, 3]
    cfg = dict(seeds=[42, 43], attack_seeds=[5101, 5102], classes=[0, 1, 2, 3], attack_gammas=[1., .1], gammas=[1.],
               primary_gamma=1., score_seeds=[42, 43], j00_tol=1e-6)

    def data(self, truth, hyb, M):
        base = [dict(seed=s, attack_seed=k, cls=cls, metrics=dict(auc=.9, balanced_accuracy=.8, cross_entropy=.4),
                     hessian_eig_min=-.01, train_grad_norm=1e-3) for s in self.seeds for k in self.K for cls in self.C]
        rng = np.random.default_rng(9)
        scores = {s: dict(arrays={'pred_gamma1': rng.normal(size=len(M)), 'pred_proxy': rng.normal(size=len(M))}, result=dict(
            gammas=[dict(gamma=1., qualified=True, reasons=[], true_rel_residual=1e-6, spearman_if_vs_proxy_images=0.)],
            lanczos=dict(steps=5))) for s in self.seeds}
        return dict(frozen=dict(omega={'1': .6, '2': .4}, qualified_classes=[1, 2], attack_gamma_star=1.), attack_base=base,
                    smoke=dict(checks={}), M=M, base={}, truth=truth, scores=scores, missing=[], hybrid=hyb, interp=None, v4=None,
                    gate=dict(passed=True), target=dict(seed=42007))

    def test_retained_only_counterexample(self):
        """Only retained rows act, effects are additive: residual and interaction must both be 0."""
        rng = np.random.default_rng(1)
        shape = (self.m, 2, 2, 4)
        R = rng.normal(size=shape) * .1
        zero = np.zeros(shape)
        hyb = dict(deleted=zero, retained=R, nonmember=zero, deleted_nonmember=zero)
        truth, hrows = synthetic(R, rng, self.m, self.seeds, self.K, self.C, hybrid=hyb)
        M = (rng.random((self.m, 20)) < .5).astype(float)
        s, tables = analysis5.analyze(self.data(truth, hrows, M), self.cfg)
        new = s['hybrid']['targets']['new']
        self.assertAlmostEqual(new['nonadditive_residual']['mean'], 0., places=12)
        self.assertAlmostEqual(new['interaction_deleted_nonmember']['mean'], 0., places=12)
        self.assertAlmostEqual(new['groups']['retained']['spearman_with_full_truth'], 1.)
        self.assertAlmostEqual(new['groups']['retained']['l1_group_lds_on_group_truth']['spearman'], 1.)
        self.assertLess(s['hybrid']['l1_partition_max_abs_error'], 1e-12)

    def test_true_interaction_is_reported_as_interaction(self):
        rng = np.random.default_rng(2)
        shape = (self.m, 2, 2, 4)
        D, N, I = (rng.normal(size=shape) * .1 for _ in range(3))
        hyb = dict(deleted=D, retained=np.zeros(shape), nonmember=N, deleted_nonmember=D + N + I)
        truth, hrows = synthetic(D + N + I, rng, self.m, self.seeds, self.K, self.C, hybrid=hyb)
        M = (rng.random((self.m, 20)) < .5).astype(float)
        s, _ = analysis5.analyze(self.data(truth, hrows, M), self.cfg)
        new = s['hybrid']['targets']['new']
        w = np.array([0, .6, .4, 0])
        expect = (I * w).sum(3).mean((1, 2)).mean()
        self.assertAlmostEqual(new['interaction_deleted_nonmember']['mean'], expect, places=12)
        self.assertAlmostEqual(new['nonadditive_residual']['mean'], expect, places=12)

    def test_extra_rungs_and_legacy(self):
        rng = np.random.default_rng(3)
        eff = rng.normal(size=(self.m, 2, 2, 4))
        truth, hrows = synthetic(eff, rng, self.m, self.seeds, self.K, self.C, legacy_shift=.3)
        M = (rng.random((self.m, 20)) < .5).astype(float)
        s, tables = analysis5.analyze(self.data(truth, hrows, M), self.cfg)
        self.assertAlmostEqual(s['ladder_extra']['L2-retrain (logit linearization)']['spearman'], 1.)
        self.assertAlmostEqual(s['ladder_extra']['L2-retrain (vs Pi(P0))']['spearman'], s['ladder']['L2-retrain']['spearman'])
        self.assertAlmostEqual(s['projection_only']['mean_shift'], .01)
        self.assertAlmostEqual(s['legacy_in_v5']['spearman_truth_new_vs_old'], 1.)   # common shift does not move ranks
        self.assertAlmostEqual(s['truth_shape']['spearman_delta_brier_vs_delta_ce'], 1.)
        self.assertEqual(s['paired_v4']['status'], 'no v4 run configured')

    def test_pairing_with_v4_and_replay_check(self):
        rng = np.random.default_rng(4)
        eff = rng.normal(size=(self.m, 2, 2, 4))
        truth, hrows = synthetic(eff, rng, self.m, self.seeds, self.K, self.C)
        M = (rng.random((self.m, 20)) < .5).astype(float)
        data = self.data(truth, hrows, M)
        # v4 rows = the legacy fields of v5, with a noisy L2-lin so that the rungs differ
        old = {}
        for key, cell in truth.items():
            rows = []
            for r in cell['rows']:
                rows.append(dict(attack_seed=r['attack_seed'], cls=r['cls'], J00=r['J00_legacy'], J10=r['J10_legacy'],
                                 J_L2_retrain=r['J_L2_retrain_legacy'], auc10=.9, L1=r['L1'], L2_lin=r['L2_lin'] + rng.normal() * 5))
            old[key] = dict(cell, rows=rows)
        d4 = dict(data, truth=old)
        data['v4'] = dict(c=self.cfg, data=d4)
        s, tables = analysis5.analyze(data, self.cfg)
        pv = s['paired_v4']
        self.assertTrue(pv['replay_passed'])
        self.assertEqual(pv['status'], 'paired')
        self.assertAlmostEqual(pv['rungs']['L2-lin']['lds_a'], 1.)
        self.assertGreater(pv['rungs']['L2-lin']['difference'], 0.)
        self.assertAlmostEqual(pv['rungs']['L2-retrain']['difference'], 0.)
        self.assertTrue(tables['paired_v4'])
        # a single non-replayed J10 blocks the pairing
        bad = {k: dict(v, rows=[dict(r) for r in v['rows']]) for k, v in old.items()}
        bad[(0, 42)]['rows'][0]['J10'] += 1e-3
        data['v4'] = dict(c=self.cfg, data=dict(d4, truth=bad))
        s, _ = analysis5.analyze(data, self.cfg)
        self.assertFalse(s['paired_v4']['replay_passed'])
        self.assertNotIn('rungs', s['paired_v4'])


class InterpTests(unittest.TestCase):
    def test_linear_and_nonlinear_regimes(self):
        rows = []
        rng = np.random.default_rng(5)
        for u in range(12):
            l1 = rng.normal()
            for t in (1e-3, 1e-2, .1, 1.):
                dj = t * l1 + (t ** 2) * 5 * rng.normal()             # linear for small t, curved for large t
                rows.append(dict(t=t, dJ=dj, dJ_legacy=dj, L1={'0.3': l1}, L1_legacy={'0.3': l1}, dP_norm=t, dP_norm_float32=t))
        out = analysis5.interp_summary(dict(panel=dict(ts=[1e-3, 1e-2, .1, 1.]), rows=rows), dict(attack_gamma_star=.3), dict(interp=[]))
        small, big = out['by_t']['0.001']['new'], out['by_t']['1']['new']
        self.assertLess(small['median_relative_error'], .05)
        self.assertGreater(small['spearman_slope_vs_l1'], .95)
        self.assertGreater(big['median_relative_error'], small['median_relative_error'])
        self.assertAlmostEqual(out['by_t']['0.001']['float32_effective_ratio_median'], 1.)


if __name__ == '__main__':
    unittest.main()
