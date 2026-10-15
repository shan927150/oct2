"""CPU unit tests for v6: prepare against a v5 r1 run (pins, attack-seed continuation), the repo
contract, submission, and the analysis on synthetic results with known answers."""
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
import analysis6  # noqa: E402


def own_run(name):
    spec = importlib.util.spec_from_file_location(name, HERE / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


class PrepareTests(unittest.TestCase):
    def build_v5(self, tmp, panel):
        full, v5 = tmp / 'full', tmp / 'v5'
        (full).mkdir()
        rt.write(full / 'experiment_config.json', dict(args=dict(attack_seeds=panel)))
        cells = [dict(name=f'truth_s{j:03d}_seed{s}', kind='truth', subset=j, seed=s, minutes=2.4) for s in (42, 43) for j in range(3)]
        c5 = {k: None for k in own_run('run_c').COPY}
        c5.update(out=str(v5), schema='OCT_cross_stage_lds_v5r1', git_commit=rt.V5_COMMIT, full=str(full), data_dir='d',
                  dataset_fingerprint='fp', seeds=[42, 43], score_seeds=[42, 43], cells=cells, attack_seeds=[5101, 5102, 5103],
                  input_sha256={})
        v5.mkdir()
        rt.write(v5 / 'config.json', c5)
        (v5 / 'config.sha256').write_text(rt.sha(v5 / 'config.json') + '\n')

        def done(d, cell, files):
            d.mkdir(parents=True)
            for f in files:
                (d / f).write_bytes(f.encode() + str(d).encode())
            rt.finish(d, c5, cell)
        done(v5 / 'smoke', dict(kind='xsmoke'), ['result.json', 'designs.json', 'frozen.json', 'fixed_rows.npz', 'gate.json'])
        for s in (42, 43):
            done(v5 / 'base' / f'seed{s}', dict(kind='base', seed=s), ['result.json', 'v.npz', 'h.npz'])
            done(v5 / 'score' / f'seed{s}', dict(kind='xscore', seed=s), ['result.json', 'scores.npz'])
        for cell in cells:
            done(v5 / 'truth' / cell['name'], cell, ['result.json'])
        (v5 / 'report').mkdir()
        rt.write(v5 / 'report/complete.json', dict(status='complete', config_sha256=rt.sha(v5 / 'config.json'), missing=[]))
        return v5, full

    def args(self, out, v5, new='5104,5105,5106'):
        return SimpleNamespace(out=str(out), v5_out=str(v5), n_new=3, new_attack_seeds=new, partition='gpuA100x4',
                               fp64_partition='gpuA100x4', score_time='03:00:00', cpu_account='c', cpu_partition='cpu',
                               concurrency=8, audit='/a')

    def run_prepare(self, panel, new='5104,5105,5106'):
        run = own_run('run_p6')
        tmp = Path(tempfile.mkdtemp())
        v5, full = self.build_v5(tmp, panel)
        base = lambda a, schema: dict(schema=schema, repo=str(rt.REPO), git_commit='x', out=str(Path(a.out).resolve()),
                                      full=str(full), data_dir='d', dataset_fingerprint='fp', account='acct',
                                      input_sha256={}, source_sha256={})
        with patch.object(rt, 'prepare_base', side_effect=base):
            run.prepare(self.args(tmp / 'out', v5, new))
        return rt.read(tmp / 'out/config.json'), tmp, v5, run, base

    def test_explicit_new_seeds_when_panel_is_short(self):
        c, tmp, v5, run, base = self.run_prepare([5101, 5102, 5103])
        self.assertEqual(c['schema'], 'OCT_cross_stage_lds_v6')
        self.assertEqual(c['attack_seeds'], [5101, 5102, 5103, 5104, 5105, 5106])
        self.assertEqual(c['new_attack_seed_source'], 'explicit (legacy panel too short)')
        self.assertIn(str(v5 / 'base/seed43/h.npz'), c['input_sha256'])
        self.assertIn(str(v5 / 'smoke/fixed_rows.npz'), c['input_sha256'])
        self.assertIn(str(v5 / 'truth/truth_s002_seed42/result.json'), c['input_sha256'])
        self.assertEqual(len(c['packs']), 2)
        self.assertIn(str(rt.V5 / 'core5.py'), c['source_sha256'])
        self.assertIn('P2', c['primary_tests'])
        (v5 / 'base/seed42/v.npz').write_bytes(b'tampered')
        with patch.object(rt, 'prepare_base', side_effect=base), self.assertRaises(ValueError):
            run.prepare(self.args(tmp / 'out2', v5))

    def test_panel_continuation_is_used_and_checked(self):
        c, *_ = self.run_prepare([5101, 5102, 5103, 5201, 5202, 5203, 5204], new='')
        self.assertEqual(c['new_attack_seeds'], [5201, 5202, 5203])
        with self.assertRaises(ValueError):
            self.run_prepare([5101, 5102, 5103, 5201, 5202, 5203])          # default 5104-5106 disagrees with the panel
        with self.assertRaises(ValueError):
            self.run_prepare([5101, 5102, 5103], new='5104,5105')          # wrong count

    def test_check_repo_requires_v4_and_v5_unchanged(self):
        ok = 'A\texperiments/pathway2_cross_stage_lds_v6/core6.py'
        for row, v5diff, good in ((ok, '', True), ('A\texperiments/pathway2_cross_stage_lds_v5/core5.py', '', True),
                                  ('M\tsrc/models.py', '', False), (ok, 'experiments/pathway2_cross_stage_lds_v5/core5.py', False)):
            with self.subTest(row=row, v5diff=v5diff):
                answers = iter(['', row, '', v5diff, 'head'])
                with patch.object(rt, 'git', side_effect=lambda *a: next(answers)), patch.object(rt.subprocess, 'run'):
                    if good:
                        self.assertEqual(rt.check_repo(), 'head')
                    else:
                        with self.assertRaises(ValueError):
                            rt.check_repo()


class SubmitTests(unittest.TestCase):
    def test_dependencies(self):
        run = own_run('run_s6')
        with tempfile.TemporaryDirectory() as tmp:
            c = dict(out=tmp, account='g', cpu_account='c', cpu_partition='cpu', partition='gpuA100x4', fp64_partition='gpuA100x4',
                     score_time='03:00:00', seeds=list(range(42, 50)), score_seeds=list(range(42, 50)), packs=list(range(32)),
                     concurrency=8)
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output',
                  side_effect=[str(i) for i in range(100, 105)]) as sub:
                run.submit(SimpleNamespace(out=tmp, score_time=None, fp64_partition=None))
            jobs = rt.read(Path(tmp) / 'jobs.json')
            calls = {x.args[0][-1]: x.args[0] for x in sub.call_args_list}
            self.assertEqual(set(jobs), {'smoke', 'base', 'pack:truth', 'score', 'collect'})
            self.assertIn('--dependency=afterok:' + jobs['smoke'], calls['base'])
            for m in ('pack:truth', 'score'):
                self.assertIn('--dependency=afterok:' + jobs['base'], calls[m])
            self.assertIn('--dependency=afterok:' + jobs['pack:truth'] + ':' + jobs['score'], calls['collect'])
            self.assertTrue(all(any(x.startswith('--job-name=oct_x6_') for x in v) for v in calls.values()))


class AnalysisTests(unittest.TestCase):
    def synthetic(self, l3_member_good=True):
        rng = np.random.default_rng(7)
        m, seeds, K, C, gA = 32, [42, 43], [5101, 5102, 5201, 5202], [0, 1, 2, 3], [.3, .1]
        sig = rng.normal(size=m)
        truth = {}
        for j in range(m):
            for s in seeds:
                rows = []
                for k in K:
                    for cls in C:
                        e = sig[j] + .3 * rng.normal()
                        rows.append(dict(attack_seed=k, cls=cls, source='v5r1' if k < 5200 else 'v6', J00=1., J10=1. + e,
                                         L1={'0.3': e + rng.normal(), '0.1': 0.}, L1_member={'0.3': e, '0.1': 0.},
                                         L2_lin=rng.normal(), L2_lin_member=rng.normal()))
                truth[(j, s)] = dict(rows=rows, j10_replay_abs=0.)
        M = (rng.random((m, 40)) < .5).astype(float)

        def res(v):
            return dict(gammas=[dict(gamma=g, qualified=True, reasons=[]) for g in (1., .03)])
        scores = {}
        for s in seeds:
            arr = {'full_pred_gamma1': rng.normal(size=m), 'full_pred_gamma0.03': sig + 3 * rng.normal(size=m),
                   'member_pred_gamma1': rng.normal(size=m),
                   'member_pred_gamma0.03': sig + (.1 if l3_member_good else 30) * rng.normal(size=m),
                   'full_pred_proxy': rng.normal(size=m), 'member_pred_proxy': rng.normal(size=m)}
            scores[s] = dict(arrays=arr, result=dict(variants={'full': res('full'), 'member': res('member')},
                                                      cos_hbar_full_member=.9, hbar_replay_vs_v5_max_abs=0.))
        data = dict(frozen=dict(omega={'1': .6, '2': .4}, qualified_classes=[1, 2], attack_gamma_star=.3),
                    gate=dict(passed=True), M=M, truth=truth, scores=scores, base={}, missing=[], v5_scores={})
        c = dict(seeds=seeds, attack_seeds=K, old_attack_seeds=K[:2], new_attack_seeds=K[2:], classes=C, attack_gammas=gA,
                 gammas=[1., .03], primary_gamma=.03, score_seeds=seeds)
        return data, c

    def test_known_primary_results(self):
        data, c = self.synthetic(True)
        s, tables = analysis6.analyze(data, c)
        self.assertTrue(s['primary']['P2']['passed'])
        self.assertGreater(s['primary']['P2']['lds'], .9)
        self.assertGreater(s['primary']['P3']['difference'], 0)
        self.assertGreater(s['primary']['P3']['ci'][0], 0)
        self.assertGreater(s['ladder']['L1 (member)']['spearman'], s['ladder']['L1 (full)']['spearman'])
        self.assertGreater(s['attack_seed_reliability']['split_half_old_vs_new'], .8)
        self.assertGreater(s['ladder']['L1 (member)']['lds_truth_new'], .8)    # new-seed-only truth, matching seeds
        self.assertEqual(len(tables['attack_pairs']), 6)
        data, c = self.synthetic(False)
        s, _ = analysis6.analyze(data, c)
        self.assertFalse(s['primary']['P2']['passed'])

    def test_unqualified_primary_gamma_is_reported_not_substituted(self):
        data, c = self.synthetic(True)
        for sc in data['scores'].values():
            sc['result']['variants']['member']['gammas'][1].update(qualified=False, reasons=['residual'])
        s, _ = analysis6.analyze(data, c)
        self.assertIn('unqualified', s['primary']['P2']['status'])
        self.assertNotIn('P3', s['primary'])


if __name__ == '__main__':
    unittest.main()
