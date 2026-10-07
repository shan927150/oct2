"""Regression tests for scientific failure modes and retry/manifest contracts."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt
import run
import scheduler
import solver
import stats2 as S
import targets as T


class NumericalFailures(unittest.TestCase):
    def test_singular_shift_preserves_other_gamma(self):
        ms = solver.multishift_lanczos(lambda v: -.1 * v, np.ones(1), [1., .1], 5, 1e-10)
        self.assertIn(.1, ms['errors'])
        self.assertNotIn(.1, ms['x'])
        np.testing.assert_allclose(ms['x'][1.], [1 / .9])
        json.dumps(ms['history'], allow_nan=False)

    def test_invalid_gamma_rejected(self):
        for gammas in ([1., float('nan')], [1., 1.], [], [-1.]):
            with self.assertRaises(ValueError):
                solver.multishift_lanczos(lambda v: v, np.ones(2), gammas, 5, 1e-6)

    def test_underflow_ce_truth_stays_finite(self):
        probs = np.array([[0., .5, .5], [0., .5, .5]])
        logs = np.array([[-1000., -np.log(2), -np.log(2)], [-1100., -np.log(2), -np.log(2)]])
        vals = T.all_values(probs, [0, 0], [.5, .5], [np.array([0, 1])], logs)
        self.assertEqual(vals['ce']['Y'], 1050.)
        self.assertEqual(vals['ce']['Yq'].tolist(), [1050.])
        # The unstable legacy interface derivative is refused, not silently clipped.
        with self.assertRaises(FloatingPointError):
            T.value_and_v('ce', probs, [0, 0], [.5, .5])

    def test_failed_legacy_replay_withholds_every_shift(self):
        import analysis
        d = dict(scores={(42, 'ce'): dict(arrays={'tau_gamma1': np.ones(2)},
                                        result=dict(lds_v1_consistency={'passed': False}))})
        tau, reason = analysis.if_scores(d, {'score_seeds': [42]}, 'ce', 1.)
        self.assertIsNone(tau)
        self.assertIn('replay mismatch', reason)


class StatisticalContracts(unittest.TestCase):
    def test_duplicate_queries_do_not_narrow_ci(self):
        rng = np.random.default_rng(22)
        p = rng.normal(size=(8, 1))
        y = p + rng.normal(size=(8, 1))
        a = S.per_query_lds(p, y, draws=80)
        b = S.per_query_lds(np.repeat(p, 163, axis=1), np.repeat(y, 163, axis=1), draws=80)
        np.testing.assert_allclose(a['bootstrap_ci_conditional_on_seed_panel'], b['bootstrap_ci_conditional_on_seed_panel'])
        self.assertEqual(a['permutation_p_one_sided'], b['permutation_p_one_sided'])

    def test_constant_queries_have_no_significance(self):
        r = S.per_query_lds(np.ones((8, 12)), np.ones((8, 12)), draws=20)
        for k in ('mean', 'median', 'permutation_p_one_sided', 'bootstrap_ci_conditional_on_seed_panel'):
            self.assertIsNone(r[k])
        self.assertEqual(r['defined'], 0)

    def test_one_seed_cannot_estimate_repeatability(self):
        y = np.arange(8.)[:, None]
        self.assertIsNone(S.se_curve(y, [1], draws=10)[0]['analytic_se_mean_over_subsets'])
        r = S.reliability_curve(y, [1], draws=10)['rows'][0]
        self.assertFalse(r['independent_split_available'])
        self.assertIsNone(r['icc_reliability'])

    def test_negative_reliability_is_not_extrapolated(self):
        y = np.column_stack([np.arange(8.), -np.arange(8.)])
        r = S.reliability_curve(y, [1, 2], draws=10)['rows']
        self.assertLess(r[0]['split_spearman_mean'], 0)
        self.assertTrue(all(x['spearman_brown_from_k1'] is None for x in r))

    def test_fitted_composition_does_not_use_fixed_prediction_null(self):
        x = np.arange(8.)
        r = S.lds_block(x, x, x ** 2, fitted_on_truth=True)
        self.assertAlmostEqual(r['spearman'], 1.)
        self.assertNotIn('permutation_p_one_sided', r)
        self.assertNotIn('bootstrap_ci_conditional_on_seed_panel', r)

    def test_equal_design_has_fixed_optimizer_steps(self):
        import designs
        rng = np.random.default_rng(9)
        sizes = np.maximum(1, rng.lognormal(2.2, .8, 148).astype(int))
        patients = [dict(oct_class=i % 4, rows=list(range(n))) for i, n in enumerate(sizes)]
        d = designs.equal_size_stratified(patients, 32, .5, 20261006, diagnostic_pool=100)
        steps = np.ceil(np.asarray(d['masks']) @ sizes / 128)
        self.assertEqual(set(steps), {d['optimizer_steps_per_epoch']})


class ArtifactAndSchedulerContracts(unittest.TestCase):
    def test_wrong_identity_and_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt.write(root / 'config.json', {})
            d = root / 'result'; d.mkdir()
            rt.write(d / 'payload.json', {'ok': True})
            rt.finish(d, {'out': tmp}, {'id': 42})
            h = rt.sha(root / 'config.json')
            with self.assertRaises(ValueError):
                rt.completed(d, h, {'id': 43})
            (d / 'payload.json').write_text('{}')
            with self.assertRaises(ValueError):
                rt.completed(d, h, {'id': 42})

    def test_empty_manifest_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt.write(Path(tmp) / 'complete.json', dict(status='complete', config_sha256='h', output_sha256={}))
            with self.assertRaises(ValueError):
                rt.completed(tmp, 'h')

    def test_live_and_unknown_jobs_block_retry(self):
        records = [dict(job_id='12')]
        with patch.object(scheduler.subprocess, 'check_output', return_value='12_0\n'):
            with self.assertRaises(RuntimeError):
                scheduler.require_terminal(records)
        for accounting in ('', '12|RUNNING|\n'):
            with patch.object(scheduler.subprocess, 'check_output', side_effect=['', accounting]):
                with self.assertRaises(RuntimeError):
                    scheduler.require_terminal(records)
        with patch.object(scheduler.subprocess, 'check_output', side_effect=['', '12|COMPLETED|\n12_0|FAILED|\n']):
            scheduler.require_terminal(records)

    def test_unknown_sbatch_receipt_blocks_resubmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = dict(out=tmp, account='acct')
            with (Path(tmp) / 'submission.jsonl').open('x') as ledger:
                with patch.object(run.subprocess, 'check_output', side_effect=OSError('transport lost')):
                    with self.assertRaises(OSError):
                        run.launcher(c, ledger)('smoke', [])
            self.assertTrue((Path(tmp) / 'submission_pending.json').exists())
            with self.assertRaises(RuntimeError):
                scheduler.receipts(tmp)

    def test_atomic_smoke_copy_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt.write(root / 'config.json', {})
            (root / 'smoke').mkdir()
            rt.write(root / 'smoke/designs.json', {'masks': [1, 0]})
            rt.write(root / 'smoke/result.json', {})
            rt.finish(root / 'smoke', {'out': tmp}, dict(kind='smoke'))
            h = rt.sha(root / 'config.json')
            run.ensure_designs(root, h)
            self.assertEqual(rt.sha(root / 'designs.json'), rt.sha(root / 'smoke/designs.json'))
            (root / 'designs.json').write_text('{}')
            with self.assertRaises(ValueError):
                run.ensure_designs(root, h)

    def test_submission_file_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            with rt.lock(path):
                with self.assertRaises(RuntimeError):
                    with rt.lock(path):
                        pass

    def test_gamma_pilot_submits_only_three_frozen_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(out=tmp, account='acct', concurrency=8, score_time='02:00:00', packs={},
                     gamma_pilot=dict(seeds=[42, 43, 44], target='oneminusp', gammas=[1., .3, .1]))
            rt.write(root / 'config.json', c)
            (root / 'smoke').mkdir()
            rt.write(root / 'smoke/result.json', {})
            rt.write(root / 'smoke/designs.json', {})
            rt.finish(root / 'smoke', c, {'kind': 'smoke'})
            with patch.object(run, 'config', return_value=c), patch.object(run.subprocess, 'check_output', return_value='700\n') as submit:
                run.resubmit(SimpleNamespace(out=tmp, only='score-pilot', score_time=None))
            cmd = submit.call_args[0][0]
            self.assertIn('--array=0-2%8', cmd)
            self.assertEqual(cmd[-1], 'score-pilot')
            self.assertEqual(len(scheduler.receipts(tmp)), 1)

    def test_queue_wait_and_gpu_allocation(self):
        raw = ('JobID|JobName|Submit|Start|End|Elapsed|State|AllocTRES|\n'
               '100|train|2026-10-06T10:00:00|2026-10-06T10:30:00||00:10:00|COMPLETED|gres/gpu=4|\n'
               '100_0|train|2026-10-06T10:00:00|2026-10-06T10:30:00||00:10:00|COMPLETED|gres/gpu=1|\n'
               '101|cpu|2026-10-06T10:00:00|2026-10-06T10:01:00||00:20:00|COMPLETED|cpu=2|\n'
               '102|pending|2026-10-06T10:00:00|Unknown||00:00:00|PENDING||\n')
        out = io.StringIO()
        with patch.object(run.subprocess, 'check_output', return_value=raw), redirect_stdout(out):
            run.queue_report(SimpleNamespace(since='2026-10-06', out=None))
        report = json.loads(out.getvalue())
        self.assertEqual(report['train']['tasks'], 1)
        self.assertEqual(report['train']['wait_minutes_median'], 30.)
        self.assertAlmostEqual(report['train']['gpu_hours'], 1 / 6)
        self.assertEqual(report['cpu']['gpu_hours'], 0.)
        self.assertIsNone(report['pending']['gpu_hours'])


if __name__ == '__main__':
    unittest.main()
