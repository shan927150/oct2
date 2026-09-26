#!/usr/bin/env python3
"""Numerical and isolation-contract tests; torch tests are mandatory in smoke."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import tarfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import numpy as np
from audit_math import VectorMoments, summary_rows
import run as controller

REQUIRE_TORCH = '--require-torch' in sys.argv
if REQUIRE_TORCH:
    sys.argv.remove('--require-torch')
try:
    import torch
    from audit_torch import snapshot
except ImportError:
    torch = None
    if REQUIRE_TORCH:
        raise RuntimeError('PyTorch tests are required by the GPU smoke gate')


class MonteCarloTests(unittest.TestCase):
    def test_vector_se_and_paired_covariance_against_batch_calculation(self):
        rng = np.random.default_rng(29)
        # Correlated full/patient gradients: omitting covariance gives wrong SE.
        g = rng.normal(size=(71, 9))
        draws = np.stack([g, .8 * g + .1 * rng.normal(size=g.shape), -g], axis=1)
        acc = VectorMoments()
        for draw in draws:
            acc.update(draw)
        np.testing.assert_allclose(acc.mean, draws.mean(0), atol=1e-15)
        for w in ([1, 0, 0], [0, 1, 0], [1, -.1, 0], [1, -1, 0], [1, 0, -1]):
            residuals = np.einsum('rvp,v->rp', draws, w)
            expected = np.sqrt(np.var(residuals, axis=0, ddof=1).sum() / len(draws))
            self.assertAlmostEqual(acc.se(w), expected, places=13)

    def test_centering_is_stable_for_large_offsets(self):
        draws = 1e8 + np.arange(60, dtype=float).reshape(10, 2, 3) / 10
        acc = VectorMoments()
        for draw in draws:
            acc.update(draw)
        expected = np.sqrt(np.var(draws[:, 0], axis=0, ddof=1).sum() / 10)
        self.assertAlmostEqual(acc.se([1, 0]), expected, places=6)

    def test_unresolved_zero_mean_does_not_become_zero_uncertainty(self):
        acc = VectorMoments()
        acc.update([[2., 0.], [1., 0.]])
        acc.update([[-2., 0.], [-1., 0.]])
        rows = summary_rows(acc.mean, [807], acc)
        self.assertIsNone(rows[0]['residual_over_perturbation'])
        self.assertLess(rows[0]['b_norm_sq_minus_mc_variance'], 0)
        self.assertGreater(rows[0]['b_vector_se'], 0)

    def test_dose_scaling_and_direction(self):
        rows = summary_rows(np.array([[3., 4.], [1., 0.], [0., 2.]]), [807, 2085])
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]['residual_over_perturbation'], 50.)
        self.assertAlmostEqual(rows[1]['residual_over_perturbation'], 5.)
        self.assertAlmostEqual(rows[0]['cosine_g_b'], .6)


class ControllerTests(unittest.TestCase):
    def make_fixture(self, root):
        """Small on-disk provenance fixture, without data/model dependencies."""
        repo, sp, audit, probe, full, data = [root / x for x in ('repo', 'sp', 'audit', 'probe', 'full', 'data')]
        for p in (repo, sp, audit, probe, full, data):
            p.mkdir()
        (repo / 'legacy.py').write_text('# frozen source\n')
        (full / 'splits').mkdir()
        panel = [dict(patient_id=807, oct_class=1, raw_indices=list(range(6))),
                 dict(patient_id=2085, oct_class=2, raw_indices=list(range(6, 11)))]
        for rel, value in [('experiment_config.json', {'args': {}}),
                ('experiment_summary.json', {'status': 'complete'}),
                ('selected_patients.json', {'patients': panel}),
                ('splits/fresh_patient_split.json', {})]:
            controller.write(full / rel, value)
        cfg = dict(git_commit=controller.BASE, probe_root=str(probe), full_dir=str(full),
                   data_dir=str(data), account='test_account', dataset_fingerprint={'fixture': True})
        controller.write(sp / 'smallpanel_config.json', cfg)
        (sp / 'config.sha256').write_text(controller.sha(sp / 'smallpanel_config.json') + '\n')
        old = dict(probe=str(probe), full=str(full), data_dir=str(data), input_sha256={})
        for seed in (42, 44):
            d = probe / f'shadow3_seed{seed}_decay'
            (d / 'checkpoints').mkdir(parents=True)
            ck = d / 'checkpoints/epoch140.pt'
            ck.write_bytes(b'fixture checkpoint, never loaded')
            old['input_sha256'][str(ck)] = controller.sha(ck)
            controller.write(d / 'manifest.json', dict(schema='pathway2_A_lr_continuation_v1',
                status='complete', seed=seed, arm='decay', resume_epoch=100, total_epochs=140,
                base_lr=.001, weight_decay=1e-5, decay_schedule='20@1e-4,20@1e-5',
                args={'checkpoint_epochs': [120, 140]}, resume_check={'passed': True},
                input_files_unchanged=True, source_files_unchanged=True, dataset_arrays_unchanged=True,
                source_sha256={'legacy.py': controller.sha(repo / 'legacy.py')}, epoch_order_sha256='order'))
            controller.write(d / 'dataset_fingerprint.json', cfg['dataset_fingerprint'])
            (d / 'curve.csv').write_text('epoch,eval_ce\n140,0.001\n')
            controller.write(d / 'input_sha256.json', {str(full / rel): controller.sha(full / rel)
                for rel in ('experiment_config.json', 'splits/fresh_patient_split.json')})
            (audit / f'seed{seed}').mkdir()
            (audit / f'seed{seed}/checkpoint_summary.csv').write_text('arm,epoch,eval_ce\ndecay,140,0.001\n')
            (audit / f'seed{seed}/paired_differences.csv').write_text('difference\n0\n')
        controller.write(audit / 'config.json', old)
        old_sha = controller.sha(audit / 'config.json')
        (audit / 'config.sha256').write_text(old_sha + '\n')
        for seed in (42, 44):
            controller.write(audit / f'seed{seed}/complete.json', dict(status='complete',
                config_sha256=old_sha, model_states_unchanged=True, input_files_unchanged=True))
        args = type('Args', (), dict(out=root/'out', smallpanel=sp, loss_audit=audit, profile='pilot', reps=128))()
        return repo, args

    def test_prepare_freezes_provenance_and_rejects_input_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            repo, args = self.make_fixture(Path(d))
            with mock.patch.object(controller, 'REPO', repo), \
                 mock.patch.object(controller, 'check_repo', return_value='commit'), \
                 mock.patch.object(controller, 'git', return_value='legacy.py'):
                controller.prepare(args)
                cfg = controller.load_config(args.out)
                self.assertEqual(len(cfg['cells']), 2)
                ck = Path(cfg['cells'][0]['checkpoint'])
                ck.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    controller.load_config(args.out)

    def test_collect_requires_all_cells_and_omits_large_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            out.mkdir()
            cfg = dict(out=str(out), cells=controller.cells_for('pilot'), mc_reps=128)
            controller.write(out / 'config.json', cfg)
            config_hash = controller.sha(out / 'config.json')
            vectors = np.array([[3., 4.], [1., 0.], [0., 2.]])
            for i, cell in enumerate([None] + cfg['cells']):
                dest = out / (cell['name'] if cell else 'smoke')
                dest.mkdir()
                rows = [dict(seed=42, arm='decay', epoch=140, objective=kind, **r)
                        for kind in ('eval', 'dropout_expected') for r in summary_rows(vectors, [807, 2085])]
                controller.write_csv(dest / 'summary.csv', rows)
                (dest / 'gradient_vectors.npz').write_bytes(b'large-vectors-placeholder')
                controller.write(dest / 'complete.json', dict(status='complete', config_sha256=config_hash,
                    cell=cell, mc_reps=128, smoke=cell is None,
                    output_sha256={p.name: controller.sha(p) for p in dest.iterdir()}))
            with mock.patch.object(controller, 'load_config', return_value=cfg):
                args = type('Args', (), {'out': out})()
                controller.collect(args)
                with self.assertRaises(FileExistsError):
                    controller.collect(args)
            with tarfile.open(out.with_suffix('.tar.gz')) as tar:
                names = tar.getnames()
                self.assertIn('out/report/stationarity_summary.csv', names)
                self.assertFalse(any(n.endswith('.npz') for n in names))

    def test_profiles_only_use_saved_checkpoints(self):
        self.assertEqual(len(controller.cells_for('pilot')), 2)
        matrix = controller.cells_for('matrix')
        self.assertEqual(len(matrix), 8)
        self.assertEqual({r['epoch'] for r in matrix}, {120, 140})
        self.assertEqual({r['seed'] for r in matrix}, {42, 44})

    def test_overlap_protection_including_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'input').mkdir()
            (root / 'alias').symlink_to(root / 'input', target_is_directory=True)
            for out in (root, root / 'input', root / 'alias/subdir'):
                with self.assertRaises(ValueError):
                    controller.disjoint(out, root / 'input')
            controller.disjoint(root / 'fresh', root / 'input')

    def test_result_integrity_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            (dest / 'summary.csv').write_text('a,b\n1,2\n')
            controller.write(dest / 'complete.json', dict(status='complete', config_sha256='cfg',
                output_sha256={'summary.csv': controller.sha(dest / 'summary.csv')}))
            controller.completed(dest, 'cfg')
            (dest / 'summary.csv').write_text('a,b\n9,9\n')
            with self.assertRaises(ValueError):
                controller.completed(dest, 'cfg')

    def test_submit_has_dependency_serial_array_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(out=d, account='test_account', cells=controller.cells_for('pilot'))
            with mock.patch.object(controller, 'load_config', return_value=cfg), \
                 mock.patch.object(controller.subprocess, 'check_output', side_effect=['111;cluster\n', '112\n']) as call:
                args = type('Args', (), {'out': Path(d)})()
                controller.submit(args)
                cmd = call.call_args_list[1].args[0]
                self.assertIn('--dependency=afterok:111', cmd)
                self.assertIn('--array=0-1%1', cmd)
                with self.assertRaises(FileExistsError):
                    controller.submit(args)
                self.assertEqual(call.call_count, 2)

    def test_partial_submission_retains_smoke_id(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(out=d, account='test_account', cells=controller.cells_for('pilot'))
            with mock.patch.object(controller, 'load_config', return_value=cfg), \
                 mock.patch.object(controller.subprocess, 'check_output', side_effect=['111\n', RuntimeError('queue error')]):
                with self.assertRaises(RuntimeError):
                    controller.submit(type('Args', (), {'out': Path(d)})())
            entries = (Path(d) / 'submission.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(entries[0])['job_id'], '111')


@unittest.skipIf(torch is None, 'PyTorch unavailable locally; required in Delta smoke')
class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = torch.nn.Linear(2, 3).double()
        self.X = np.array([[.1, .3], [.7, -.2], [-.4, .6], [.3, -.5], [.2, .9]])
        self.y = np.array([0, 1, 2, 0, 2])
        self.kw = dict(model=self.model, X=self.X, y=self.y, indices=np.arange(5),
                       patient_rows={807: [0, 3], 2085: [2]}, batch=2, wd=.07)

    def analytical(self, weights, wd):
        W = self.model.weight.detach().numpy()
        b = self.model.bias.detach().numpy()
        logits = self.X @ W.T + b
        prob = np.exp(logits - logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        err = (prob - np.eye(3)[self.y]) * np.asarray(weights)[:, None] / 5
        return np.r_[(err.T @ self.X + wd * W).ravel(), err.sum(0) + wd * b]

    def test_full_patient_and_deleted_gradients_against_analytic_softmax(self):
        _, v = snapshot(**self.kw, check_identity=True)
        np.testing.assert_allclose(v[0], self.analytical(np.ones(5), .07), atol=2e-15)
        for j, mask in enumerate(([1, 0, 0, 1, 0], [0, 0, 1, 0, 0]), 1):
            np.testing.assert_allclose(v[j], self.analytical(mask, 0), atol=2e-15)
            for alpha in (.1, 1.):
                np.testing.assert_allclose(v[0] - alpha * v[j],
                    self.analytical(1 - alpha * np.array(mask), .07), atol=2e-15)

    def test_directional_derivative_by_finite_difference(self):
        _, v = snapshot(**self.kw)
        original = np.r_[self.model.weight.detach().numpy().ravel(), self.model.bias.detach().numpy()]
        direction = np.arange(1., len(original)+1)
        direction /= np.linalg.norm(direction)
        weights = 1. - .1 * np.array([1, 0, 0, 1, 0])
        def objective(theta):
            logits = self.X @ theta[:6].reshape(3, 2).T + theta[6:]
            top = logits.max(1)
            ce = top + np.log(np.exp(logits - top[:, None]).sum(1)) - logits[np.arange(5), self.y]
            return (ce * weights).sum() / 5 + .07 * np.dot(theta, theta) / 2
        h = 1e-6
        finite = (objective(original+h*direction)-objective(original-h*direction))/(2*h)
        self.assertAlmostEqual(finite, np.dot(v[0]-.1*v[1], direction), places=8)

    def test_l2_excluded_from_patient_term(self):
        _, v = snapshot(**self.kw)
        _, no_l2 = snapshot(**{**self.kw, 'wd': 0})
        np.testing.assert_array_equal(v[1:], no_l2[1:])
        theta = np.concatenate([p.detach().numpy().ravel() for p in self.model.parameters()])
        np.testing.assert_allclose(v[0]-no_l2[0], .07*theta, atol=1e-15)

    def test_dropout_shared_masks_identity_and_state_preservation(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(),
                                    torch.nn.Dropout(.3), torch.nn.Linear(4, 3)).double()
        model.train()
        model[0].eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .123
        before = copy.deepcopy(model.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state().clone()
        python_rng, np_rng = random.getstate(), np.random.get_state()
        kwargs = {**self.kw, 'model': model, 'dropout': True, 'seed': 928}
        meta, first = snapshot(**kwargs, check_identity=True)
        _, second = snapshot(**kwargs)
        _, different = snapshot(**{**kwargs, 'seed': 929})
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.linalg.norm(first-different), 1e-8)
        self.assertTrue(all(x['passed'] for x in meta['identity_checks']))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(python_rng, random.getstate())
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        for p, g in zip(model.parameters(), grads):
            self.assertTrue(torch.equal(p.grad, g))


if __name__ == '__main__':
    unittest.main(verbosity=2)
