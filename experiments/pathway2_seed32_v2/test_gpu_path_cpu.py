"""Runs the real GPU-side code path (core2.Session with the legacy pilot/score/
stable kernels) on CPU with a tiny synthetic OCT-like context. Exercises:
training vs harvest cells, the smoke replay/no-op gates, proxies and the
multi-shift IF including its consistency check against legacy CG."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt  # noqa: E402

RECIPE = dict(lr=.001, warmup_frac=.1, ema_decay=.99, wd=1e-5, batch=128)


def legacy():
    import importlib.util
    cross = rt.REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(rt.REPO / 'src')]
    spec = importlib.util.spec_from_file_location('cpu_score', cross / '07_cross_stage_score_ladder.py')
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score, score.import_pilot_module()


def context(seed=0):
    rng = np.random.default_rng(seed)
    tr, q = rng.integers(2, 6, 12), rng.integers(1, 4, 8)
    groups = np.r_[np.repeat(100 + np.arange(12), tr), np.repeat(900 + np.arange(8), q)]
    y = np.r_[np.repeat(np.arange(12) % 4, tr), np.repeat(np.arange(8) % 4, q)]
    X = (rng.normal(size=(len(y), 1, 16, 16)) + y[:, None, None, None] * .3).astype(np.float32)
    train, test = np.arange(tr.sum()), np.arange(tr.sum(), len(y))
    return dict(X=X, y=y, groups=groups, train=train, test=test, interface=np.r_[train, test])


class GpuPathOnCpu(unittest.TestCase):
    def test_session(self):
        import torch
        import core2
        from design import make_design
        score, pilot = legacy()
        ctx = context()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, full, base_out, pilot_out = tmp / 'exp', tmp / 'full', tmp / 'baseline', tmp / 'pilot'
            for d in (root, full, base_out / 'seed42_baseline', pilot_out / 'score_seed42', pilot_out / 'subset000_seed42'):
                d.mkdir(parents=True)
            c = dict(out=str(root), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                     epochs=2, recipe=RECIPE, batch=64,
                     # A random-ish tiny CNN has Hessian eigenvalues below -1; a large objective L2
                     # makes H + gamma I positive definite so the gamma=1 consistency check is meaningful.
                     weight_decay=2., gammas=[1., .3, .03],
                     gamma_pilot=dict(seeds=[42, 43, 44], target='oneminusp', gammas=[1., .3, .1]),
                     lanczos_max_steps=60, lanczos_est_tol=1e-6, probe_steps=8, cg_fail_tol=1e-3,
                     designs=dict(pilot8=dict(kind='pilot', m=8, retain=.5),
                                  eq32=dict(kind='eq', m=6, retain=.5, seed=4, total_tol_frac=.5, class_tol_frac=1.)))
            rt.write(root / 'config.json', c)
            for s in (42, 43, 44):
                np.savez(full / f'stage1_order_seed{s}.npz', raw_index_order=pilot.make_epoch_orders(ctx['train'], 2, s + 700000))
            base = make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929)
            rt.write(pilot_out / 'score_seed42/design.json', base)
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                s0 = core2.Session(c, freeze_to=tmp / 'designs_tmp.json')
                # ---- frozen "stable baseline" and "LDS v1" artefacts built by the legacy kernels
                m, e, opt, _ = s0.train(dict(kind='full', seed=42))
                torch.save(dict(raw=m.state_dict(), ema=e.state_dict(), optimizer=opt.state_dict()),
                           base_out / 'seed42_baseline/final.pt')
                probs = {k: pilot.get_predictions(x, ctx['X'][ctx['interface']]) for k, x in (('raw', m), ('ema', e))}
                np.savez(base_out / 'seed42_baseline/interface.npz', raw_index=ctx['interface'], **probs)
                m2, e2, _, _ = s0.train(dict(kind='subset', design='pilot8', subset=0, seed=42))
                torch.save(dict(raw=m2.state_dict(), ema=e2.state_dict()), pilot_out / 'subset000_seed42/final.pt')
                np.savez(pilot_out / 'subset000_seed42/interface.npz', raw_index=ctx['interface'],
                         raw=s0.predict(s0.initialize(42, m2.state_dict(), double=True)))
                md = s0.initialize(42, m.state_dict(), double=True)
                P = s0.predict(md)
                import targets as T
                _, v = T.value_and_v('ce', P[s0.N:], s0.labels_test, s0.weights)
                h = score.shadow_h(md, ctx['X'][ctx['test']], v, pilot.DEVICE, 64)
                hvp1 = score.make_hvp(md, ctx['X'], ctx['y'], ctx['train'], pilot.DEVICE, c['weight_decay'], 1., 64)
                cg = score.conjugate_gradient(hvp1, h, 200, 1e-10)
                self.assertFalse(cg['nonpositive_curvature'])
                w = cg['x']
                B = s0.patient_matrix(md)
                with (pilot_out / 'score_seed42/scores.csv').open('w', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['patient', 'tau_upweight', 'frozen_h_upweight'])
                    for p, b in zip(s0.patients, B):
                        wr.writerow([p['patient'], -float(w @ b), -float(h @ b)])
                # ---- smoke: subset replay + full no-op + timing
                smoke = tmp / 'smoke'
                smoke.mkdir()
                s0.smoke(smoke)
                res = rt.read(smoke / 'result.json')
                self.assertTrue(res['noop_exact'] and res['pilot_replay_exact'])
                (root / 'designs.json').write_text((tmp / 'designs_tmp.json').read_text())
                s = core2.Session(c)
                # ---- cells
                cells = [dict(name='full_seed42', kind='full', seed=42, source='harvest'),
                         dict(name='pilot8_s000_seed42', kind='subset', design='pilot8', subset=0, seed=42, source='harvest'),
                         dict(name='full_seed45', kind='full', seed=45, source='train'),
                         dict(name='eq32_s002_seed45', kind='subset', design='eq32', subset=2, seed=45, source='train')]
                for cell in cells:
                    self.assertEqual(s.run_cell(cell, 't'), 'done')
                    self.assertEqual(s.run_cell(cell, 't'), 'skipped')
                r = rt.read(root / 'cells/eq32_s002_seed45/result.json')
                self.assertEqual(r['n_train_images'], int(np.asarray(s.designs['eq32']['masks'][2]) @
                                                          [len(p['rows']) for p in s.patients]))
                self.assertEqual(r['optimizer_steps'], 2)
                # ---- proxies (seed42 checked against LDS v1 frozen_h)
                self.assertEqual(s.proxy_seed(42, 't'), 'done')
                self.assertEqual(s.proxy_seed(45, 't'), 'done')
                pr = rt.read(root / 'proxy/seed42/result.json')
                self.assertLess(pr['checks']['lds_v1_frozen_h_ce_rel_difference'], 1e-10)
                # ---- multi-shift IF; CE gamma=1 must reproduce legacy CG
                self.assertEqual(s.score_task(42, 'ce', 't'), 'done')
                sr = rt.read(root / 'score/seed42_ce/result.json')
                self.assertTrue(sr['lds_v1_consistency']['passed'], sr['lds_v1_consistency'])
                self.assertLess(sr['lds_v1_consistency']['rel_l2_difference'], 1e-6)
                g1 = next(x for x in sr['gammas'] if x['gamma'] == 1.)
                self.assertTrue(g1['qualified'])
                self.assertEqual(s.score_task(45, 'oneminusp', 't'), 'done')
                self.assertEqual(s.score_task(42, 'oneminusp', 'pilot', pilot=True), 'done')
                with np.load(root / 'score_pilot/seed42_oneminusp/scores.npz') as z:
                    self.assertIn('tau_gamma0.1', z.files)
                    self.assertNotIn('tau_gamma0.03', z.files)
                self.assertFalse((root / 'score/seed42_oneminusp').exists())
                # A failed numerical screen is preserved as an assessment with no
                # tau arrays. Completion must not mean that the shifts solved.
                with patch.object(core2.solver, 'probe_screen', return_value={'positive': False}):
                    self.assertEqual(s.score_task(45, 'ce', 'unqualified'), 'assessed_with_unqualified_shifts')
                uq = rt.read(root / 'score/seed45_ce/result.json')
                self.assertTrue(all(not g['qualified'] for g in uq['gammas']))
                with np.load(root / 'score/seed45_ce/scores.npz') as z:
                    self.assertFalse(any(k.startswith('tau_gamma') for k in z.files))
                # A hard legacy replay mismatch must never acquire a complete
                # marker; a retry cannot skip it as a success.
                old_tau = s.old_scores(42, 'tau_upweight')
                (root / 'score/seed42_ce').rename(root / 'score/seed42_ce_verified')
                with patch.object(s, 'old_scores', return_value=2 * old_tau):
                    with self.assertRaisesRegex(ValueError, 'differ from LDS v1'):
                        s.score_task(42, 'ce', 'mismatch')
                self.assertFalse((root / 'score/seed42_ce/complete.json').exists())
                self.assertTrue((root / 'score/.seed42_ce.tmp-mismatch/failed.json').exists())
                # Saturated logits: probabilities underflow to zero, but exact
                # CE values and direct log-softmax gradients remain finite.
                saturated = s.initialize(45, s.baseline_state(45)['raw'], double=True)
                last = [layer for layer in saturated.modules() if isinstance(layer, torch.nn.Linear)][-1]
                with torch.no_grad():
                    last.weight.zero_()
                    last.bias.copy_(torch.tensor([-1000., 0., 0., 0.], dtype=torch.float64))
                probs_sat, logs_sat = s.predict(saturated, return_log=True)
                self.assertTrue(np.any(probs_sat == 0))
                ce_sat = s.values(probs_sat, logs_sat)['ce']['Y']
                grad_sat = s.query_gradient(saturated, 'ce', ctx['test'], s.weights, probs_sat[s.N:])
                self.assertGreater(ce_sat, 200.)
                self.assertTrue(torch.isfinite(grad_sat).all())
                self.assertGreater(float(grad_sat.norm()), 0.)
                # ---- a corrupted harvest source must be refused
                with np.load(pilot_out / 'subset000_seed42/interface.npz') as z:
                    raw = z['raw'].copy()
                raw[0, 0] += 1e-6
                np.savez(pilot_out / 'subset000_seed42/interface.npz', raw_index=ctx['interface'], raw=raw)
                with self.assertRaises(ValueError):
                    s.run_cell(dict(name='pilot8_s000_seed42_again', kind='subset', design='pilot8', subset=0,
                                    seed=42, source='harvest'), 'u')
                print(json.dumps(dict(gammas=sr['gammas'], lanczos=sr['lanczos']), indent=1)[:1500])


if __name__ == '__main__':
    unittest.main()
