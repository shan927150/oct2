"""Runs the real GPU-side code path (core3.Session + legacy stable/HVP kernels +
seed32 solver) on CPU with a tiny synthetic OCT-like context. Covers smoke gates,
harvest of seed32 full baselines, image-subset training, the image-level proxy
(including the LDS v1 frozen_h patient gate) and the multi-shift IF (including
the LDS v1 CG patient gate), then collect on the produced artefacts."""
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
    spec = importlib.util.spec_from_file_location('cpu_score3', cross / '07_cross_stage_score_ladder.py')
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
    def test_session_and_collect(self):
        import torch
        import core3
        import run
        from design import make_design
        score, pilot = legacy()
        ctx = context()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, full, base_out, pilot_out, s32 = (tmp / x for x in ('exp', 'full', 'baseline', 'pilot', 'seed32'))
            for d in (root, full, pilot_out / 'score_seed42'):
                d.mkdir(parents=True)
            seeds = [42, 45, 46]
            c = dict(out=str(root), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                     seed32_out=str(s32), epochs=2, recipe=RECIPE, batch=64,
                     # A tiny CNN has eigenvalues below -1; a large objective L2 keeps H + gamma I positive
                     # definite so the gamma=1 patient gate against the legacy CG solve is meaningful.
                     weight_decay=2., gammas=[1., .3, .03], lanczos_max_steps=80, lanczos_est_tol=1e-7,
                     probe_steps=8, cg_fail_tol=1e-3, harvest_tol=1e-10, cross_gpu_tol=1e-7,
                     design_name='img8', design=dict(kind='image_random', m=8, retain=.5, seed=3),
                     seeds=seeds, score_seeds=[42, 45], score_targets=['ce', 'oneminusp', 'brier'], input_sha256={})
            c['cells'] = run.make_cells(c)
            c['proxy_tasks'], c['score_tasks'] = seeds, c['score_seeds']
            for s in (42, 43, 44):
                np.savez(full / f'stage1_order_seed{s}.npz', raw_index_order=pilot.make_epoch_orders(ctx['train'], 2, s + 700000))
            base = make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929)
            rt.write(pilot_out / 'score_seed42/design.json', base)
            rt.write(root / 'config.json', c)
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                s0 = core3.Session(c, freeze_to=tmp / 'designs_tmp.json')
                # ---- upstream artefacts built by the legacy kernels: stable baseline (42) and seed32 full cells
                for s in seeds:
                    m, e, opt, _ = s0.train(dict(kind='full', seed=s))
                    if s == 42:
                        p = base_out / 'seed42_baseline'; p.mkdir(parents=True)
                        torch.save(dict(raw=m.state_dict(), ema=e.state_dict(), optimizer=opt.state_dict()), p / 'final.pt')
                    else:
                        p = s32 / 'cells' / f'full_seed{s}'; p.mkdir(parents=True)
                        torch.save(dict(raw=m.state_dict(), ema=e.state_dict()), p / 'final.pt')
                    c['input_sha256'][str(p / 'final.pt')] = rt.sha(p / 'final.pt')
                    st = torch.load(p / 'final.pt', weights_only=False)
                    P = {}
                    for ep in ('raw', 'ema'):
                        P[ep], P['log_' + ep] = s0.predict(s0.initialize(s, st[ep], double=True))
                    d32 = s32 / 'cells' / f'full_seed{s}'; d32.mkdir(parents=True, exist_ok=True)
                    np.savez(d32 / 'interface.npz', raw_index=ctx['interface'], **P)
                # LDS v1 legacy scores at seed 42: frozen_h and CG tau with the SAME macro CE h
                md = s0.initialize(42, torch.load(base_out / 'seed42_baseline/final.pt', weights_only=False)['raw'], double=True)
                _, L = s0.predict(md)
                h = s0.query_gradient(md, 'ce', s0.macro_w, L[s0.N:])
                hvp1 = score.make_hvp(md, ctx['X'], ctx['y'], ctx['train'], pilot.DEVICE, c['weight_decay'], 1., 64)
                cg = score.conjugate_gradient(hvp1, h, 300, 1e-11)
                self.assertFalse(cg['nonpositive_curvature'])
                with (pilot_out / 'score_seed42/scores.csv').open('w', newline='') as fh:
                    w = csv.writer(fh); w.writerow(['patient', 'tau_upweight', 'frozen_h_upweight'])
                    for p in base['train_patients']:
                        b = s0.batch_gradient(md, np.asarray(p['rows']))
                        w.writerow([p['patient'], -float(cg['x'] @ b), -float(h @ b)])
                # ---- smoke
                (tmp / 'smoke').mkdir()
                s0.smoke(tmp / 'smoke')
                res = rt.read(tmp / 'smoke/result.json')
                self.assertTrue(res['noop_exact'] and res['subset_replay_exact'])
                self.assertLess(max(res['checks']['image_gradients_sum_to_patient_gradient_rel']), 1e-10)
                (root / 'smoke').mkdir()
                (root / 'smoke/designs.json').write_text((tmp / 'designs_tmp.json').read_text())
                rt.write(root / 'smoke/result.json', res)
                rt.finish(root / 'smoke', c, dict(kind='smoke'))
                (root / 'designs.json').write_text((tmp / 'designs_tmp.json').read_text())
                s = core3.Session(c)
                # ---- cells: harvest full, train image subsets (all, so collect has a complete panel)
                for cell in c['cells']:
                    self.assertEqual(s.run_cell(cell, 't'), 'done')
                self.assertEqual(s.run_cell(c['cells'][0], 't'), 'skipped')
                r = rt.read(root / 'cells' / c['cells'][-1]['name'] / 'result.json')
                self.assertEqual(r['n_train_images'], s.design['retained_images'])
                self.assertEqual(r['optimizer_steps'], 2 * int(np.ceil(s.design['retained_images'] / 128)))
                # ---- proxies, IF scores and their gates
                for seed in seeds:
                    self.assertEqual(s.proxy_seed(seed, 't'), 'done')
                pr = rt.read(root / 'proxy/seed42/result.json')['checks']
                self.assertLess(pr['patient_sum_vs_lds_v1_frozen_h_rel'], 1e-10)
                self.assertLess(max(v for k, v in pr.items() if k.endswith('jacobian_vs_direct_rel')), 1e-10)
                for seed in c['score_seeds']:
                    self.assertEqual(s.score_seed(seed, 't'), 'done')
                    self.assertEqual(s.score_seed(seed, 't'), 'skipped')
                gate = rt.read(root / 'score/seed42_check/result.json')['gate']
                self.assertTrue(gate['passed'], gate)
                self.assertLess(gate['rel_l2_difference'], 1e-6)
                sr = rt.read(root / 'score/seed45_ce/result.json')
                self.assertTrue(all(g['qualified'] for g in sr['gammas']), sr['gammas'])
                # ---- a corrupted seed32 interface must be refused on harvest
                z = dict(np.load(s32 / 'cells/full_seed45/interface.npz'))
                z['raw'] = z['raw'] + 1e-6
                np.savez(s32 / 'cells/full_seed45/interface.npz', **z)
                with self.assertRaises(ValueError):
                    s.run_cell(dict(c['cells'][1], name='full_seed45_again'), 'u')
                # ---- collect on the real artefacts
                c['packs'] = run.make_packs(c['cells'], 20.)
                with patch.object(run, 'config', return_value=c):
                    run.collect(SimpleNamespace(out=str(root), allow_partial=False))
                summary = rt.read(root / 'report/summary.json')
                self.assertEqual(summary['design']['n_images_range'][0], summary['design']['n_images_range'][1])
                self.assertIn('ce_image', summary['per_query'])
                self.assertEqual(summary['gamma']['ce']['1']['status'], 'solved')
                eq = summary['equivalence']['ce']
                self.assertAlmostEqual(eq['deleter'], eq['retained_sum'])
                self.assertAlmostEqual(eq['deleter'], eq['matched'])
                self.assertTrue((root / 'report/REPORT_CN.md').exists())


if __name__ == '__main__':
    unittest.main()
