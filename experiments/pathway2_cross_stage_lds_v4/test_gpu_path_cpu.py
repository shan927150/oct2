"""Runs the real GPU-side code path on CPU with a tiny synthetic OCT-like context: a v3-style run
(full + subset Stage-1 models built with the frozen stable kernel), frozen legacy shadows/target
saved in the legacy checkpoint format, then smoke -> base -> truth -> score -> collect."""
import argparse
import importlib.util
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
HERE = Path(__file__).resolve().parent


def own(name):
    spec = importlib.util.spec_from_file_location(name + '_x4', HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legacy():
    cross = rt.REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(rt.REPO / 'src')]
    spec = importlib.util.spec_from_file_location('cpu_score4', cross / '07_cross_stage_score_ladder.py')
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score, score.import_pilot_module()


def context(seed=0):
    rng = np.random.default_rng(seed)
    tr, q = rng.integers(2, 6, 12), rng.integers(1, 4, 8)
    groups = np.r_[np.repeat(100 + np.arange(12), tr), np.repeat(900 + np.arange(8), q)]
    y = np.r_[np.repeat(np.arange(12) % 4, tr), np.repeat(np.arange(8) % 4, q)]
    n3 = len(y)
    extra = 4 * 2 * 24                       # shadows 0-2 and the target: 24 train + 24 test rows each
    y = np.r_[y, np.tile(np.arange(4), extra // 4)]
    groups = np.r_[groups, 5000 + np.arange(extra)]
    X = (rng.normal(size=(len(y), 1, 16, 16)) + y[:, None, None, None] * .3).astype(np.float32)
    train, test = np.arange(tr.sum()), np.arange(tr.sum(), n3)
    pargs = argparse.Namespace(attack_epochs=3, attack_lr=.01, attack_batch_size=128)
    return dict(X=X, y=y, groups=groups, train=train, test=test, interface=np.r_[train, test], pargs=pargs, n3=n3)


class GpuPathOnCpu(unittest.TestCase):
    def test_full_pipeline(self):
        import torch
        import core3
        from design import make_design
        run = own('run')
        score, pilot = legacy()
        ctx = context()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, full, base_out, pilot_out, s32, v3 = (tmp / x for x in ('exp', 'full', 'baseline', 'pilot', 'seed32', 'v3'))
            for d in (root, full / 'splits', full / 'checkpoints', pilot_out / 'score_seed42', v3):
                d.mkdir(parents=True)
            seeds = [42, 45]
            design = dict(kind='image_random', m=8, retain=.5, seed=3)
            # ---- frozen legacy shadows 0-2 and target, legacy split and checkpoint format
            off = ctx['n3']
            split = dict(shadow_models=[], target_train_idx=None, target_test_idx=None)
            for sid in range(4):
                if sid == 3:
                    split['shadow_models'].append(dict(train_idx=ctx['train'].tolist(), test_idx=ctx['test'].tolist()))
                    continue
                split['shadow_models'].append(dict(train_idx=list(range(off, off + 24)), test_idx=list(range(off + 24, off + 48))))
                off += 48
            split['target_train_idx'], split['target_test_idx'] = list(range(off, off + 24)), list(range(off + 24, off + 48))
            rt.write(full / 'splits/fresh_patient_split.json', split)
            torch.manual_seed(0)
            for name in ('shadow_0_fixed', 'shadow_1_fixed', 'shadow_2_fixed', 'target_fixed'):
                pilot.save_model(full / 'checkpoints' / f'{name}.pt', pilot.build_model('cnn', 1, 128, 4), {})
            for s in (42, 43, 44):
                np.savez(full / f'stage1_order_seed{s}.npz', raw_index_order=pilot.make_epoch_orders(ctx['train'], 2, s + 700000))
            rt.write(pilot_out / 'score_seed42/design.json',
                     make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929))
            # ---- a v3-style run built with the frozen stable kernel
            cv3 = dict(out=str(v3), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                       seed32_out=str(s32), epochs=2, recipe=RECIPE, batch=64, weight_decay=2., design=design,
                       design_name='img32', input_sha256={})
            rt.write(v3 / 'config.json', cv3)
            pins = {}
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                s0 = core3.Session(cv3, freeze_to=v3 / 'designs.json')
                v3_cells = []
                for s in seeds:
                    m, e, opt, _ = s0.train(dict(kind='full', seed=s))
                    p = base_out / 'seed42_baseline' if s == 42 else s32 / 'cells' / f'full_seed{s}'
                    p.mkdir(parents=True)
                    torch.save(dict(raw=m.state_dict(), ema=e.state_dict(), optimizer=opt.state_dict()), p / 'final.pt')
                    pins[str(p / 'final.pt')] = rt.sha(p / 'final.pt')
                    cells = [(dict(name=f'full_seed{s}', group='full', kind='full', seed=s, source='harvest'), m)]
                    for j in range(design['m']):
                        mj, ej, _, _ = s0.train(dict(kind='subset', subset=j, seed=s))
                        cells.append((dict(name=f'img32_s{j:03d}_seed{s}', group='img32', kind='subset', subset=j,
                                           seed=s, source='train'), (mj, ej)))
                    for cell, mod in cells:
                        d = v3 / 'cells' / cell['name']
                        d.mkdir(parents=True)
                        if cell['kind'] == 'subset':
                            torch.save(dict(raw=mod[0].state_dict(), ema=mod[1].state_dict(), cell=cell), d / 'final.pt')
                            mod = mod[0]
                        P, _ = s0.predict(s0.initialize(s, mod.state_dict(), double=True))
                        np.savez(d / 'interface.npz', raw=P, raw_index=ctx['interface'])
                        rt.write(d / 'result.json', dict(cell=cell))
                        rt.finish(d, cv3, cell)
                        pins[str(d / 'complete.json')] = rt.sha(d / 'complete.json')
                        v3_cells.append(cell)
            c = dict(out=str(root), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                     seed32_out=str(s32), v3_out=str(v3), v3_config_sha256=rt.sha(v3 / 'config.json'), v3_cells=v3_cells,
                     input_sha256=pins, epochs=2, recipe=RECIPE, batch=64, weight_decay=2., design=design,
                     design_name='img32', seeds=seeds, v3_prediction_tol=1e-10, harvest_tol=1e-10,
                     h_chain_fd_rel_tol=5e-3, jvp_fd_max_rel_tol=2e-3,
                     attack=dict(epochs=3, lr=.01, batch=128, hidden=64, deterministic=True), attack_seeds=[5101, 5102],
                     classes=[0, 1, 2, 3], primary_classes=[1, 2], n_shadow=4,
                     background_recipe='stable', background_shadows=[0, 1, 2], fixed_shadow_seed=42100, primary_gamma=.3, attack_gammas=[1., .1],
                     attack_solve_tol=1e-10, j00_tol=0., score_seeds=seeds, gammas=[1., .3], lanczos_max_steps=80,
                     lanczos_est_tol=1e-7, probe_steps=8, cg_fail_tol=1e-3)
            c['cells'] = run.make_cells(c)
            c['packs'] = run.make_packs(c['cells'], 20.)
            rt.write(root / 'config.json', c)
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                import core4
                # ---- smoke
                sm = root / '.smoke.tmp-t'
                sm.mkdir()
                s1 = core4.XSession(c, freeze_to=sm / 'designs.json')
                for sid in c['background_shadows']:
                    self.assertEqual(s1.background_shadow(sid, 't'), 'done')
                s1.xsmoke(sm)
                rt.finish(sm, c, dict(kind='xsmoke'))
                sm.rename(root / 'smoke')
                run.ensure_designs(root, rt.sha(root / 'config.json'))
                res = rt.read(root / 'smoke/result.json')
                fr = rt.read(root / 'smoke/frozen.json')
                self.assertLess(res['checks']['adjoint_vs_legacy_rel'], 1e-10)
                self.assertLessEqual(res['checks']['jvp_fd_best_relative_error'], c['jvp_fd_max_rel_tol'])
                self.assertLessEqual(max(res['checks']['full_prediction_vs_v3_max_abs'].values()), 1e-12)
                self.assertEqual(fr['qualified_classes'], [1, 2])
                self.assertAlmostEqual(sum(fr['omega'].values()), 1.)
                self.assertIn(fr['attack_gamma_star'], [1., .1])
                s = core4.XSession(c)
                # ---- base, truth, score
                for seed in seeds:
                    self.assertEqual(s.base_seed(seed, 't'), 'done')
                    self.assertEqual(s.base_seed(seed, 't'), 'skipped')
                with np.load(root / 'base/seed42/h.npz') as z:
                    self.assertEqual(z['h'].shape[:2], (2, 4))
                for cell in c['cells']:
                    self.assertEqual(s.truth_cell(cell, 't'), 'done')
                self.assertEqual(s.truth_cell(c['cells'][0], 't'), 'skipped')
                tr = rt.read(root / 'truth' / c['cells'][3]['name'] / 'result.json')
                self.assertEqual(len(tr['rows']), 8)
                self.assertTrue(all(abs(r['J00'] - tr['rows'][0]['J00']) >= 0 for r in tr['rows']))
                # L2-lin is h . dtheta with h from base: recompute one entry independently
                cell = c['cells'][3]
                V, H, _ = s.base_arrays(cell['seed'])
                th0 = s.flat_params(s.model(s.full_name(cell['seed']), cell['seed']))
                th1 = s.flat_params(s.model(s.subset_name(cell['subset'], cell['seed']), cell['seed']))
                self.assertAlmostEqual(tr['rows'][0]['L2_lin'], float(H[0, 0] @ (th1 - th0).cpu().numpy()), places=10)
                for seed in seeds:
                    self.assertEqual(s.xscore_seed(seed, 't'), 'done')
                sr = rt.read(root / 'score/seed45/result.json')
                self.assertTrue(all(g['qualified'] for g in sr['gammas']), sr['gammas'])
                # ---- a tampered v3 subset model must be refused
                bad = v3 / 'cells' / 'img32_s000_seed45' / 'final.pt'
                st = torch.load(bad, weights_only=False)
                next(iter(st['raw'].values())).add_(1e-3)
                torch.save(st, bad)
                with self.assertRaises(ValueError):
                    s.model('img32_s000_seed45', 45)
            # ---- collect on the produced artefacts (tamper does not touch completed results)
            with patch.object(run, 'config', return_value=c):
                run.collect(SimpleNamespace(out=str(root), allow_partial=False))
            summary = rt.read(root / 'report/summary.json')
            for name in ('L1 (gamma_A=%g)' % fr['attack_gamma_star'], 'L2-retrain', 'L2-lin', 'L3 (gamma=1)', 'proxy'):
                self.assertIn(name, summary['ladder'])
                self.assertIsNotNone(summary['ladder'][name]['spearman'])
            self.assertEqual(summary['ladder_gamma_star'], .3)
            self.assertTrue((root / 'report/REPORT_CN.md').exists())
            self.assertTrue((root / 'report/fig1_ladder.png').exists())
            tars = list(tmp.glob('exp.report.tar.gz'))
            self.assertEqual(len(tars), 1)


if __name__ == '__main__':
    unittest.main()
