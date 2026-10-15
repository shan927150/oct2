"""v6 on CPU with a tiny synthetic OCT-like context: a v3-style run, then a v5 r1 run (stable target,
attack seeds 5101/5102) built with the reused v5 code, then v6 on top of it with one new attack seed
(5103): smoke (old-attack replay, gate) -> base (V, h, h_member) -> truth (old rows copied, J10 replayed,
new seed trained) -> score (two right-hand sides, 3-seed hbar replayed) -> collect."""
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
    spec = importlib.util.spec_from_file_location(name + '_x6', HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legacy():
    cross = rt.REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(rt.REPO / 'src')]
    spec = importlib.util.spec_from_file_location('cpu_score6', cross / '07_cross_stage_score_ladder.py')
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
    def test_v5_then_v6(self):
        import torch
        import core3
        from design import make_design
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
                     h_chain_fd_rel_tol=0., jvp_fd_max_rel_tol=0., finite_difference_policy='diagnostic_only',
                     attack=dict(epochs=3, lr=.01, batch=128, hidden=64, deterministic=True), attack_seeds=[5101, 5102],
                     classes=[0, 1, 2, 3], primary_classes=[1, 2], n_shadow=4,
                     background_recipe='stable', background_shadows=[0, 1, 2], fixed_shadow_seed=42100, primary_gamma=.3, attack_gammas=[1., .1],
                     background_jobs=[0, 1, 2, 'target'], target_recipe='stable', target_seed=42007, j00_gate=50.,
                     adjoint_identity_rel_tol=1e-9, interp=dict(subsets=[0, 1], seeds=[42], ts=[1e-3, .1, 1.]),
                     attack_solve_tol=1e-10, j00_tol=0., score_seeds=seeds, gammas=[1., .3], lanczos_max_steps=80,
                     lanczos_est_tol=1e-7, probe_steps=8, cg_fail_tol=1e-3)
            # ---- a v5 r1 run (stable target), produced with the reused v5 code
            c5 = c
            c5.update(schema='OCT_cross_stage_lds_v5r1', git_commit=rt.V5_COMMIT, h_chain_fd_rel_tol=0., jvp_fd_max_rel_tol=0.,
                      finite_difference_policy='diagnostic_only', attack=dict(epochs=3, lr=.01, batch=128, hidden=64, deterministic=True),
                      attack_seeds=[5101, 5102], classes=[0, 1, 2, 3], primary_classes=[1, 2], n_shadow=4,
                      background_recipe='stable', background_shadows=[0, 1, 2], fixed_shadow_seed=42100, primary_gamma=.3,
                      attack_gammas=[1., .1], background_jobs=[0, 1, 2, 'target'], target_recipe='stable', target_seed=42007,
                      j00_gate=50., adjoint_identity_rel_tol=1e-9, attack_solve_tol=1e-10, j00_tol=0., score_seeds=seeds,
                      gammas=[1., .3], lanczos_max_steps=80, lanczos_est_tol=1e-7, probe_steps=8, cg_fail_tol=1e-3)
            c5['cells'] = [dict(name=f'truth_s{j:03d}_seed{s}', kind='truth', subset=j, seed=s, minutes=2.4)
                           for s in seeds for j in range(design['m'])]
            rt.write(root / 'config.json', c5)
            (root / 'config.sha256').write_text(rt.sha(root / 'config.json') + '\n')
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                import core5
                sm = root / '.smoke.tmp-t'
                sm.mkdir()
                s1 = core5.X5Session(c5, freeze_to=sm / 'designs.json')
                for sid in c5['background_shadows']:
                    s1.background_shadow(sid, 't')
                s1.background_target('t')
                s1.xsmoke(sm)
                rt.finish(sm, c5, dict(kind='xsmoke'))
                sm.rename(root / 'smoke')
                (root / 'designs.json').write_bytes((root / 'smoke/designs.json').read_bytes())
                s5 = core5.X5Session(c5)
                for seed in seeds:
                    s5.base_seed(seed, 't')
                for cell in c5['cells']:
                    s5.truth_cell(cell, 't')
                for seed in seeds:
                    s5.xscore_seed(seed, 't')
            (root / 'report').mkdir()
            rt.write(root / 'report/complete.json', dict(status='complete', config_sha256=rt.sha(root / 'config.json'), missing=[]))
            # ---- v6 on top of it
            run6 = own('run')
            cfg5, h5, pins5 = run6.check_v5(root)
            root6 = tmp / 'exp6'
            c6 = {k: c5[k] for k in run6.COPY if k in c5}
            c6.update(out=str(root6), schema='OCT_cross_stage_lds_v6', v5_out=str(root), v5_config_sha256=h5,
                      input_sha256=dict(c5['input_sha256'], **pins5), old_attack_seeds=[5101, 5102], new_attack_seeds=[5103],
                      attack_seeds=[5101, 5102, 5103], cpu_account='cpu', cpu_partition='cpu', partition='p', fp64_partition='p')
            c6['packs'] = run6.make_packs(c6['cells'])
            root6.mkdir()
            rt.write(root6 / 'config.json', c6)
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                import core6
                sm = root6 / '.smoke.tmp-t'
                sm.mkdir()
                s6 = core6.X6Session(c6, freeze_to=sm / 'designs.json')
                s6.smoke6(sm)
                rt.finish(sm, c6, dict(kind='xsmoke'))
                sm.rename(root6 / 'smoke')
                run6.ensure_designs(root6, rt.sha(root6 / 'config.json'))
                g = rt.read(root6 / 'smoke/gate.json')
                self.assertTrue(g['passed'])
                self.assertTrue(g['gamma_star_qualified_on_new_seeds'])
                self.assertEqual(max(g['old_attack_replay_max_abs'].values()), 0.)
                fr = rt.read(root6 / 'smoke/frozen.json')
                self.assertEqual(fr['attack_seeds'], [5101, 5102, 5103])
                self.assertEqual(fr['attack_gamma_star'], rt.read(root / 'smoke/frozen.json')['attack_gamma_star'])
                s = core6.X6Session(c6)
                real_condition = s.condition
                def drifted(*args, **kwargs):
                    out, model = real_condition(*args, **kwargs)
                    out['metrics']['cross_entropy'] += 1.
                    return out, model
                with patch.object(s, 'condition', side_effect=drifted):
                    with self.assertRaisesRegex(ValueError, 'J00 vs smoke'):
                        s.base6(seeds[0], 'drift')
                self.assertFalse((root6 / 'base' / f'seed{seeds[0]}/complete.json').exists())
                for seed in seeds:
                    self.assertEqual(s.base6(seed, 't'), 'done')
                    self.assertEqual(s.base6(seed, 't'), 'skipped')
                b = rt.read(root6 / 'base/seed45/result.json')
                self.assertLess(b['h_replay_max_abs'], 1e-10)
                self.assertTrue(all(r['J00_minus_smoke'] == 0. for r in b['rows']))
                with np.load(root6 / 'base/seed45/h.npz') as z, np.load(root / 'base/seed45/h.npz') as z5:
                    self.assertEqual(z['h'].shape[:2], (3, 4))
                    np.testing.assert_array_equal(z['h'][:2], z5['h'])
                    self.assertFalse(np.allclose(z['h_member'], z['h']))
                # member-only h equals h computed on the member rows alone (independent check)
                fixed = s.fixed()
                V, H, Hm, _ = s.base6_arrays(45)
                mem = fixed['membership'][fixed['shadow_id'] == 3] == 1
                model = s.model(s.full_name(45), 45)
                rows3 = s.rows3
                gi = c6['attack_gammas'].index(fr['attack_gamma_star'])
                hm = s.score.shadow_h(model, ctx['X'][rows3[mem]], V[2, 1, gi][mem] * (fixed['classes'][fixed['shadow_id'] == 3][mem] == 1)[:, None],
                                      s.dev, 64).cpu().numpy()
                np.testing.assert_allclose(Hm[2, 1], hm, rtol=1e-10, atol=1e-14)
                for cell in c6['cells']:
                    self.assertEqual(s.truth6(cell, 't'), 'done')
                self.assertEqual(s.truth6(c6['cells'][0], 't'), 'skipped')
                tr6 = rt.read(root6 / 'truth' / c6['cells'][3]['name'] / 'result.json')
                tr5 = {(r['attack_seed'], r['cls']): r for r in rt.read(root / 'truth' / c6['cells'][3]['name'] / 'result.json')['rows']}
                self.assertEqual(tr6['j10_replay_abs'], 0.)
                self.assertEqual(len(tr6['rows']), 3 * 4)
                for r in tr6['rows']:
                    if r['source'] == 'v5r1':
                        self.assertEqual(r['J10'], tr5[(r['attack_seed'], r['cls'])]['J10'])
                        self.assertLess(r['L1_check_abs'], 1e-12)
                        self.assertLess(r['L2_lin_check_abs'], 1e-10)
                    for gk in ('1', '0.1'):
                        self.assertNotEqual(r['L1_member'][gk], r['L1'][gk])
                for seed in seeds:
                    self.assertEqual(s.score6(seed, 't'), 'done')
                sr = rt.read(root6 / 'score/seed45/result.json')
                self.assertLess(sr['hbar_replay_vs_v5_max_abs'], 1e-12)
                self.assertTrue(all(q['qualified'] for v in sr['variants'].values() for q in v['gammas']))
                self.assertLess(sr['cos_hbar_full_member'], 1.)
            with patch.object(run6, 'config', return_value=c6):
                run6.collect(SimpleNamespace(out=str(root6), allow_partial=False))
            summ = rt.read(root6 / 'report/summary.json')
            self.assertEqual(set(summ['primary']), {'P1', 'P2', 'P3'})
            for name in ('L1 (full)', 'L1 (member)', 'L2-lin (full)', 'L2-lin (member)', 'L3 full (gamma=0.3)',
                         'L3 member (gamma=0.3)', 'proxy (full)', 'proxy (member)'):
                self.assertIn(name, summ['ladder'])
            self.assertEqual(summ['replay']['j10_replay'], 0.)
            self.assertEqual(summ['replay']['new_j00_vs_smoke'], 0.)
            self.assertIn('frozen_v5_on_new_attack_seeds', summ)
            # the old-seed truth must equal the v5 r1 truth
            ar = __import__('analysis6').arrays(__import__('analysis6').load(root6, c6), c6)
            self.assertEqual(ar['D'].shape[2], 3)
            rep = (root6 / 'report/REPORT_CN.md').read_text()
            for sec in ('预设主检验', '门与复现', '读法'):
                self.assertIn(sec, rep)
            self.assertEqual(len(list(tmp.glob('exp6.report.tar.gz'))), 1)


if __name__ == '__main__':
    unittest.main()
