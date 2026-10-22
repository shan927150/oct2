"""The interface LDS on CPU with a tiny synthetic OCT-like context: a v3-style run (full + subset Stage-1 models
built with the frozen stable kernel), a v6-style run computed from those models with the reused kernels
(adjoints, hbar by VJP, tau by multi-shift Lanczos, matched L3 scores, L1 truth rows), then
smoke -> solve (every seed) -> collect, with independent checks of the JVP path and of the v6 ties."""
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
    spec = importlib.util.spec_from_file_location(name + '_if', HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legacy():
    cross = rt.REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(rt.REPO / 'src')]
    spec = importlib.util.spec_from_file_location('cpu_score_if', cross / '07_cross_stage_score_ladder.py')
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
    pargs = argparse.Namespace(attack_epochs=3, attack_lr=.01, attack_batch_size=128)
    return dict(X=X, y=y, groups=groups, train=train, test=test, interface=np.r_[train, test], pargs=pargs)


class GpuPathOnCpu(unittest.TestCase):
    def test_full_pipeline(self):
        import torch
        import core3
        import solver
        from design import make_design
        from imgdesign import matched_prediction
        run = own('run')
        score, pilot = legacy()
        ctx = context()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, full, base_out, pilot_out, s32, v3, v6 = (tmp / x for x in ('exp', 'full', 'baseline', 'pilot', 'seed32', 'v3', 'v6'))
            for d in (root, full, pilot_out / 'score_seed42', v3, v6):
                d.mkdir(parents=True)
            seeds, gam, gA = [42, 45], [1., .3, .1], [1., .1]
            design = dict(kind='image_random', m=8, retain=.5, seed=3)
            for s in (42, 43, 44):
                np.savez(full / f'stage1_order_seed{s}.npz', raw_index_order=pilot.make_epoch_orders(ctx['train'], 2, s + 700000))
            rt.write(pilot_out / 'score_seed42/design.json',
                     make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929))
            # ---------------------------------------------------- v3-style run
            cv3 = dict(out=str(v3), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                       seed32_out=str(s32), epochs=2, recipe=RECIPE, batch=64, weight_decay=2., design=design,
                       design_name='img32', input_sha256={})
            rt.write(v3 / 'config.json', cv3)
            pins, models = {}, {}
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                s0 = core3.Session(cv3, freeze_to=v3 / 'designs.json')
                v3_cells = []
                for s in seeds:
                    m, e, opt, _ = s0.train(dict(kind='full', seed=s))
                    p = base_out / 'seed42_baseline' if s == 42 else s32 / 'cells' / f'full_seed{s}'
                    p.mkdir(parents=True)
                    torch.save(dict(raw=m.state_dict(), ema=e.state_dict(), optimizer=opt.state_dict()), p / 'final.pt')
                    pins[str(p / 'final.pt')] = rt.sha(p / 'final.pt')
                    models[s] = s0.initialize(s, m.state_dict(), double=True)
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
                # ---------------------------------------------------- v6-style run from the same models
                rows3, y3 = ctx['interface'], ctx['y'][ctx['interface']]
                R = len(rows3)
                K6 = [5101, 5104]
                fr6 = dict(attack_seeds=K6, attack_gamma_star=.1, omega={'1': .6, '2': .4}, qualified_classes=[1, 2])
                truth_cells = [dict(name=f'truth_s{j:03d}_seed{s}', kind='truth', subset=j, seed=s, minutes=1.)
                               for s in seeds for j in range(design['m'])]
                c6 = dict(out=str(v6), schema='OCT_cross_stage_lds_v6', git_commit=rt.V6_COMMIT, seeds=seeds, score_seeds=seeds,
                          cells=truth_cells, attack_seeds=K6, attack_gammas=gA, gammas=gam)
                rt.write(v6 / 'config.json', c6)
                h6 = rt.sha(v6 / 'config.json')
                rng = np.random.default_rng(11)
                M = s0.M
                for d, cell, files in ((v6 / 'smoke', dict(kind='xsmoke'), ('result.json', 'designs.json', 'fixed_rows.npz')),):
                    d.mkdir()
                    for f in files:
                        (d / f).write_text('{}')
                    rt.write(d / 'frozen.json', fr6)
                    rt.finish(d, c6, cell)
                V, vbars = {}, {}
                for s in seeds:
                    Vs = np.zeros((len(K6), 4, len(gA), R, 4))
                    for ci in range(4):
                        Vs[:, ci, :, y3 == ci] = rng.normal(size=(len(K6), len(gA), int((y3 == ci).sum()), 4)).transpose(2, 0, 1, 3)
                    V[s] = Vs
                    vbars[s] = sum(fr6['omega'][str(cl)] * Vs[:, cl, gA.index(.1)].mean(0) for cl in (1, 2))
                    d = v6 / 'base' / f'seed{s}'
                    d.mkdir(parents=True)
                    np.savez_compressed(d / 'v.npz', v=Vs, attack_gammas=np.asarray(gA), classes=np.arange(4), attack_seeds=np.asarray(K6))
                    np.savez(d / 'h.npz', h=np.zeros(1))
                    rt.write(d / 'result.json', {})
                    rt.finish(d, c6, dict(kind='base6', seed=s))
                    model = models[s]
                    hbar = score.shadow_h(model, ctx['X'][rows3], vbars[s], torch.device('cpu'), 64)
                    G = s0.image_gradients(model)
                    hvp = score.make_hvp(model, ctx['X'], ctx['y'], ctx['train'], torch.device('cpu'), 2., 0., 64)
                    ms = solver.multishift_lanczos(hvp, hbar, gam, 300, 1e-11, dot=lambda a, b: float(a @ b))
                    arrays = dict(full_hbar=hbar.numpy())
                    for g in gam:
                        tau = -(G @ ms['x'][g]).numpy()
                        arrays[f'full_tau_gamma{g:g}'] = tau
                        arrays[f'full_pred_gamma{g:g}'] = matched_prediction(M, tau)
                    d = v6 / 'score' / f'seed{s}'
                    d.mkdir(parents=True)
                    np.savez_compressed(d / 'scores.npz', **arrays)
                    rt.write(d / 'result.json', dict(variants=dict(full=dict(gammas=[dict(gamma=g, qualified=True, reasons=[]) for g in gam]))))
                    rt.finish(d, c6, dict(kind='score6', seed=s))
                    full_p = np.load(v3 / 'cells' / f'full_seed{s}' / 'interface.npz')['raw']
                    for j in range(design['m']):
                        dP = np.load(v3 / 'cells' / f'img32_s{j:03d}_seed{s}' / 'interface.npz')['raw'] - full_p
                        rows = []
                        for ki, k in enumerate(K6):
                            for cl in range(4):
                                sel = y3 == cl
                                rows.append(dict(attack_seed=k, cls=cl, J00=1., J10=1. + float(rng.normal() * .01),
                                                 L1={f'{g:g}': float((Vs[ki, cl, gi][sel] * dP[sel]).sum()) for gi, g in enumerate(gA)}))
                        cell = truth_cells[seeds.index(s) * design['m'] + j]
                        d = v6 / 'truth' / cell['name']
                        d.mkdir(parents=True)
                        rt.write(d / 'result.json', dict(cell=cell, rows=rows))
                        rt.finish(d, c6, dict(cell, kind='truth6'))
                for f in [v6 / 'smoke/complete.json', v6 / 'smoke/frozen.json'] + \
                        [v6 / x / f'seed{s}' / f for s in seeds for x, f in (('base', 'complete.json'), ('base', 'v.npz'),
                                                                              ('score', 'complete.json'), ('score', 'scores.npz'))] + \
                        [v6 / 'truth' / x['name'] / f for x in truth_cells for f in ('complete.json', 'result.json')]:
                    pins[str(f)] = rt.sha(f)
            # ---------------------------------------------------- the interface run
            c = dict(out=str(root), full=str(full), lds_pilot_out=str(pilot_out), baseline_out=str(base_out),
                     seed32_out=str(s32), v3_out=str(v3), v3_config_sha256=rt.sha(v3 / 'config.json'), v3_cells=v3_cells,
                     input_sha256=pins, epochs=2, recipe=RECIPE, batch=64, weight_decay=2., design=design,
                     design_name='img32', seeds=seeds, v3_prediction_tol=1e-10, harvest_tol=1e-10,
                     attack=dict(epochs=3, lr=.01, batch=128, hidden=64, deterministic=True), attack_seeds=K6,
                     classes=[0, 1, 2, 3], primary_classes=[1, 2], attack_gammas=gA, gammas=gam, primary_gamma=.3,
                     cg_fail_tol=1e-3, probe_steps=8, cells=truth_cells, adjoint_identity_rel_tol=1e-9,
                     v6_out=str(v6), v6_config_sha256=h6, tol_main=1e-8, tol_small=1e-6, max_dim=320, max_new=120,
                     pilot_rhs=2, budget_gpu_hours=1e6, solve_hours=1e6, permutations=200, bootstraps=200)
            rt.write(root / 'config.json', c)
            with patch.object(rt, 'load_context', return_value=(ctx, None, score, pilot, None)):
                import core_if
                sm = root / '.smoke.tmp-t'
                sm.mkdir()
                s1 = core_if.ISession(c, freeze_to=sm / 'designs.json')
                s1.if_smoke(sm)
                rt.finish(sm, c, dict(kind='ifsmoke'))
                sm.rename(root / 'smoke')
                run.ensure_designs(root, rt.sha(root / 'config.json'))
                res = rt.read(root / 'smoke/result.json')
                fr = rt.read(root / 'smoke/frozen.json')
                self.assertEqual(fr['score_panel'], seeds)
                self.assertLess(res['checks']['vbar_reproduces_v6_hbar_rel'], 1e-10)
                self.assertTrue(all(v['rel_l2'] < 1e-6 for v in res['checks']['tau_vs_v6'].values()), res['checks']['tau_vs_v6'])
                self.assertLess(res['checks']['forward_reverse_identity_rel'], 1e-9)
                self.assertLess(max(res['checks']['symmetry_hbar_xb_vs_b_xhbar_rel'].values()), 1e-6)
                s = core_if.ISession(c)
                for seed in seeds:
                    self.assertEqual(s.if_solve(seed, 't'), 'done')
                    self.assertEqual(s.if_solve(seed, 't'), 'skipped')
                r = rt.read(root / 'solve/seed42/result.json')
                self.assertTrue(all(g['qualified'] for g in r['gammas']), r['gammas'])
                self.assertLess(r['forward_reverse_identity_max_rel'], 1e-9)
                with np.load(root / 'solve/seed42/interface.npz') as z:
                    self.assertEqual(z['dz_if'].shape, (8, 3, R, 4))
                    dz, p0 = z['dz_if'], z['p0']
                # independent path: multi-shift Lanczos on b_0 and the legacy linearised predictions
                model = models[42]
                G = s0.image_gradients(model)
                b0 = torch.as_tensor(1. - M[0], dtype=G.dtype) @ G
                hvp = score.make_hvp(model, ctx['X'], ctx['y'], ctx['train'], torch.device('cpu'), 2., 0., 64)
                x = solver.multishift_lanczos(hvp, b0, [.3], 300, 1e-11, dot=lambda a, b: float(a @ b))['x'][.3]
                lin = score.linearized_predictions(model, ctx['X'][rows3], x, torch.device('cpu'), 64)
                ours = core_if.softmax_jvp(p0, dz[0, 1])
                ref = lin['raw'] - p0
                self.assertLess(np.abs(ours - ref).max() / np.abs(ref).max(), 1e-6)
            with patch.object(run, 'config', return_value=c):
                run.collect(SimpleNamespace(out=str(root), allow_partial=False))
            summ = rt.read(root / 'report/summary.json')
            self.assertIn('mean', summ['primary'])
            self.assertEqual(summ['design']['primary_truth_panel'], seeds)
            self.assertLess(summ['bridge']['L1_reproduces_v6_max_abs'], 1e-9)
            self.assertTrue(all(v < 1e-6 for v in summ['bridge']['v6_reproduction'].values()), summ['bridge']['v6_reproduction'])
            for name in ('IF (gamma=0.3)', 'IF (gamma=1)', 'proxy', 'true dtheta (linear)', 'IF logit (gamma=0.3)'):
                self.assertIn('nonmember', summ['per_output'][name])
            self.assertIn('L3 (gamma=0.3), deleted rows <- retrained change', summ['bridge']['rungs'])
            text = (root / 'report/REPORT_CN.md').read_text()
            for sec in ('预设主检验', '逐输出 LDS', '行组平均的概率变化', '自身删除效应', '回到 attack', '读法'):
                self.assertIn(sec, text)
            self.assertEqual(len(list(tmp.glob('exp.report.tar.gz'))), 1)
            print('CPU_BRIDGE_ERRORS', dict(hbar=res['checks']['vbar_reproduces_v6_hbar_rel'],
                  tau=max(v['rel_l2'] for v in res['checks']['tau_vs_v6'].values()),
                  forward_reverse=r['forward_reverse_identity_max_rel'],
                  L1=summ['bridge']['L1_reproduces_v6_max_abs'],
                  L3=max(summ['bridge']['v6_reproduction'].values())), flush=True)
            # A compute-limited panel must use the same Stage-1 seeds in primary truth.
            # This exercises collect-only fallbacks, with no additional training or HVPs.
            import analysis_if as AI
            data = AI.load(root, c)
            data['frozen']['score_panel'] = seeds[:1]
            data['solves'] = {seeds[0]: data['solves'][seeds[0]]}
            panel_summary, _ = AI.analyze(data, c)
            preds, _, _ = AI.predictions(data, c)
            rr = np.arange(len(ctx['train']), R)
            expected = AI.finite_mean(AI.per_output_lds(AI.flat(preds['IF (gamma=0.3)'][0], rr),
                                                       AI.flat(data['truth'][0], rr)))
            self.assertAlmostEqual(panel_summary['primary']['mean'], expected, places=12)
            self.assertIn('secondary_all_seed_truth', panel_summary)
            self.assertEqual(panel_summary['bridge']['oracle_truth_stage1_panel'], seeds[:1])
            self.assertTrue(all(e['perm_p'] is None for e in panel_summary['bridge']['rungs'].values()))
            report = own('report_if')
            report.render(run.clean(panel_summary), c, False)


if __name__ == '__main__':
    unittest.main()
