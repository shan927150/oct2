"""GPU work for the image-level LDS (mentor 2026-10-09): training/harvest cells,
per-image gradients, per-query-image proxies and the multi-shift IF.

Reuses frozen kernels only (stable_core.train, legacy make_hvp/shadow_h,
seed32 solver/targets/formula_tests); edits none of them.

Score convention (unchanged): g_i = grad CE_i(theta) / N, tau_i = -h^T (H + gamma I)^-1 g_i
is the UPWEIGHT effect, so deleting image i contributes -tau_i. Summing g_i over a
patient's images gives the patient vector b_p of LDS v1, which is used as a gate.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from functools import wraps
import numpy as np
import runtime as rt
import imgdesign as ID
import qtargets as Q

STABLE_SEEDS = (42, 43, 44)  # full baselines live in the stable-training run


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def roundtrip(x):
    return json.loads(json.dumps(x, allow_nan=False))


def exclusive(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        key = args[0]['name'] if method.__name__ == 'run_cell' else str(args[0])  # never the tag
        with rt.lock(self.root / '.locks' / f'{method.__name__}_{key}.lock'):
            return method(self, *args, **kwargs)
    return wrapped


def staged(root, rel, tag):
    final = root / rel
    tmp = final.parent / f'.{final.name}.tmp-{tag}'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    return final, tmp


class Session:
    def __init__(self, c, freeze_to=None):
        from design import make_design  # LDS v1, unchanged
        import targets as T32           # seed32, unchanged (query_units)
        self.c, self.root = c, Path(c['out'])
        self.h = rt.sha(self.root / 'config.json')
        self.ctx, self.legacy, self.score, self.pilot, self.conv = rt.load_context(c)
        self.stable = module('img3_stable_core', rt.STABLE_V1 / 'stable_core.py')
        import stage1_diagnostic_core as dc
        self.dc = dc
        ctx = self.ctx
        self.base = make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929)
        if roundtrip(self.base) != rt.read(Path(c['lds_pilot_out']) / 'score_seed42/design.json'):
            raise ValueError('Regenerated patient design differs from LDS v1 score_seed42/design.json')
        d = c['design']
        self.design = roundtrip(ID.image_design(len(ctx['train']), d['m'], d['retain'], d['seed']))
        frozen = dict(base=self.base, design=self.design)
        if freeze_to is not None:
            rt.write(freeze_to, frozen)
        elif rt.read(self.root / 'designs.json') != roundtrip(frozen):
            raise ValueError('Designs differ from the smoke-frozen designs.json')
        self.N = len(ctx['train'])
        self.M = np.asarray(self.design['masks'], dtype=float)
        self.train_groups = ctx['groups'][ctx['train']]
        self.train_labels = ctx['y'][ctx['train']]
        self.labels_test = ctx['y'][ctx['test']]
        self.macro_w = np.asarray(self.base['query_weights'])
        self.img_w = Q.image_weights(len(ctx['test']))
        self.qids, self.units, _ = T32.query_units(self.base)
        self.patient_ids = [p['patient'] for p in self.base['train_patients']]
        self.A = ID.aggregation(self.train_groups, self.patient_ids)
        # Patient design rows must coincide with the image-to-patient map.
        pos = {int(r): i for i, r in enumerate(ctx['train'])}
        for p, row in zip(self.base['train_patients'], self.A):
            if sorted(pos[int(r)] for r in p['rows']) != np.flatnonzero(row).tolist():
                raise ValueError('Patient rows disagree with the image aggregation map')

    # ------------------------------------------------------------ model utils
    def initialize(self, seed, state=None, double=False):
        import torch
        self.pilot.seed_everything(seed, True)
        m = self.pilot.build_model('cnn', self.ctx['X'].shape[1], 128, int(self.ctx['y'].max()) + 1)
        for layer in m.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.
        if state is not None:
            m.load_state_dict(state)
        return m.double().eval() if double else m

    def predict(self, model, batch=64):
        import torch
        model.eval()
        p = next(model.parameters())
        rows, X, out, logs = self.ctx['interface'], self.ctx['X'], [], []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                z = model(torch.as_tensor(X[rows[s:s + batch]], device=p.device, dtype=p.dtype))
                out.append(z.softmax(1).cpu().numpy())
                logs.append(z.log_softmax(1).cpu().numpy())
        return np.concatenate(out), np.concatenate(logs)

    def values(self, log_probs):
        return Q.values(log_probs[self.N:], self.labels_test, self.macro_w, self.units)

    def orders(self, seed, epochs=None):
        o = self.pilot.make_epoch_orders(self.ctx['train'], epochs or self.c['epochs'], seed + 700000)
        if seed in STABLE_SEEDS and epochs is None:
            saved = np.load(Path(self.c['full']) / f'stage1_order_seed{seed}.npz')['raw_index_order']
            if not np.array_equal(o[:len(saved)], saved[:len(o)]):
                raise ValueError('Baseline order prefix mismatch')
        return o

    def removed_rows(self, subset):
        return self.ctx['train'][self.M[subset] == 0]

    def train(self, cell, epochs=None):
        orders = self.orders(cell['seed'], epochs)
        if cell['kind'] == 'subset':
            removed = self.removed_rows(cell['subset'])
            orders = [o[~np.isin(o, removed)] for o in orders]
        return self.stable.train(self.initialize(cell['seed']), self.ctx['X'], self.ctx['y'], orders,
                                 np.array([], dtype=np.int64), 1., **self.c['recipe'])

    def baseline_state(self, seed):
        import torch
        if seed in STABLE_SEEDS:
            path = Path(self.c['baseline_out']) / f'seed{seed}_baseline/final.pt'
        else:
            path = Path(self.c['seed32_out']) / 'cells' / f'full_seed{seed}' / 'final.pt'
        rt.verify_hashes({str(path): self.c['input_sha256'][str(path)]})
        return torch.load(path, map_location='cpu', weights_only=False)

    # ---------------------------------------------------------- derivatives
    def flat(self, grads, params):
        return self.score.flatten([g if g is not None else p.new_zeros(p.shape) for g, p in zip(grads, params)])

    def image_gradients(self, model, positions=None):
        """G[k] = grad CE(x_i, y_i) / N for training positions (default: all), fp64, on device."""
        import torch
        params = list(model.parameters())
        dev, dt = params[0].device, params[0].dtype
        positions = np.arange(self.N) if positions is None else np.asarray(positions)
        G = torch.empty((len(positions), sum(p.numel() for p in params)), dtype=dt, device=dev)
        for k, i in enumerate(positions):
            r = self.ctx['train'][i]
            x = torch.as_tensor(self.ctx['X'][[r]], dtype=dt, device=dev)
            y = torch.as_tensor(self.ctx['y'][[r]], dtype=torch.long, device=dev)
            loss = torch.nn.functional.cross_entropy(model(x), y, reduction='sum') / self.N
            G[k] = self.flat(torch.autograd.grad(loss, params), params)
        if not torch.isfinite(G).all():
            raise FloatingPointError('Non-finite per-image gradients')
        return G

    def batch_gradient(self, model, rows):
        """Batched sum_i grad CE_i / N over raw rows (independent path for the per-image gate)."""
        import torch
        params = list(model.parameters())
        dev, dt = params[0].device, params[0].dtype
        out = torch.zeros(sum(p.numel() for p in params), dtype=dt, device=dev)
        for s in range(0, len(rows), 64):
            idx = rows[s:s + 64]
            x = torch.as_tensor(self.ctx['X'][idx], dtype=dt, device=dev)
            y = torch.as_tensor(self.ctx['y'][idx], dtype=torch.long, device=dev)
            loss = torch.nn.functional.cross_entropy(model(x), y, reduction='sum') / self.N
            out += self.flat(torch.autograd.grad(loss, params), params)
        return out

    def query_jacobian_products(self, model, G, rows=None, chunk=32):
        """GJ[i, q, c] = g_i . d logit_c(x_q) / d theta for query rows (default: all heldout)."""
        import torch
        params = list(model.parameters())
        dev, dt = params[0].device, params[0].dtype
        rows = self.ctx['test'] if rows is None else np.asarray(rows)
        C = int(self.ctx['y'].max()) + 1
        out = np.empty((G.shape[0], len(rows), C))
        for s in range(0, len(rows), chunk):
            J = []
            for r in rows[s:s + chunk]:
                z = model(torch.as_tensor(self.ctx['X'][[r]], dtype=dt, device=dev))[0]
                for c in range(C):
                    J.append(self.flat(torch.autograd.grad(z[c], params, retain_graph=c < C - 1), params))
            GJ = G @ torch.stack(J).T
            out[:, s:s + len(J) // C, :] = GJ.reshape(G.shape[0], -1, C).cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError('Non-finite query Jacobian products')
        return out

    def query_gradient(self, model, target, weights, log_probs_test):
        """h = grad_theta sum_q w_q loss_q, via sum_q w_q J_q^T u_q in batches (fp64)."""
        import torch
        params = list(model.parameters())
        dev, dt = params[0].device, params[0].dtype
        u = Q.logit_derivative(target, log_probs_test, self.labels_test) * np.asarray(weights)[:, None]
        test = self.ctx['test']
        out = torch.zeros(sum(p.numel() for p in params), dtype=dt, device=dev)
        for s in range(0, len(test), 64):
            z = model(torch.as_tensor(self.ctx['X'][test[s:s + 64]], dtype=dt, device=dev))
            ub = torch.as_tensor(u[s:s + 64], dtype=dt, device=dev)
            out += self.flat(torch.autograd.grad((z * ub).sum(), params), params)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite query gradient')
        return out

    def old_scores(self, seed, key):
        import csv
        with (Path(self.c['lds_pilot_out']) / f'score_seed{seed}/scores.csv').open() as f:
            rows = list(csv.DictReader(f))
        if [int(r['patient']) for r in rows] != self.patient_ids:
            raise ValueError('Legacy score patient order differs')
        return np.asarray([float(r[key]) for r in rows])

    def predictions(self, tau_img):
        """Deleter-sum predictions for every subset from one image-score vector."""
        return ID.deleter_prediction(self.M, tau_img)

    # ------------------------------------------------------------------ cells
    @exclusive
    def run_cell(self, cell, tag):
        import torch
        final, tmp = staged(self.root, Path('cells') / cell['name'], tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, cell)
            return 'skipped'
        start, steps, harvest = time.monotonic(), None, None
        if cell['source'] == 'harvest':
            state = self.baseline_state(cell['seed'])
            raw, ema = state['raw'], state['ema']
        else:
            m, e, opt, curve = self.train(cell)
            raw = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            ema = {k: v.detach().cpu() for k, v in e.state_dict().items()}
            torch.save(dict(raw=raw, ema=ema, cell=cell, recipe=self.c['recipe']), tmp / 'final.pt')
            rt.csv_write(tmp / 'curve.csv', curve)
            steps = curve[-1]['step']
            del m, e, opt
        P, L = {}, {}
        for ep, st in (('raw', raw), ('ema', ema)):
            P[ep], L[ep] = self.predict(self.initialize(cell['seed'], st, double=True))
        if cell['source'] == 'harvest':
            src = Path(self.c['seed32_out']) / 'cells' / f"full_seed{cell['seed']}" / 'interface.npz'
            with np.load(src) as z:
                if not np.array_equal(z['raw_index'], self.ctx['interface']):
                    raise ValueError('Harvest interface rows differ')
                diffs = {k: float(np.abs(z[k] - arr).max()) for k, arr in
                         (('raw', P['raw']), ('ema', P['ema']), ('log_raw', L['raw']), ('log_ema', L['ema']))}
            if max(diffs.values()) > self.c['harvest_tol']:
                raise ValueError(f'Harvested full baseline differs from the seed32 cell: {diffs}')
            harvest = dict(source=str(src), max_abs_difference=diffs, tolerance=self.c['harvest_tol'])
        np.savez_compressed(tmp / 'interface.npz', raw=P['raw'], ema=P['ema'], log_raw=L['raw'], log_ema=L['ema'],
                            raw_index=self.ctx['interface'], classes=self.ctx['y'][self.ctx['interface']])
        n_images = self.N if cell['kind'] == 'full' else int(self.M[cell['subset']].sum())
        expected = self.c['epochs'] * int(np.ceil(n_images / self.c['recipe']['batch']))
        if steps is not None and steps != expected:
            raise ValueError('Unexpected optimizer step count')
        Y = {ep: {t: dict(img=v['Y_img'], macro=v['Y_macro']) for t, v in self.values(L[ep]).items()} for ep in L}
        rt.write(tmp / 'result.json', dict(cell=cell, Y=Y, n_train_images=n_images, optimizer_steps=expected,
                                           harvest_check=harvest))
        rt.finish(tmp, self.c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    # ------------------------------------------------------------------ smoke
    def smoke(self, dest):
        import torch
        from formula_tests import torch_solver_checks  # seed32, unchanged
        dev = self.pilot.DEVICE
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        timing, checks = {}, dict(solver=torch_solver_checks(device=dev))
        # 1) full-horizon no-op against the frozen stable baseline
        t0 = time.monotonic()
        m, e, opt, _ = self.train(dict(kind='full', seed=42))
        timing['full_training_seconds'] = time.monotonic() - t0
        base = self.baseline_state(42)
        for a, b in [(m.state_dict(), base['raw']), (e.state_dict(), base['ema']), (opt.state_dict(), base['optimizer'])]:
            if not self.dc.exact_tree(a, b):
                raise ValueError('Full-horizon no-op differs from frozen stable baseline')
        del m, e, opt
        # 2) two-epoch double replay of an image subset (filtered orders, sequential determinism)
        cell = dict(kind='subset', subset=0, seed=42)
        r1, r2 = self.train(cell, epochs=2), self.train(cell, epochs=2)
        if not all(self.dc.exact_tree(a.state_dict(), b.state_dict()) for a, b in zip(r1[:3], r2[:3])):
            raise ValueError('Image-subset replay is not deterministic')
        if r1[3][-1]['step'] != 2 * int(np.ceil(self.design['retained_images'] / self.c['recipe']['batch'])):
            raise ValueError('Image-subset step count differs from the design')
        del r1, r2
        # 3) derivative gates at the seed-42 raw baseline
        model = self.initialize(42, base['raw'], double=True)
        _, L = self.predict(model)
        pick = [0, len(self.patient_ids) // 2, len(self.patient_ids) - 1]
        pos = np.flatnonzero(self.A[pick].sum(0))
        sync(); t0 = time.monotonic()
        Gp = self.image_gradients(model, pos)
        sync(); timing['per_image_gradient_seconds'] = (time.monotonic() - t0) / len(pos)
        rel = []
        for p in pick:
            sel = np.flatnonzero(self.A[p][pos])
            direct = self.batch_gradient(model, self.ctx['train'][pos[sel]])
            rel.append(float((Gp[sel].sum(0) - direct).norm() / direct.norm()))
        checks['image_gradients_sum_to_patient_gradient_rel'] = rel
        if max(rel) > 1e-9:
            raise ValueError(f'Per-image gradients do not sum to patient gradients: {rel}')
        qrows = self.ctx['test'][:6]
        sync(); t0 = time.monotonic()
        GJ = self.query_jacobian_products(model, Gp, qrows, chunk=3)
        sync(); timing['query_jacobian_seconds_per_image'] = (time.monotonic() - t0) / len(qrows)
        jac = {}
        for t in Q.TARGETS:
            u = Q.logit_derivative(t, L[self.N:self.N + len(qrows)], self.labels_test[:len(qrows)])
            via = -(GJ * u[None]).sum(2)  # [images, queries]
            for k, r in enumerate(qrows):
                w = np.zeros(len(self.labels_test)); w[k] = 1.
                h = self.query_gradient(model, t, w, L[self.N:])
                direct = -(Gp @ h).cpu().numpy()
                jac.setdefault(t, []).append(float(np.linalg.norm(via[:, k] - direct) / max(np.linalg.norm(direct), 1e-300)))
        checks['query_jacobian_vs_direct_rel'] = jac
        if max(max(v) for v in jac.values()) > 1e-9:
            raise ValueError(f'Query Jacobian contraction differs from direct gradients: {jac}')
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], dev,
                                  self.c['weight_decay'], 0., self.c['batch'])
        v = torch.randn(Gp.shape[1], dtype=torch.float64, device=dev)
        sync(); t0 = time.monotonic(); hvp(v); sync()
        timing['fp64_hvp_seconds_on_smoke_gpu'] = time.monotonic() - t0
        timing['note'] = ('Smoke runs on the training partition; scale HVP time by the fp64 throughput ratio '
                          'for the score partition.')
        rt.write(dest / 'result.json', dict(checks=checks, timing=timing, noop_exact=True, subset_replay_exact=True,
                                            design=dict(m=len(self.design['masks']),
                                                        retained_images=self.design['retained_images'])))

    # ----------------------------------------------------------------- proxy
    @exclusive
    def proxy_seed(self, seed, tag):
        """Image-level gradient-inner-product scores for every target and weighting,
        their deleter predictions per subset, and per-query (image and patient) predictions."""
        final, tmp = staged(self.root, Path('proxy') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, dict(kind='proxy', seed=seed))
            return 'skipped'
        start = time.monotonic()
        model = self.initialize(seed, self.baseline_state(seed)['raw'], double=True)
        P, L = self.predict(model)
        cell = self.root / 'cells' / f'full_seed{seed}'
        rt.completed(cell, self.h)
        with np.load(cell / 'interface.npz') as z:
            diff = float(np.abs(z['log_raw'] - L).max())
        if diff > self.c['cross_gpu_tol']:
            raise ValueError(f'Proxy model differs from the full cell predictions ({diff})')
        G = self.image_gradients(model)
        GJ = self.query_jacobian_products(model, G)  # [N, Qimg, C]
        Lt = L[self.N:]
        arrays = dict(train_rows=self.ctx['train'], query_rows=self.ctx['test'], query_patient_ids=self.qids)
        checks = dict(cross_gpu_max_abs_log_prob_difference=diff)
        for t in Q.TARGETS:
            U = Q.logit_derivative(t, Lt, self.labels_test)          # [Qimg, C]
            Tq = -(GJ * U[None]).sum(2).T                            # [Qimg, N] per-query-image tau
            for wname, w in (('img', self.img_w), ('macro', self.macro_w)):
                tau = w @ Tq
                h = self.query_gradient(model, t, w, Lt)
                direct = -(G @ h).cpu().numpy()
                rel = float(np.linalg.norm(tau - direct) / max(np.linalg.norm(direct), 1e-300))
                checks[f'{t}_{wname}_jacobian_vs_direct_rel'] = rel
                if not np.isfinite(rel) or rel > 1e-8:
                    raise ValueError(f'Recombined proxy differs from direct proxy ({t}, {wname}: {rel})')
                arrays[f'tau_{wname}_{t}'] = direct
                arrays[f'pred_{wname}_{t}'] = self.predictions(direct)
            Tp = np.stack([Tq[u].mean(0) for u in self.units])       # [Qpat, N]
            arrays[f'pred_qimage_{t}'] = -(1. - self.M) @ Tq.T        # [m, Qimg]
            arrays[f'pred_qpatient_{t}'] = -(1. - self.M) @ Tp.T      # [m, Qpat]
        if seed in STABLE_SEEDS:
            patient = self.A @ arrays['tau_macro_ce']
            old = self.old_scores(seed, 'frozen_h_upweight')
            rel = float(np.linalg.norm(patient - old) / np.linalg.norm(old))
            checks['patient_sum_vs_lds_v1_frozen_h_rel'] = rel
            if not np.isfinite(rel) or rel > 1e-6:
                raise ValueError(f'Patient-aggregated CE proxy differs from LDS v1 frozen_h ({rel})')
        np.savez_compressed(tmp / 'proxy.npz', **arrays)
        rt.write(tmp / 'result.json', dict(seed=seed, checks=checks, endpoint='raw', sign='upweight: -h.g_i; deletion = -tau'))
        rt.finish(tmp, self.c, dict(kind='proxy', seed=seed), elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    # ----------------------------------------------------------------- score
    @exclusive
    def score_seed(self, seed, tag):
        """Multi-shift IF for every image-mean target of one seed (each target is its own
        resumable result), plus the patient-level gate against LDS v1 for seeds 42-44."""
        import torch
        import solver
        from analysis_stats import spearman
        todo = [t for t in self.c['score_targets'] if not (self.root / 'score' / f'seed{seed}_{t}' / 'complete.json').exists()]
        check_needed = seed in STABLE_SEEDS and not (self.root / 'score' / f'seed{seed}_check' / 'complete.json').exists()
        if not todo and not check_needed:
            return 'skipped'
        dev = self.pilot.DEVICE
        model = self.initialize(seed, self.baseline_state(seed)['raw'], double=True)
        _, L = self.predict(model)
        Lt = L[self.N:]
        G = self.image_gradients(model)
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], dev,
                                  self.c['weight_decay'], 0., self.c['batch'])
        if check_needed:
            # Gate: same solver, patient-macro CE, gamma = 1, aggregated over each patient's images.
            final, tmp = staged(self.root, Path('score') / f'seed{seed}_check', tag)
            start = time.monotonic()
            h = self.query_gradient(model, 'ce', self.macro_w, Lt)
            ms = solver.multishift_lanczos(hvp, h, [1.], self.c['lanczos_max_steps'], self.c['lanczos_est_tol'],
                                           dot=lambda a, b: float(a @ b))
            x = ms['x'][1.]
            true = float((hvp(x) + x - h).norm() / h.norm())
            patient = self.A @ (-(G @ x)).cpu().numpy()
            old = self.old_scores(seed, 'tau_upweight')
            gate = dict(true_rel_residual=true, rel_l2_difference=float(np.linalg.norm(patient - old) / np.linalg.norm(old)),
                        spearman=spearman(patient, old), lanczos_steps=ms['steps'])
            gate['passed'] = bool(true <= self.c['cg_fail_tol'] and gate['rel_l2_difference'] <= 1e-3
                                  and gate['spearman'] is not None and gate['spearman'] >= .999)
            rt.write(tmp / 'result.json', dict(seed=seed, gate=gate))
            if not gate['passed']:
                raise ValueError(f'Image-level IF does not reproduce LDS v1 patient scores: {gate}; diagnostics in {tmp}')
            np.savez_compressed(tmp / 'scores.npz', patient_tau_gamma1=patient)
            rt.finish(tmp, self.c, dict(kind='score', seed=seed, target='check'), elapsed_seconds=time.monotonic() - start)
            os.rename(tmp, final)
        if not todo:
            return 'done'
        probe = self.dc.lanczos_probe(hvp, G.shape[1], self.c['probe_steps'], 20261009 + seed, dev, torch.float64)
        for t in todo:
            final, tmp = staged(self.root, Path('score') / f'seed{seed}_{t}', tag)
            start = time.monotonic()
            h = self.query_gradient(model, t, self.img_w, Lt)
            proxy = -(G @ h).cpu().numpy()
            ms = solver.multishift_lanczos(hvp, h, self.c['gammas'], self.c['lanczos_max_steps'],
                                           self.c['lanczos_est_tol'], dot=lambda a, b: float(a @ b))
            arrays = dict(train_rows=self.ctx['train'], proxy=proxy, pred_proxy=self.predictions(proxy))
            gates = []
            for g in self.c['gammas']:
                x = ms['x'].get(g)
                true = None if x is None else float((hvp(x) + g * x - h).norm() / h.norm())
                true = true if true is not None and np.isfinite(true) else None
                q = solver.qualify(g, true, ms['ritz_min'], solver.probe_screen(probe, g), self.c['cg_fail_tol'])
                q.update(estimated_rel_residual=ms['final_estimated_rel_residual'][g],
                         solver_error=ms.get('errors', {}).get(g), spearman_if_vs_proxy_images=None)
                if q['qualified']:
                    tau = -(G @ x).cpu().numpy()
                    if not np.isfinite(tau).all():
                        q['qualified'] = False
                        q['reasons'].append('non-finite influence scores')
                    else:
                        arrays[f'tau_gamma{g:g}'] = tau
                        arrays[f'pred_gamma{g:g}'] = self.predictions(tau)
                        q['spearman_if_vs_proxy_images'] = spearman(tau, proxy)
                gates.append(q)
            rt.csv_write(tmp / 'lanczos_history.csv', ms['history'])
            np.savez_compressed(tmp / 'scores.npz', **arrays)
            rt.write(tmp / 'result.json', dict(seed=seed, target=t, weighting='img', gammas=gates,
                lanczos=dict(steps=ms['steps'], ritz_min=ms['ritz_min'], ritz_max=ms['ritz_max'],
                             invariant_subspace=ms['invariant_subspace']),
                random_probe=dict(steps=probe['steps'], endpoints=probe['endpoints']),
                stationarity_assumption_satisfied=False,
                note='Unqualified shifts have no tau and are excluded from LDS; not a proof of noninvertibility.'))
            rt.finish(tmp, self.c, dict(kind='score', seed=seed, target=t), elapsed_seconds=time.monotonic() - start)
            os.rename(tmp, final)
        return 'done'
