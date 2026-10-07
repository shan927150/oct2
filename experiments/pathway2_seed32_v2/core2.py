"""GPU work: training/harvest cells, gradient-inner-product proxies and the
multi-shift IF gamma sweep. Imports only frozen kernels; edits none of them."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import runtime as rt
import targets as T
import designs as DS
import solver
from functools import wraps

PILOT_SEEDS = (42, 43, 44)


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
        if method.__name__ == 'run_cell':
            key = args[0]['name']
        elif method.__name__ == 'score_task':
            key = f'{args[0]}_{args[1]}' + ('_pilot' if kwargs.get('pilot') else '')
        else:
            key = str(args[0])
        with rt.lock(self.root / '.locks' / f'{method.__name__}_{key}.lock'):
            return method(self, *args, **kwargs)
    return wrapped


class Session:
    """Loaded once per Slurm task; every cell in a pack reuses the data context."""

    def __init__(self, c, freeze_to=None):
        from design import make_design  # LDS v1, unchanged
        self.c, self.root = c, Path(c['out'])
        self.h = rt.sha(self.root / 'config.json')
        self.ctx, self.legacy, self.score, self.pilot, self.conv = rt.load_context(c)
        self.stable = module('seed32_stable_core', rt.STABLE_V1 / 'stable_core.py')
        import stage1_diagnostic_core as dc
        self.dc = dc
        ctx = self.ctx
        self.base = make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], 8, .5, 20260929)
        pilot_design = rt.read(Path(c['lds_pilot_out']) / 'score_seed42/design.json')
        if roundtrip(self.base) != pilot_design:
            raise ValueError('Regenerated pilot design differs from LDS v1 score_seed42/design.json')
        self.designs = roundtrip(DS.build_all(self.base, c['designs']))
        if freeze_to is not None:
            rt.write(freeze_to, dict(base=self.base, designs=self.designs))
        else:
            stored = rt.read(self.root / 'designs.json')
            if stored != roundtrip(dict(base=self.base, designs=self.designs)):
                raise ValueError('Designs differ from the smoke-frozen designs.json')
        self.N = len(ctx['train'])
        self.labels_test = ctx['y'][ctx['test']]
        self.weights = np.asarray(self.base['query_weights'])
        self.qids, self.units, self.coef = T.query_units(self.base)
        self.patients = self.base['train_patients']

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

    def predict(self, model, batch=64, return_log=False):
        import torch
        model.eval()
        p = next(model.parameters())
        rows, X, out, logs = self.ctx['interface'], self.ctx['X'], [], []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                x = torch.as_tensor(X[rows[s:s + batch]], device=p.device, dtype=p.dtype)
                logits = model(x)
                out.append(logits.softmax(1).cpu().numpy())
                if return_log:
                    logs.append(logits.log_softmax(1).cpu().numpy())
        return (np.concatenate(out), np.concatenate(logs)) if return_log else np.concatenate(out)

    def query_gradient(self, model, target, rows, weights, probs):
        """Use log-softmax CE directly: -w/p times softmax may overflow/underflow."""
        if target != 'ce':
            _, v = T.value_and_v(target, probs, self.ctx['y'][rows], weights)
            return self.score.shadow_h(model, self.ctx['X'][rows], v, self.pilot.DEVICE, self.c['batch'])
        import torch
        params = list(model.parameters())
        out = torch.zeros(sum(p.numel() for p in params), dtype=params[0].dtype, device=params[0].device)
        for i in range(0, len(rows), self.c['batch']):
            idx = rows[i:i + self.c['batch']]
            x = torch.as_tensor(self.ctx['X'][idx], dtype=params[0].dtype, device=params[0].device)
            y = torch.as_tensor(self.ctx['y'][idx], dtype=torch.long, device=params[0].device)
            w = torch.as_tensor(weights[i:i + len(idx)], dtype=params[0].dtype, device=params[0].device)
            loss = (torch.nn.functional.cross_entropy(model(x), y, reduction='none') * w).sum()
            out += self.score.flatten(torch.autograd.grad(loss, params))
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite CE query gradient')
        return out

    def old_scores(self, seed, key):
        import csv
        with (Path(self.c['lds_pilot_out']) / f'score_seed{seed}/scores.csv').open() as f:
            rows = list(csv.DictReader(f))
        if [int(r['patient']) for r in rows] != [p['patient'] for p in self.patients]:
            raise ValueError('Legacy score patient order differs')
        values = np.asarray([float(r[key]) for r in rows])
        if not np.isfinite(values).all() or np.linalg.norm(values) == 0:
            raise ValueError('Legacy score vector is nonfinite or zero')
        return values

    def gradient(self, model, rows, batch=64):
        """b_p = sum_{i in rows} grad CE_i / N (same as LDS v1 core.gradient)."""
        import torch
        params = list(model.parameters())
        out = torch.zeros(sum(p.numel() for p in params), dtype=params[0].dtype, device=params[0].device)
        for i in range(0, len(rows), batch):
            idx = rows[i:i + batch]
            x = torch.as_tensor(self.ctx['X'][idx], dtype=params[0].dtype, device=params[0].device)
            y = torch.as_tensor(self.ctx['y'][idx], dtype=torch.long, device=params[0].device)
            loss = torch.nn.functional.cross_entropy(model(x), y, reduction='sum') / self.N
            out += self.score.flatten(torch.autograd.grad(loss, params))
        return out

    def patient_matrix(self, model):
        import torch
        return torch.stack([self.gradient(model, p['rows']) for p in self.patients])

    def orders(self, seed):
        o = self.pilot.make_epoch_orders(self.ctx['train'], self.c['epochs'], seed + 700000)
        if seed in PILOT_SEEDS:
            saved = np.load(Path(self.c['full']) / f'stage1_order_seed{seed}.npz')['raw_index_order']
            if not np.array_equal(o[:len(saved)], saved[:len(o)]):
                raise ValueError('Baseline order prefix mismatch')
        return o

    def removed_rows(self, design, subset):
        mask = self.designs[design]['masks'][subset]
        return np.asarray([r for keep, p in zip(mask, self.patients) if not keep for r in p['rows']], dtype=np.int64)

    def train(self, cell):
        orders = self.orders(cell['seed'])
        if cell['kind'] == 'subset':
            removed = self.removed_rows(cell['design'], cell['subset'])
            orders = [o[~np.isin(o, removed)] for o in orders]
        return self.stable.train(self.initialize(cell['seed']), self.ctx['X'], self.ctx['y'], orders,
                                 np.array([], dtype=np.int64), 1., **self.c['recipe'])

    def baseline_state(self, seed):
        """Raw/EMA state of the full-data baseline for any seed (frozen or new)."""
        import torch
        if seed in PILOT_SEEDS:
            path = Path(self.c['baseline_out']) / f'seed{seed}_baseline/final.pt'
        else:
            d = self.root / 'cells' / f'full_seed{seed}'
            rt.completed(d, self.h)
            path = d / 'final.pt'
        return torch.load(path, map_location='cpu', weights_only=False)

    def values(self, P, log_probs=None):
        return T.all_values(P[self.N:], self.labels_test, self.weights, self.units,
                            None if log_probs is None else log_probs[self.N:])

    # ------------------------------------------------------------------ cells
    def harvest_source(self, cell):
        if cell['kind'] == 'full':
            d = Path(self.c['baseline_out']) / f"seed{cell['seed']}_baseline"
            return d, 1e-4  # stored probabilities are fp32 there
        d = Path(self.c['lds_pilot_out']) / f"subset{cell['subset']:03d}_seed{cell['seed']}"
        return d, 1e-10  # LDS v1 stored fp64 raw probabilities

    @exclusive
    def run_cell(self, cell, tag):
        import torch
        cells = self.root / 'cells'
        final = cells / cell['name']
        if final.exists():
            rt.completed(final, self.h, cell)
            return 'skipped'
        tmp = cells / f".{cell['name']}.tmp-{tag}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        start = time.monotonic()
        steps = None
        if cell['source'] == 'harvest':
            src, tol = self.harvest_source(cell)
            payload = torch.load(src / 'final.pt', map_location='cpu', weights_only=False)
            raw, ema = payload['raw'], payload['ema']
        else:
            m, e, opt, curve = self.train(cell)
            raw = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            ema = {k: v.detach().cpu() for k, v in e.state_dict().items()}
            torch.save(dict(raw=raw, ema=ema, cell=cell, recipe=self.c['recipe']), tmp / 'final.pt')
            rt.csv_write(tmp / 'curve.csv', curve)
            steps = curve[-1]['step']
            del m, e, opt
        interfaces = {ep: self.predict(self.initialize(cell['seed'], st, double=True), return_log=True)
                      for ep, st in (('raw', raw), ('ema', ema))}
        P = {ep: pair[0] for ep, pair in interfaces.items()}
        logs = {ep: pair[1] for ep, pair in interfaces.items()}
        harvest_check = None
        if cell['source'] == 'harvest':
            ce_difference = None
            with np.load(src / 'interface.npz') as z:
                if not np.array_equal(z['raw_index'], self.ctx['interface']):
                    raise ValueError('Harvest interface rows differ')
                diffs = {ep: float(np.abs(z[ep].astype(float) - P[ep]).max()) for ep in ('raw', 'ema') if ep in z.files}
                if cell['kind'] == 'subset':
                    ce_difference = abs(self.values(z['raw'])['ce']['Y'] - self.values(P['raw'], logs['raw'])['ce']['Y'])
            if 'raw' not in diffs or max(diffs.values()) > tol:
                raise ValueError(f'Harvested predictions differ from stored interface: {diffs}')
            if ce_difference is not None and (not np.isfinite(ce_difference) or ce_difference > 1e-8):
                raise ValueError('Harvest CE differs despite small absolute probability error')
            harvest_check = dict(source=str(src), max_abs_probability_difference=diffs, tolerance=tol,
                                 ce_macro_abs_difference=ce_difference)
        np.savez_compressed(tmp / 'interface.npz', raw=P['raw'], ema=P['ema'],
                            log_raw=logs['raw'], log_ema=logs['ema'],
                            raw_index=self.ctx['interface'], classes=self.ctx['y'][self.ctx['interface']])
        vals = {ep: {t: v['Y'] for t, v in self.values(P[ep], logs[ep]).items()} for ep in P}
        n_images = self.N - (len(self.removed_rows(cell['design'], cell['subset'])) if cell['kind'] == 'subset' else 0)
        expected_steps = self.c['epochs'] * int(np.ceil(n_images / self.c['recipe']['batch']))
        if steps is not None and steps != expected_steps:
            raise ValueError('Unexpected optimizer step count')
        steps = expected_steps
        rt.write(tmp / 'result.json', dict(cell=cell, Y=vals, n_train_images=n_images, optimizer_steps=steps,
                                           harvest_check=harvest_check))
        rt.finish(tmp, self.c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    # ------------------------------------------------------------------ smoke
    def smoke(self, dest):
        import torch
        from formula_tests import torch_solver_checks
        timing = {}
        checks = torch_solver_checks(device=self.pilot.DEVICE)
        # 1) the new controller reproduces an LDS v1 subset model bit for bit
        cell = dict(name='replay', kind='subset', design='pilot8', subset=0, seed=42, source='train')
        t0 = time.monotonic()
        m, e, _, _ = self.train(cell)
        timing['half_data_training_seconds'] = time.monotonic() - t0
        ref = torch.load(Path(self.c['lds_pilot_out']) / 'subset000_seed42/final.pt', map_location='cpu', weights_only=False)
        if not (self.dc.exact_tree(m.state_dict(), ref['raw']) and self.dc.exact_tree(e.state_dict(), ref['ema'])):
            raise ValueError('Pilot subset000_seed42 replay differs from LDS v1')
        del m, e
        # 2) full-horizon no-op against the frozen stable baseline (same process,
        #    after another training: also proves pack-sequential determinism)
        t0 = time.monotonic()
        m, e, opt, _ = self.train(dict(name='noop', kind='full', seed=42, source='train'))
        timing['full_data_training_seconds'] = time.monotonic() - t0
        base = self.baseline_state(42)
        for a, b in [(m.state_dict(), base['raw']), (e.state_dict(), base['ema']), (opt.state_dict(), base['optimizer'])]:
            if not self.dc.exact_tree(a, b):
                raise ValueError('Full-horizon no-op differs from frozen stable baseline')
        del m, e, opt
        # 3) fp64 operator timing for the score/proxy budgets
        model = self.initialize(42, base['raw'], double=True)
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], self.pilot.DEVICE,
                                  self.c['weight_decay'], 0., self.c['batch'])
        v = torch.randn(sum(p.numel() for p in model.parameters()), dtype=torch.float64, device=self.pilot.DEVICE)
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sync(); t0 = time.monotonic()
        for _ in range(2):
            hvp(v)
        sync(); timing['fp64_hvp_seconds'] = (time.monotonic() - t0) / 2
        t0 = time.monotonic(); self.patient_matrix(model); sync()
        timing['fp64_patient_gradients_seconds'] = time.monotonic() - t0
        steps = self.c['lanczos_max_steps'] + self.c['probe_steps'] + len(self.c['gammas']) + 2
        timing['score_task_operator_estimate_hours'] = (steps * timing['fp64_hvp_seconds'] + timing['fp64_patient_gradients_seconds']) / 3600
        timing['note'] = 'Estimate excludes reorthogonalization, query gradients, startup and I/O; not an upper bound.'
        rt.write(dest / 'result.json', dict(solver_checks=checks, timing=timing, pilot_replay_exact=True,
                                            noop_exact=True, designs={k: dict(kind=v['kind'], m=len(v['masks']),
                                            n_images_range=v['n_images_range'], diagnostics=v.get('diagnostics', {}).get('acceptance_rate'))
                                            for k, v in self.designs.items()}))

    # ----------------------------------------------------------------- proxy
    @exclusive
    def proxy_seed(self, seed, tag):
        """Gradient-inner-product proxy -h_Y . b_p for the macro target and every
        query patient, all targets, raw endpoint. Cheap; used for tasks 1 and 5."""
        import torch
        outdir = self.root / 'proxy'
        final = outdir / f'seed{seed}'
        if final.exists():
            rt.completed(final, self.h, dict(kind='proxy', seed=seed))
            return 'skipped'
        tmp = outdir / f'.seed{seed}.tmp-{tag}'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        start = time.monotonic()
        state = self.baseline_state(seed)
        model = self.initialize(seed, state['raw'], double=True)
        P, log_probs = self.predict(model, return_log=True)
        rt.completed(self.root / 'cells' / f'full_seed{seed}', self.h)
        with np.load(self.root / 'cells' / f'full_seed{seed}' / 'interface.npz') as z:
            if float(np.abs(z['raw'] - P).max()) > 1e-12:
                raise ValueError('Proxy model differs from the full_seed cell predictions')
        B = self.patient_matrix(model)
        Xt, test = self.ctx['X'], self.ctx['test']
        Pt = P[self.N:]
        arrays = dict(train_patient_ids=np.asarray([p['patient'] for p in self.patients]),
                      query_patient_ids=self.qids, b_norm=B.norm(dim=1).cpu().numpy())
        checks = {}
        for t in T.TARGETS:
            Y = self.values(P, log_probs)[t]['Y']
            h = self.query_gradient(model, t, test, self.weights, Pt)
            macro = -(B @ h).cpu().numpy()
            per_query = np.zeros((len(self.units), len(self.patients)))
            for qi, u in enumerate(self.units):
                hq = self.query_gradient(model, t, test[u], np.full(len(u), 1. / len(u)), Pt[u])
                per_query[qi] = -(B @ hq).cpu().numpy()
            recombined = self.coef @ per_query
            rel = float(np.linalg.norm(recombined - macro) / max(np.linalg.norm(macro), 1e-300))
            if not np.isfinite(rel) or rel > 1e-8:
                raise ValueError(f'Per-query proxies do not recombine to the macro proxy ({t}: {rel})')
            checks[t] = dict(per_query_recombination_rel_error=rel, Y=Y)
            arrays[f'proxy_macro_{t}'] = macro
            arrays[f'proxy_query_{t}'] = per_query
        if seed in PILOT_SEEDS:
            old = self.old_scores(seed, 'frozen_h_upweight')
            rel = float(np.linalg.norm(old - arrays['proxy_macro_ce']) / np.linalg.norm(old))
            checks['lds_v1_frozen_h_ce_rel_difference'] = rel
            if not np.isfinite(rel) or rel > 1e-6:
                raise ValueError(f'CE proxy differs from LDS v1 frozen_h ({rel})')
        np.savez_compressed(tmp / 'proxy.npz', **arrays)
        rt.write(tmp / 'result.json', dict(seed=seed, checks=checks, endpoint='raw', sign='upweight: -h.b_p'))
        rt.finish(tmp, self.c, dict(kind='proxy', seed=seed), elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    # ----------------------------------------------------------------- score
    @exclusive
    def score_task(self, seed, target, tag, pilot=False):
        """Multi-shift IF for one (seed, target): every gamma from one Krylov space."""
        import torch
        outdir = self.root / ('score_pilot' if pilot else 'score')
        gammas = self.c['gamma_pilot']['gammas'] if pilot else self.c['gammas']
        name = f'seed{seed}_{target}'
        final = outdir / name
        if final.exists():
            rt.completed(final, self.h, dict(kind='score', seed=seed, target=target))
            return 'skipped'
        tmp = outdir / f'.{name}.tmp-{tag}'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        start = time.monotonic()
        dev = self.pilot.DEVICE
        model = self.initialize(seed, self.baseline_state(seed)['raw'], double=True)
        P, log_probs = self.predict(model, return_log=True)
        Y = self.values(P, log_probs)[target]['Y']
        h = self.query_gradient(model, target, self.ctx['test'], self.weights, P[self.N:])
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], dev,
                                  self.c['weight_decay'], 0., self.c['batch'])
        probe = self.dc.lanczos_probe(hvp, h.numel(), self.c['probe_steps'], 20261006 + seed, dev, torch.float64)
        ms = solver.multishift_lanczos(hvp, h, gammas, self.c['lanczos_max_steps'],
                                       self.c['lanczos_est_tol'], dot=lambda a, b: float(a @ b))
        B = self.patient_matrix(model)
        proxy = -(B @ h).cpu().numpy()
        arrays = dict(train_patient_ids=np.asarray([p['patient'] for p in self.patients]), proxy=proxy)
        gate = []
        from analysis_stats import spearman
        for g in gammas:
            x = ms['x'].get(g)
            true = None if x is None else float((hvp(x) + g * x - h).norm() / h.norm())
            true = true if true is not None and np.isfinite(true) else None
            q = solver.qualify(g, true, ms['ritz_min'], solver.probe_screen(probe, g), self.c['cg_fail_tol'])
            q['x_norm'] = None if x is None else float(x.norm())
            q['estimated_rel_residual'] = ms['final_estimated_rel_residual'][g]
            q['solver_error'] = ms.get('errors', {}).get(g)
            q['spearman_if_vs_proxy_148_patients'] = None
            if q['qualified']:
                tau = -(B @ x).cpu().numpy()
                if not np.isfinite(tau).all():
                    q['qualified'] = False
                    q['reasons'].append('non-finite influence scores')
                else:
                    arrays[f'tau_gamma{g:g}'] = tau
                    q['spearman_if_vs_proxy_148_patients'] = spearman(tau, proxy)
            gate.append(q)
        consistency = None
        if target == 'ce' and seed in PILOT_SEEDS and 'tau_gamma1' in arrays:
            old = self.old_scores(seed, 'tau_upweight')
            new = arrays['tau_gamma1']
            consistency = dict(rel_l2_difference=float(np.linalg.norm(new - old) / np.linalg.norm(old)),
                               spearman=spearman(new, old))
            consistency['passed'] = bool(consistency['rel_l2_difference'] <= 1e-3
                                         and consistency['spearman'] is not None and consistency['spearman'] >= .999)
        g1 = next((q for q in gate if q['gamma'] == 1.), None)
        rt.csv_write(tmp / 'lanczos_history.csv', ms['history'])
        np.savez_compressed(tmp / 'scores.npz', **arrays)
        rt.write(tmp / 'result.json', dict(seed=seed, target=target, Y=Y, gammas=gate,
            lanczos=dict(steps=ms['steps'], ritz_min=ms['ritz_min'], ritz_max=ms['ritz_max'],
                         invariant_subspace=ms['invariant_subspace']),
            random_probe=dict(steps=probe['steps'], endpoints=probe['endpoints']),
            lds_v1_consistency=consistency, gamma1_qualified=None if g1 is None else g1['qualified'],
            stationarity_assumption_satisfied=False,
            note='Completed assessment is not a successful solve. Unqualified shifts have no tau and are excluded from LDS.'))
        if consistency is not None and not consistency['passed']:
            rt.write(tmp / 'failed.json', dict(reason='CE gamma1 replay mismatch', consistency=consistency))
            raise ValueError(f'gamma=1 CE scores differ from LDS v1 CG scores: {consistency}; diagnostics: {tmp}')
        rt.finish(tmp, self.c, dict(kind='score', seed=seed, target=target), elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done' if all(q['qualified'] for q in gate) else 'assessed_with_unqualified_shifts'
