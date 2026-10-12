"""GPU work for the cross-stage LDS ladder (mentor 2026-10-12).

Reuses the v3 Session (core3.py, byte-for-byte) for the Stage-1 context, the frozen image
design, model initialisation, fp64 predictions and per-image gradients, plus the legacy
07/05 kernels (shadow_h, make_hvp, linearized_predictions, attack training/evaluation).
No Stage-1 model is trained here: subset and full models are read from the v3 run.

Conventions (unchanged from v3): g_i = grad CE_i(theta)/N; tau_i = -hbar^T (H + gamma I)^-1 g_i is the
UPWEIGHT effect on the attack objective, so deleting image i contributes -tau_i.
"""
import os
from pathlib import Path
import shutil
import time
import numpy as np
import runtime as rt
import attack4 as A
from functools import wraps
from core3 import Session, staged, exclusive  # image LDS v3, unchanged

V3_DESIGN = 'img32'


def exclusive_cell(method):
    """One writer per truth cell, keyed by the cell name (never the attempt tag)."""
    @wraps(method)
    def wrapped(self, cell, *args, **kwargs):
        with rt.lock(self.root / '.locks' / f"{method.__name__}_{cell['name']}.lock"):
            return method(self, cell, *args, **kwargs)
    return wrapped


class XSession(Session):
    def __init__(self, c, freeze_to=None):
        super().__init__(c, freeze_to=freeze_to)
        pargs = self.ctx['pargs']
        hp = c['attack']
        for key, attr in (('epochs', 'attack_epochs'), ('lr', 'attack_lr'), ('batch', 'attack_batch_size')):
            if getattr(pargs, attr) != hp[key]:
                raise ValueError(f'Legacy attack {attr} differs from the prepared config')
        self.hp = hp
        self.v3 = Path(c['v3_out'])
        self.v3_hash = c['v3_config_sha256']
        self.v3_cells = {x['name']: x for x in c['v3_cells']}
        self.rows3 = self.ctx['interface']             # 4343 shadow-3 rows: train then heldout
        self.dev = self.pilot.DEVICE

    # ------------------------------------------------------------ frozen inputs
    def frozen(self):
        rt.completed(self.root / 'smoke', self.h, dict(kind='xsmoke'))
        return rt.read(self.root / 'smoke' / 'frozen.json')

    def fixed(self):
        if getattr(self, '_fixed', None) is None:
            rt.completed(self.root / 'smoke', self.h, dict(kind='xsmoke'))
            with np.load(self.root / 'smoke' / 'fixed_rows.npz') as z:
                self._fixed = {k: z[k] for k in z.files}
        return self._fixed

    def build_fixed(self):
        """Legacy interface of the frozen shadows + target queries; the shadow-3 block is a placeholder."""
        import torch
        c, ctx, pilot = self.c, self.ctx, self.pilot
        splits = rt.read(Path(c['full']) / 'splits/fresh_patient_split.json')
        X, y = ctx['X'], ctx['y']
        n_in, n_cls = X.shape[1], int(y.max()) + 1
        ck = Path(c['full']) / 'checkpoints'
        xs, ms, cs, sids, raws = [], [], [], [], []
        for sid, split in enumerate(splits['shadow_models']):
            tr = np.asarray(split['train_idx'], dtype=np.int64)
            te = np.asarray(split['test_idx'], dtype=np.int64)
            rows = np.concatenate([tr, te])
            if sid == A.AFFECTED:
                if not np.array_equal(rows, self.rows3):
                    raise ValueError('Legacy shadow-3 interface order differs from the v3 interface rows')
                p = np.full((len(rows), n_cls), np.nan, dtype=np.float32)
            else:
                model = pilot.load_model(ck / f'shadow_{sid}_fixed.pt', n_in, 128, n_cls)
                p = pilot.get_predictions(model, X[rows]).astype(np.float32)
                del model
            xs.append(p); raws.append(rows); cs.append(y[rows].astype(np.int64)); sids.append(np.full(len(rows), sid))
            ms.append(np.r_[np.ones(len(tr), dtype=np.int64), np.zeros(len(te), dtype=np.int64)])
        if len(splits['shadow_models']) != c['n_shadow']:
            raise ValueError('Shadow count differs from the legacy config')
        target = pilot.load_model(ck / 'target_fixed.pt', n_in, 128, n_cls)
        q = pilot.make_target_queries(target, splits, X, y)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return dict(x=np.concatenate(xs), membership=np.concatenate(ms), classes=np.concatenate(cs),
                    shadow_id=np.concatenate(sids).astype(np.int64), raw_index=np.concatenate(raws),
                    q_x=np.asarray(q['x'], dtype=np.float32), q_membership=np.asarray(q['membership'], dtype=np.int64),
                    q_classes=np.asarray(q['classes'], dtype=np.int64), q_raw_index=np.asarray(q['raw_index']))

    # --------------------------------------------------------------- v3 inputs
    def v3_cell(self, name):
        cell = self.v3_cells[name]
        d = self.v3 / 'cells' / name
        rt.verify_hashes({str(d / 'complete.json'): self.c['input_sha256'][str(d / 'complete.json')]})
        rt.completed(d, self.v3_hash, cell)
        return d

    def v3_probs(self, name):
        with np.load(self.v3_cell(name) / 'interface.npz') as z:
            if not np.array_equal(z['raw_index'], self.rows3):
                raise ValueError('v3 interface rows differ: ' + name)
            return np.asarray(z['raw'], dtype=np.float64)

    def full_name(self, seed):
        return f'full_seed{seed}'

    def subset_name(self, j, seed):
        return f'{V3_DESIGN}_s{j:03d}_seed{seed}'

    def state(self, name, seed):
        import torch
        if name == self.full_name(seed):
            return self.baseline_state(seed)['raw']
        return torch.load(self.v3_cell(name) / 'final.pt', map_location='cpu', weights_only=False)['raw']

    def model(self, name, seed):
        return self.initialize(seed, self.state(name, seed), double=True)

    def flat_params(self, model):
        return self.score.flatten([p.detach() for p in self.score.flat_params(model)])

    # ----------------------------------------------------------------- attacks
    def condition(self, fixed, p3, k, cls, adjoint_gammas=None):
        return A.class_condition(self.pilot, self.hp, fixed, A.assemble(fixed, p3), cls, k,
                                 adjoint_gammas=adjoint_gammas, device=self.dev)

    def h_vectors(self, model, fixed, v_by_class):
        """h_c = sum_r (dp_r/dtheta)^T v_r over the class-c shadow-3 rows (unweighted)."""
        X = self.ctx['X']
        out = {}
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        for cls, v in v_by_class.items():
            sel = np.flatnonzero(cls3 == cls)
            out[cls] = self.score.shadow_h(model, X[self.rows3[sel]], v[sel], self.dev, self.c['batch'])
        return out

    def interface_objective(self, model, v):
        """F(theta) = sum_r v_r . p_r(theta) over all shadow-3 rows (for the FD check of h)."""
        p, _ = self.predict(model)
        return float((p * v).sum())

    # ------------------------------------------------------------------- smoke
    def xsmoke(self, dest):
        import torch
        c, timing, checks = self.c, {}, {}
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        # 1) designs identical to the v3 run (written by the caller through freeze_to)
        v3d = rt.read(self.v3 / 'designs.json')
        if rt.read(dest / 'designs.json') != v3d:
            raise ValueError('Regenerated designs differ from the v3 run designs.json')
        # 2) frozen shadows / target
        t0 = time.monotonic()
        fixed = self.build_fixed()
        timing['fixed_rows_seconds'] = time.monotonic() - t0
        # 3) full-model predictions reproduce the v3 interface for every seed
        diffs = {}
        for s in c['seeds']:
            p, _ = self.predict(self.model(self.full_name(s), s))
            diffs[s] = float(np.abs(p - self.v3_probs(self.full_name(s))).max())
        checks['full_prediction_vs_v3_max_abs'] = diffs
        if max(diffs.values()) > c['v3_prediction_tol']:
            raise ValueError(f'Full-model predictions differ from the v3 interface: {diffs}')
        # 4) attack gate and attack-Hessian spectra on every full baseline (no subset outcome is read)
        gA = c['attack_gammas']
        base, t_attack, t_adj = [], [], []
        for s in c['seeds']:
            p3 = self.v3_probs(self.full_name(s))
            x = A.assemble(fixed, p3)
            for k in c['attack_seeds']:
                for cls in c['classes']:
                    tr, _, _ = A.class_rows(fixed, cls)
                    te = fixed['q_classes'] == cls
                    sync(); t0 = time.monotonic()
                    model = A.train(self.pilot, self.hp, x[tr], fixed['membership'][tr], k)
                    sync(); t_attack.append(time.monotonic() - t0)
                    met = A.evaluate(self.pilot, model, fixed['q_x'][te], fixed['q_membership'][te])
                    t0 = time.monotonic()
                    adj = A.adjoint(model, x[tr], fixed['membership'][tr], fixed['q_x'][te], fixed['q_membership'][te], gA, self.dev)
                    sync(); t_adj.append(time.monotonic() - t0)
                    for e in adj['per_gamma'].values():
                        e.pop('v')
                    base.append(dict(seed=s, attack_seed=k, cls=cls, metrics=met,
                                     passed=A.class_gate(met, c['gate_auc'], c['gate_bacc']),
                                     qualified_gamma=A.qualified_gamma(adj['per_gamma'], c['attack_solve_tol']),
                                     **{key: adj[key] for key in ('hessian_eig_min', 'hessian_eig_max', 'train_grad_norm', 'q_norm')},
                                     per_gamma=adj['per_gamma']))
        timing['attack_train_seconds_median'] = float(np.median(t_attack))
        timing['attack_adjoint_seconds_median'] = float(np.median(t_adj))
        qualified = [cls for cls in c['classes'] if all(r['passed'] for r in base if r['cls'] == cls)]
        if not qualified:
            raise ValueError('No attack class passes the gate on every full baseline; stop and discuss')
        omega = A.weights(fixed['q_classes'], qualified)
        ok = [g for g in gA if all(r['qualified_gamma'][g] for r in base if r['cls'] in qualified)]
        if not ok:
            raise ValueError('No attack damping value is positive definite for every qualified attack')
        gstar = min(ok)
        # 5) determinism and agreement with the legacy adjoint
        s, k, cls = c['seeds'][0], c['attack_seeds'][0], qualified[0]
        x = A.assemble(fixed, self.v3_probs(self.full_name(s)))
        tr, _, _ = A.class_rows(fixed, cls)
        te = fixed['q_classes'] == cls
        m1 = A.train(self.pilot, self.hp, x[tr], fixed['membership'][tr], k)
        m2 = A.train(self.pilot, self.hp, x[tr], fixed['membership'][tr], k)
        if not self.dc.exact_tree(m1.state_dict(), m2.state_dict()):
            raise ValueError('Attack training is not deterministic')
        mine = A.adjoint(m1, x[tr], fixed['membership'][tr], fixed['q_x'][te], fixed['q_membership'][te], [gstar], self.dev)
        import copy
        legacy = self.score.attack_implicit_v(copy.deepcopy(m1).double(), x[tr].astype(np.float64), fixed['membership'][tr],
                                              fixed['q_x'][te].astype(np.float64), fixed['q_membership'][te], gstar, self.dev)
        rel = float(np.linalg.norm(mine['per_gamma'][gstar]['v'] - legacy['v']) / max(np.linalg.norm(legacy['v']), 1e-300))
        checks['adjoint_vs_legacy_rel'] = rel
        if not rel <= 1e-10:
            raise ValueError(f'Dense adjoint differs from legacy attack_implicit_v ({rel})')
        # 6) chain rule for h and the JVP, at the first seed and the first subset
        model = self.model(self.full_name(s), s)
        v_by_class, v_all = {}, np.zeros((len(self.rows3), 4))
        for cl in qualified:
            out, _ = self.condition(fixed, self.v3_probs(self.full_name(s)), k, cl, adjoint_gammas=[gstar])
            v_by_class[cl] = out['v'][gstar] * omega[cl]
            v_all += v_by_class[cl]
        sync(); t0 = time.monotonic()
        hs = self.h_vectors(model, fixed, v_by_class)
        sync(); timing['shadow_h_seconds_all_classes'] = time.monotonic() - t0
        h = sum(hs.values())
        theta = self.flat_params(model)
        sub = self.subset_name(0, s)
        dtheta = self.flat_params(self.model(sub, s)) - theta
        # FD direction = a real subset displacement (what L2-lin contracts h with), rescaled small
        d = dtheta / dtheta.norm() * theta.norm() * 1e-3
        fd_rows = []
        for eps in (1., 1e-1, 1e-2, 1e-3):
            fp = self.interface_objective(self.shift(model, s, eps * d), v_all)
            fm = self.interface_objective(self.shift(model, s, -eps * d), v_all)
            fd_rows.append(dict(eps=eps, fd=(fp - fm) / (2 * eps), analytic=float(h @ d)))
        checks['h_chain_fd'] = fd_rows
        if not any(abs(r['fd'] - r['analytic']) <= 1e-5 * max(abs(r['analytic']), 1e-12) + 1e-14 for r in fd_rows):
            raise ValueError(f'h chain rule differs from finite differences: {fd_rows}')
        sync(); t0 = time.monotonic()
        lin = self.score.linearized_predictions(model, self.ctx['X'][self.rows3], dtheta, self.dev, self.c['batch'])
        sync(); timing['jvp_seconds'] = time.monotonic() - t0
        p0, _ = self.predict(model)
        scale = float(np.abs(lin['raw'] - p0).max())
        jvp_rows = []
        for eps in (1e-4, 1e-5, 1e-6):
            pp, _ = self.predict(self.shift(model, s, eps * dtheta))
            pm, _ = self.predict(self.shift(model, s, -eps * dtheta))
            jvp_rows.append(dict(eps=eps, max_abs_error=float(np.abs((lin['raw'] - p0) - (pp - pm) / (2 * eps)).max()),
                                 scale=scale))
        checks['jvp_vs_fd'] = jvp_rows
        if not any(r['max_abs_error'] <= 1e-5 * max(scale, 1e-12) for r in jvp_rows):
            raise ValueError(f'JVP differs from finite differences: {jvp_rows}')
        frozen = dict(qualified_classes=qualified, omega={str(k2): v for k2, v in omega.items()}, attack_gamma_star=gstar,
                      attack_gammas_qualified=ok, query_counts={str(cl): int((fixed['q_classes'] == cl).sum()) for cl in c['classes']},
                      rule='Classes: AUC and balanced-accuracy gate on all full-baseline attacks. gamma_A*: smallest grid value '
                           'with positive damped spectrum and solve residual <= tol for every qualified attack. '
                           'Fixed before any subset outcome is computed.')
        np.savez_compressed(dest / 'fixed_rows.npz', **fixed)
        rt.write(dest / 'frozen.json', frozen)
        rt.write(dest / 'attack_baselines.json', dict(rows=_clean(base)))
        rt.write(dest / 'result.json', _clean(dict(checks=checks, timing=timing, frozen=frozen)))

    def shift(self, model, seed, delta):
        import torch
        m = self.initialize(seed, None, double=True)
        m.load_state_dict(model.state_dict())
        params = self.score.flat_params(m)
        with torch.no_grad():
            for p, dp in zip(params, self.score.unflatten(delta, params)):
                p.add_(dp)
        return m.eval()

    # -------------------------------------------------------------------- base
    @exclusive
    def base_seed(self, seed, tag):
        """Full-baseline attacks J00, adjoint v for every gamma_A, and h at gamma_A* (per attack seed, class)."""
        import torch
        final, tmp = staged(self.root, Path('base') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, dict(kind='base', seed=seed))
            return 'skipped'
        start = time.monotonic()
        c, fr, fixed = self.c, self.frozen(), self.fixed()
        smoke = {(r['seed'], r['attack_seed'], r['cls']): r for r in rt.read(self.root / 'smoke/attack_baselines.json')['rows']}
        gA, gstar = c['attack_gammas'], fr['attack_gamma_star']
        p3 = self.v3_probs(self.full_name(seed))
        model = self.model(self.full_name(seed), seed)
        V = np.zeros((len(c['attack_seeds']), len(c['classes']), len(gA), len(self.rows3), 4))
        Hs = []
        rows = []
        for ki, k in enumerate(c['attack_seeds']):
            v_star = {}
            for ci, cls in enumerate(c['classes']):
                out, _ = self.condition(fixed, p3, k, cls, adjoint_gammas=gA)
                ref = smoke[(seed, k, cls)]['metrics']['cross_entropy']
                diff = abs(out['metrics']['cross_entropy'] - ref)
                if diff > c['j00_tol']:
                    raise ValueError(f'J00 differs from smoke for seed {seed} attack {k} class {cls}: {diff}')
                for gi, g in enumerate(gA):
                    V[ki, ci, gi] = out['v'][g]
                v_star[cls] = out['v'][gstar]
                rows.append(dict(attack_seed=k, cls=cls, metrics=out['metrics'], J00_minus_smoke=diff, adjoint=out['adjoint']))
            hk = self.h_vectors(model, fixed, v_star)
            Hs.append(torch.stack([hk[cls] for cls in c['classes']]).cpu().numpy())
        np.savez_compressed(tmp / 'v.npz', v=V, attack_gammas=np.asarray(gA), classes=np.asarray(c['classes']),
                            attack_seeds=np.asarray(c['attack_seeds']))
        np.savez(tmp / 'h.npz', h=np.stack(Hs), gamma_star=gstar)   # [K, C, D], unweighted, fp64
        rt.write(tmp / 'result.json', _clean(dict(seed=seed, attack_gamma_star=gstar, rows=rows)))
        rt.finish(tmp, c, dict(kind='base', seed=seed), elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    def base_arrays(self, seed):
        cache = self.__dict__.setdefault('_base', {})
        if seed not in cache:
            cache.clear()   # packs are seed-major; keep at most one seed's h in memory
            cache[seed] = self._base_arrays(seed)
        return cache[seed]

    def _base_arrays(self, seed):
        d = self.root / 'base' / f'seed{seed}'
        rt.completed(d, self.h, dict(kind='base', seed=seed))
        with np.load(d / 'v.npz') as z:
            V = z['v']
        with np.load(d / 'h.npz') as z:
            H = z['h']
        J00 = {(r['attack_seed'], r['cls']): r['metrics']['cross_entropy'] for r in rt.read(d / 'result.json')['rows']}
        return V, H, J00

    # ------------------------------------------------------------------- truth
    @exclusive_cell
    def truth_cell(self, cell, tag):
        """J10 on the true subset predictions, L1 for every gamma_A, L2-lin and L2-retrain for one (subset, seed)."""
        import torch
        final, tmp = staged(self.root, Path('truth') / cell['name'], tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, cell)
            return 'skipped'
        start = time.monotonic()
        c, fixed = self.c, self.fixed()
        s, j = cell['seed'], cell['subset']
        V, H, J00_base = self.base_arrays(s)
        J00 = self.j00(s, fixed, J00_base)
        p_full = self.v3_probs(self.full_name(s))
        p_sub = self.v3_probs(self.subset_name(j, s))
        model = self.model(self.full_name(s), s)
        theta = self.flat_params(model)
        dtheta = self.flat_params(self.model(self.subset_name(j, s), s)) - theta
        lin = self.score.linearized_predictions(model, self.ctx['X'][self.rows3], dtheta, self.dev, c['batch'])
        p_hat = lin['projected']
        dP = p_sub - p_full
        dth = dtheta.cpu().numpy()
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        rows = []
        for ki, k in enumerate(c['attack_seeds']):
            for ci, cls in enumerate(c['classes']):
                sel = cls3 == cls
                j10, _ = self.condition(fixed, p_sub, k, cls)
                jhat, _ = self.condition(fixed, p_hat, k, cls)
                rows.append(dict(attack_seed=k, cls=cls, J00=J00[(k, cls)], J10=j10['metrics']['cross_entropy'],
                                 J_L2_retrain=jhat['metrics']['cross_entropy'],
                                 auc10=j10['metrics']['auc'], auc_L2_retrain=jhat['metrics']['auc'],
                                 L1={f'{g:g}': float((V[ki, ci, gi][sel] * dP[sel]).sum()) for gi, g in enumerate(c['attack_gammas'])},
                                 L2_lin=float(H[ki, ci] @ dth)))
        member = fixed['membership'][fixed['shadow_id'] == A.AFFECTED] == 1
        deleted = np.r_[self.M[j] == 0, np.zeros(len(self.rows3) - self.N, dtype=bool)]
        groups = dict(deleted=deleted, retained=member & ~deleted, nonmember=~member)
        drift = {g: dict(js_true=float(A.js(p_full[msk], p_sub[msk]).mean()),
                         js_linearized=float(A.js(p_full[msk], p_hat[msk]).mean()),
                         js_true_vs_linearized=float(A.js(p_sub[msk], p_hat[msk]).mean())) for g, msk in groups.items()}
        rt.write(tmp / 'result.json', _clean(dict(cell=cell, rows=rows, dtheta_norm=float(np.linalg.norm(dth)),
            jvp=dict(row_clip_rate=lin['row_clip_rate'], mean_negative_mass=lin['mean_negative_mass'],
                     mean_projection_l1=lin['mean_projection_l1']), drift=drift)))
        rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'

    def j00(self, seed, fixed, J00_base):
        """Pack-local J00 (same device as J10) cross-checked against the base task."""
        cache = getattr(self, '_j00', {})
        if seed not in cache:
            p3 = self.v3_probs(self.full_name(seed))
            out = {}
            for k in self.c['attack_seeds']:
                for cls in self.c['classes']:
                    r, _ = self.condition(fixed, p3, k, cls)
                    out[(k, cls)] = r['metrics']['cross_entropy']
                    if abs(out[(k, cls)] - J00_base[(k, cls)]) > self.c['j00_tol']:
                        raise ValueError(f'Pack J00 differs from base for seed {seed}, attack {k}, class {cls}')
            cache[seed] = out
            self._j00 = cache
        return cache[seed]

    # ------------------------------------------------------------------- score
    @exclusive
    def xscore_seed(self, seed, tag):
        """Multi-shift Stage-1 IF for hbar_s = sum_c omega_c mean_k h_{k,c} (attack objective), plus proxy."""
        import torch
        import solver
        from analysis_stats import spearman
        final, tmp = staged(self.root, Path('score') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, dict(kind='xscore', seed=seed))
            return 'skipped'
        start = time.monotonic()
        c, fr = self.c, self.frozen()
        _, H, _ = self.base_arrays(seed)
        hbar = sum(fr['omega'][str(cls)] * H[:, ci].mean(0) for ci, cls in enumerate(c['classes'])
                   if cls in fr['qualified_classes'])
        model = self.model(self.full_name(seed), seed)
        h = torch.as_tensor(hbar, dtype=torch.float64, device=self.dev)
        G = self.image_gradients(model)
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], self.dev,
                                  c['weight_decay'], 0., c['batch'])
        probe = self.dc.lanczos_probe(hvp, G.shape[1], c['probe_steps'], 20261012 + seed, self.dev, torch.float64)
        proxy = -(G @ h).cpu().numpy()
        ms = solver.multishift_lanczos(hvp, h, c['gammas'], c['lanczos_max_steps'], c['lanczos_est_tol'],
                                       dot=lambda a, b: float(a @ b))
        arrays = dict(train_rows=self.ctx['train'], proxy=proxy, pred_proxy=self.predictions(proxy), hbar=hbar)
        gates = []
        for g in c['gammas']:
            x = ms['x'].get(g)
            true = None if x is None else float((hvp(x) + g * x - h).norm() / h.norm())
            true = true if true is not None and np.isfinite(true) else None
            q = solver.qualify(g, true, ms['ritz_min'], solver.probe_screen(probe, g), c['cg_fail_tol'])
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
        rt.write(tmp / 'result.json', _clean(dict(seed=seed, gammas=gates, hbar_norm=float(np.linalg.norm(hbar)),
            lanczos=dict(steps=ms['steps'], ritz_min=ms['ritz_min'], ritz_max=ms['ritz_max'],
                         invariant_subspace=ms['invariant_subspace']),
            random_probe=dict(steps=probe['steps'], endpoints=probe['endpoints']),
            stationarity_assumption_satisfied=False,
            note='Unqualified shifts have no tau and are excluded from LDS; not a proof of noninvertibility.')))
        rt.finish(tmp, c, dict(kind='xscore', seed=seed), elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'


def _clean(x):
    import math
    if isinstance(x, dict):
        return {str(k) if not isinstance(k, float) else f'{k:g}': _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
