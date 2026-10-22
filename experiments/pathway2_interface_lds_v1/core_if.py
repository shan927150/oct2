"""GPU work for the interface LDS: Stage-1 attribution evaluated on the four class probabilities, without
the attack.

For Stage-1 seed s and subset j of the image LDS v3 design (32 subsets, 50% of shadow-3 training images
retained), the static damped influence function predicts the parameter change of deleting the subset's
held-out images,
    dtheta_hat_{s,j}(gamma) = (H_s + gamma I)^{-1} b_{s,j},   b_{s,j} = sum_{i deleted} g_{s,i},   g_i = grad CE_i / N,
and its first-order effect on the four class probabilities of every shadow-3 interface row,
    dp_hat_{s,j,r} = J_{p,r} dtheta_hat_{s,j}.
The logit tangent J_{z,r} dtheta_hat is stored; J_p = (diag p - p p^T) J_z is applied at collect time.
This is the Stage-1 factor of L3: L3_j = sum_r vbar_r . dp_hat_{j,r}, with vbar the attack adjoint of v6 r1.
The truth is the retrained change p_sub - p_full of the same rows (v3 run). No attack is trained here.

H, g, the gamma grid and the models are those of L3 in v6 r1. Also stored per seed: the proxy J b_j (no
Hessian), the true parameter change linearised J (theta_j - theta_full), and J (H + gamma I)^{-1} sum_i g_i for
the matched normalisation. All 33 right-hand sides of a seed share one Galerkin subspace (subspace.py).

Gates that tie this run to v6 r1 (same models, same H): vbar built from v6's adjoints reproduces v6's hbar;
the IF scores of hbar solved here reproduce v6's tau; hbar . x_j = b_j . x_hbar (symmetry); and
vbar . (J_p x_j) = hbar . x_j (forward / reverse identity), so sum_r vbar . dp_hat reproduces v6's L3.
"""
import os
from pathlib import Path
import shutil
import time
import numpy as np
import runtime as rt
from core3 import staged, exclusive
from core4 import XSession, _clean
from numerics4 import require_replay_close
from subspace import SharedGalerkin


def softmax_jvp(p0, dz):
    """J_p tangent from the logit tangent: dp = p0 * (dz - <p0, dz>)."""
    return p0 * (dz - (p0 * dz).sum(-1, keepdims=True))


def projected_hours(t_hvp, n_first, n_rhs_pilot, n_rhs, probe_steps, t_grad, t_jvp, n_jvp):
    """Conservative per-seed hours: the first right-hand side, then every other at the pilot mean."""
    hvps = n_first + n_rhs * float(np.mean(n_rhs_pilot)) + probe_steps + 12
    return (hvps * t_hvp + t_grad + n_jvp * t_jvp) / 3600., hvps


def choose_panel(seeds, per_seed_hours, budget, walltime_hours):
    """Prespecified, outcome-blind: all seeds if they fit the GPU budget, else the first three, else stop."""
    if per_seed_hours > walltime_hours:
        return None, f'projected {per_seed_hours:.1f} h per seed exceeds the solve walltime {walltime_hours:g} h'
    if len(seeds) * per_seed_hours <= budget:
        return list(seeds), f'all {len(seeds)} seeds fit the budget ({len(seeds) * per_seed_hours:.1f} <= {budget:g} GPU-h)'
    if 3 * per_seed_hours <= budget:
        return list(seeds[:3]), f'only the first three seeds fit the budget ({3 * per_seed_hours:.1f} <= {budget:g} GPU-h)'
    return None, f'even three seeds exceed the budget ({3 * per_seed_hours:.1f} > {budget:g} GPU-h)'


class ISession(XSession):
    def __init__(self, c, freeze_to=None):
        super().__init__(c, freeze_to=freeze_to)
        self.v6 = Path(c['v6_out'])
        self.v6_hash = c['v6_config_sha256']

    # ------------------------------------------------------------- v6 inputs
    def v6_dir(self, rel, cell):
        d = self.v6 / rel
        rt.verify_hashes({str(d / 'complete.json'): self.c['input_sha256'][str(d / 'complete.json')]})
        rt.completed(d, self.v6_hash, cell)
        return d

    def v6_frozen(self):
        d = self.v6_dir('smoke', dict(kind='xsmoke'))
        return rt.read(d / 'frozen.json')

    def vbar(self, seed):
        """vbar = sum_c omega_c mean_k V[k, c, gamma_A*]: the adjoint whose VJP is v6's hbar."""
        fr = self.v6_frozen()
        c = self.c
        d = self.v6_dir(f'base/seed{seed}', dict(kind='base6', seed=seed))
        with np.load(d / 'v.npz') as z:
            V = z['v']
            if (list(z['attack_seeds']) != list(fr['attack_seeds']) or list(z['classes']) != c['classes']
                    or [float(g) for g in z['attack_gammas']] != [float(g) for g in c['attack_gammas']]):
                raise ValueError('v6 adjoint layout differs from the frozen v6 panel')
        gi = c['attack_gammas'].index(fr['attack_gamma_star'])
        out = np.zeros(V.shape[-2:])
        for ci, cls in enumerate(c['classes']):
            if cls in fr['qualified_classes']:
                out += fr['omega'][str(cls)] * V[:, ci, gi].mean(0)
        return out

    def v6_scores(self, seed):
        d = self.v6_dir(f'score/seed{seed}', dict(kind='score6', seed=seed))
        with np.load(d / 'scores.npz') as z:
            arrays = {k: z[k] for k in z.files if k.startswith('full_')}
        res = rt.read(d / 'result.json')['variants']['full']['gammas']
        return arrays, {float(q['gamma']): bool(q['qualified']) for q in res}

    # ------------------------------------------------------------ primitives
    def logit_jvp(self, model, tangent):
        """J_z tangent for every shadow-3 interface row (fp64, numpy [rows, classes])."""
        import torch
        from torch.func import functional_call, jvp
        names = [n for n, _ in model.named_parameters()]
        base = {n: p.detach() for n, p in model.named_parameters()}
        tan = dict(zip(names, self.score.unflatten(tangent, [base[n] for n in names])))
        out, b = [], self.c['batch']
        for s in range(0, len(self.rows3), b):
            xb = torch.as_tensor(self.ctx['X'][self.rows3[s:s + b]], dtype=torch.float64, device=self.dev)
            _, dz = jvp(lambda P: functional_call(model, P, (xb,)), (base,), (tan,))
            out.append(dz.detach().cpu().numpy())
        dz = np.concatenate(out)
        if not np.isfinite(dz).all():
            raise FloatingPointError('Non-finite logit JVP')
        return dz

    def timed_hvp(self, model):
        import torch
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], self.dev,
                                  self.c['weight_decay'], 0., self.c['batch'])
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        times = []

        def mv(v):
            t0 = time.monotonic()
            out = hvp(v)
            sync()
            times.append(time.monotonic() - t0)
            return out
        return mv, hvp, times

    def deleted_sums(self, G):
        import torch
        Mdel = torch.as_tensor(1. - self.M, dtype=G.dtype, device=G.device)
        return Mdel @ G

    def seed_setup(self, seed):
        """Model, interface probabilities, per-image gradients, deleted sums, vbar and the hbar gate."""
        import torch
        model = self.model(self.full_name(seed), seed)
        p0, _ = self.predict(model)
        require_replay_close(f'Full-model interface vs v3, seed {seed}', p0, self.v3_probs(self.full_name(seed)),
                             self.c['v3_prediction_tol'])
        t0 = time.monotonic()
        G = self.image_gradients(model)
        t_grad = time.monotonic() - t0
        vbar = self.vbar(seed)
        h = self.score.shadow_h(model, self.ctx['X'][self.rows3], vbar, self.dev, self.c['batch']).cpu().numpy()
        arrays6, qual6 = self.v6_scores(seed)
        hbar6 = arrays6['full_hbar']
        rel = float(np.linalg.norm(h - hbar6) / max(np.linalg.norm(hbar6), 1e-300))
        if not rel <= 1e-10:
            raise ValueError(f'vbar from v6 adjoints does not reproduce v6 hbar for seed {seed}: rel {rel:.3e}')
        hbar = torch.as_tensor(hbar6, dtype=torch.float64, device=self.dev)
        return dict(model=model, p0=p0, G=G, B=self.deleted_sums(G), vbar=vbar, hbar=hbar, arrays6=arrays6, qual6=qual6,
                    hbar_rel=rel, t_grad=t_grad)

    def tols(self):
        g0 = self.c['primary_gamma']
        return [self.c['tol_main'] if g >= g0 else self.c['tol_small'] for g in self.c['gammas']]

    def tau_tol(self, g):
        """Agreement required with v6's tau (v6 solved to <= 1e-5): the qualification bound, widened in proportion
        to this run's own residual tolerance at that gamma."""
        return max(self.c['cg_fail_tol'], 100 * self.tols()[self.c['gammas'].index(g)])

    def tau_check(self, S, x_h):
        """Our IF scores of hbar against v6's, for every gamma v6 qualified."""
        from analysis_stats import spearman
        out = {}
        for g in self.c['gammas']:
            key = f'full_tau_gamma{g:g}'
            if not S['qual6'].get(float(g)) or key not in S['arrays6']:
                out[f'{g:g}'] = dict(status='v6 unqualified')
                continue
            tau = -(S['G'] @ x_h[g]).cpu().numpy()
            ref = S['arrays6'][key]
            out[f'{g:g}'] = dict(rel_l2=float(np.linalg.norm(tau - ref) / max(np.linalg.norm(ref), 1e-300)),
                                 spearman=spearman(tau, ref))
        return out

    def explicit_residual(self, hvp, x, b, g):
        return float((hvp(x) + g * x - b).norm() / b.norm())

    # ------------------------------------------------------------------ smoke
    def if_smoke(self, dest):
        """Designs, row identity, v3 replay, the v6 ties, a solver pilot on the first seed and the panel rule."""
        import torch
        dest = Path(dest)
        c, checks, timing = self.c, {}, {}
        if rt.read(dest / 'designs.json') != rt.read(Path(c['v3_out']) / 'designs.json'):
            raise ValueError('Regenerated designs differ from the v3 run designs.json')
        y = self.ctx['y'][self.rows3]
        np.savez_compressed(dest / 'rows.npz', raw_index=self.rows3, y=y, train_position=np.r_[np.arange(self.N),
                            -np.ones(len(self.rows3) - self.N, dtype=np.int64)], masks=self.M)
        checks['full_prediction_vs_v3_max_abs'] = {
            str(s): require_replay_close(f'Full-model interface vs v3, seed {s}', self.predict(self.model(self.full_name(s), s))[0],
                                         self.v3_probs(self.full_name(s)), c['v3_prediction_tol']) for s in c['seeds']}
        s0 = c['seeds'][0]
        S = self.seed_setup(s0)
        checks['vbar_reproduces_v6_hbar_rel'] = S['hbar_rel']
        timing['per_image_gradients_seconds'] = S['t_grad']
        mv, hvp, times = self.timed_hvp(S['model'])
        sg = SharedGalerkin(mv, S['G'].shape[1], c['max_dim'], torch.float64, self.dev)
        gam, tols = c['gammas'], self.tols()
        pilot, x, stored = [], {}, {}
        for name, b in [('hbar', S['hbar'])] + [(f'b{j}', S['B'][j]) for j in range(c['pilot_rhs'])]:
            out = sg.solve(b, gam, tols, c['max_new'])
            pilot.append(dict(rhs=name, new_vectors=out['new_vectors'], dim=out['dim'], converged=out['converged'],
                              max_rel_residual=max(out['rel_residual'].values())))
            x[name], stored[name] = out['x'], out['rel_residual']
        checks['pilot'] = pilot
        checks['tau_vs_v6'] = self.tau_check(S, x['hbar'])
        bad = {g: v for g, v in checks['tau_vs_v6'].items() if 'rel_l2' in v and not v['rel_l2'] <= self.tau_tol(float(g))}
        if bad:
            raise ValueError(f'IF scores of hbar do not reproduce v6 tau: {bad}')
        b0 = S['B'][0]
        sym = {}
        for g in gam:
            a1, a2 = float(S['hbar'] @ x['b0'][g]), float(b0 @ x['hbar'][g])
            sym[f'{g:g}'] = abs(a1 - a2) / max(abs(a1), abs(a2), 1e-300)
        checks['symmetry_hbar_xb_vs_b_xhbar_rel'] = sym
        g0 = c['primary_gamma']
        t0 = time.monotonic()
        dz = self.logit_jvp(S['model'], x['b0'][g0])
        timing['logit_jvp_seconds'] = time.monotonic() - t0
        lhs, rhs = float((S['vbar'] * softmax_jvp(S['p0'], dz)).sum()), float(S['hbar'] @ x['b0'][g0])
        checks['forward_reverse_identity_rel'] = abs(lhs - rhs) / max(abs(rhs), 1e-300)
        if checks['forward_reverse_identity_rel'] > c['adjoint_identity_rel_tol']:
            raise ValueError(f'vbar . (J_p x) differs from hbar . x: {lhs} vs {rhs}')
        explicit = {f'{g:g}': self.explicit_residual(hvp, x['b0'][g], b0, g) for g in gam}
        checks['explicit_residual_b0'] = dict(explicit=explicit, stored={f'{g:g}': stored['b0'][g] for g in gam})
        for g, t in zip(gam, tols):
            if not (explicit[f'{g:g}'] <= max(10 * t, 1e-9) or explicit[f'{g:g}'] <= 10 * stored['b0'][g]):
                raise ValueError(f'Explicit residual {explicit[f"{g:g}"]:.2e} at gamma {g:g} does not confirm the stored HQ')
        timing['hvp_seconds_median'] = float(np.median(times))
        n_rhs = c['design']['m'] + 1
        hours, hvps = projected_hours(timing['hvp_seconds_median'], pilot[0]['new_vectors'], [p['new_vectors'] for p in pilot[1:]],
                                      n_rhs, c['probe_steps'], timing['per_image_gradients_seconds'],
                                      timing['logit_jvp_seconds'], n_rhs * len(gam) + 2 * c['design']['m'])
        panel, why = choose_panel(c['seeds'], hours, c['budget_gpu_hours'], c['solve_hours'])
        frozen = dict(score_panel=panel, panel_rule=why, projected_hours_per_seed=hours, projected_hvps_per_seed=hvps,
                      primary_gamma=g0, gammas=gam, tolerances=dict(zip([f'{g:g}' for g in gam], tols)),
                      rule='Panel fixed from compute measurements only, before any interface prediction is compared with truth.')
        rt.write(dest / 'result.json', _clean(dict(checks=checks, timing=timing, solver=dict(dim=sg.m, matvecs=sg.matvecs,
                                                                                          ritz_min=sg.ritz_min()))))
        if panel is None:
            rt.write(self.root / f'smoke_budget_stop_{dest.name.lstrip(".")}.json', _clean(frozen))
            raise ValueError('Stop and discuss: ' + why)
        rt.write(dest / 'frozen.json', _clean(frozen))

    # ------------------------------------------------------------------ solve
    @exclusive
    def if_solve(self, seed, tag):
        """All 32 subsets (+ sum of all gradients) for one seed on one shared subspace; logit JVPs of every
        prediction on all 4343 interface rows; the v6 ties and per-gamma qualification."""
        import torch
        import solver
        cell = dict(kind='ifsolve', seed=seed)
        final, tmp = staged(self.root, Path('solve') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, cell)
            return 'skipped'
        start = time.monotonic()
        c = self.c
        gam, tols, g0 = c['gammas'], self.tols(), c['primary_gamma']
        S = self.seed_setup(seed)
        model, G, B = S['model'], S['G'], S['B']
        mv, hvp, times = self.timed_hvp(model)
        probe = self.dc.lanczos_probe(mv, G.shape[1], c['probe_steps'], 20261022 + seed, self.dev, torch.float64)
        sg = SharedGalerkin(mv, G.shape[1], c['max_dim'], torch.float64, self.dev)
        m, R = c['design']['m'], len(self.rows3)
        out_h = sg.solve(S['hbar'], gam, tols, c['max_new'])
        dz_if = np.zeros((m, len(gam), R, 4))
        rel = np.zeros((m + 2, len(gam)))
        rel[0] = [out_h['rel_residual'][g] for g in gam]
        stats = [dict(rhs='hbar', new_vectors=out_h['new_vectors'], converged=out_h['converged'])]
        sym, ident, spot = np.zeros((m, len(gam))), np.zeros(m), {}
        for j in range(m):
            out = sg.solve(B[j], gam, tols, c['max_new'])
            rel[j + 1] = [out['rel_residual'][g] for g in gam]
            stats.append(dict(rhs=f'b{j}', new_vectors=out['new_vectors'], converged=out['converged'],
                              stalled=out['stalled'], capped=out['capped']))
            for gi, g in enumerate(gam):
                dz_if[j, gi] = self.logit_jvp(model, out['x'][g])
                a1, a2 = float(S['hbar'] @ out['x'][g]), float(B[j] @ out_h['x'][g])
                sym[j, gi] = abs(a1 - a2) / max(abs(a1), abs(a2), 1e-300)
            gi0 = gam.index(g0)
            lhs, rhs = float((S['vbar'] * softmax_jvp(S['p0'], dz_if[j, gi0])).sum()), float(S['hbar'] @ out['x'][g0])
            ident[j] = abs(lhs - rhs) / max(abs(rhs), 1e-300)
            if j in (0, m - 1):
                spot[f'b{j}'] = {f'{g:g}': self.explicit_residual(hvp, out['x'][g], B[j], g) for g in (g0, min(gam))}
        gall = G.sum(0)
        out_a = sg.solve(gall, gam, tols, c['max_new'])
        rel[m + 1] = [out_a['rel_residual'][g] for g in gam]
        stats.append(dict(rhs='g_all', new_vectors=out_a['new_vectors'], converged=out_a['converged']))
        dz_gall = np.stack([self.logit_jvp(model, out_a['x'][g]) for g in gam])
        dz_proxy = np.stack([self.logit_jvp(model, B[j]) for j in range(m)])
        theta = self.flat_params(model)
        dz_true = np.stack([self.logit_jvp(model, self.flat_params(self.model(self.subset_name(j, seed), seed)) - theta)
                            for j in range(m)])
        if max(ident) > c['adjoint_identity_rel_tol']:
            raise ValueError(f'Forward / reverse identity fails for seed {seed}: {max(ident):.2e}')
        for name, r in spot.items():
            j = int(name[1:])
            for g, val in r.items():
                t = tols[gam.index(float(g))]
                if not val <= max(10 * t, 1e-9) and not val <= 10 * rel[j + 1, gam.index(float(g))]:
                    raise ValueError(f'Explicit residual {val:.2e} ({name}, gamma {g}) does not confirm the stored HQ')
        tau = self.tau_check(S, out_h['x'])
        ritz = sg.ritz_min()
        gates = []
        for gi, g in enumerate(gam):
            q = solver.qualify(g, float(rel[:, gi].max()), ritz, solver.probe_screen(probe, g), c['cg_fail_tol'])
            t = tau.get(f'{g:g}', {})
            if 'rel_l2' in t and not t['rel_l2'] <= self.tau_tol(g):
                q['qualified'] = False
                q['reasons'].append('IF scores of hbar do not reproduce v6 tau')
            q.update(max_symmetry_rel=float(sym[:, gi].max()), tolerance_met_all_rhs=bool((rel[:, gi] <= tols[gi]).all()))
            gates.append(q)
        np.savez_compressed(tmp / 'interface.npz', p0=S['p0'], dz_if=dz_if, dz_gall=dz_gall, dz_proxy=dz_proxy, dz_true=dz_true,
                            vbar=S['vbar'], tau_hbar=np.stack([-(G @ out_h['x'][g]).cpu().numpy() for g in gam]),
                            rel_residual=rel, gammas=np.asarray(gam))
        rt.write(tmp / 'result.json', _clean(dict(
            seed=seed, gammas=gates, tau_vs_v6=tau, vbar_reproduces_v6_hbar_rel=S['hbar_rel'],
            forward_reverse_identity_max_rel=float(ident.max()), explicit_residual_spot=spot,
            solver=dict(dim=sg.m, matvecs=sg.matvecs, ritz_min=ritz, per_rhs=stats, probe_endpoints=probe['endpoints']),
            timing=dict(hvp_seconds_median=float(np.median(times)), per_image_gradients_seconds=S['t_grad'],
                        elapsed_seconds=time.monotonic() - start))))
        rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'
