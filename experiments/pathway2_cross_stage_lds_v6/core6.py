"""GPU work for cross-stage LDS v6: v5 r1 (5143843) extended along two prespecified axes.

A. Attack seeds 3 -> 6. The v5 r1 run supplies everything for attack seeds 5101-5103 (truth rows,
   adjoints V, h). v6 trains the attacks of three new seeds on the same table and GPU type and
   computes their J00, J10, V and h. Old and new seeds are commensurable only if the attack replays
   exactly, which smoke and every truth cell check against v5 r1.
B. Member-only h. h_member = sum over shadow-3 member (training) rows of (dp_r/dtheta)^T v_r, i.e.
   v with the nonmember rows set to zero. It needs no knowledge of the deleted subset, so L3_member
   = -sum_{i not in S_j} tau_i with tau_i = -hbar_member^T (H + gamma I)^{-1} g_i is a deployable
   score. It drops a real pathway (deletion also moves heldout predictions), so it is an alternative
   estimator, not the exact formula; the v5 r1 L1 decomposition suggested it (post hoc).

Everything else is v5 r1: stable-target table (copied, hash-checked), primary classes, omega,
gamma_A* (inherited, re-qualified on the new seeds), Stage-1 gamma grid, subsets, seeds.
"""
import os
from pathlib import Path
import shutil
import time
import numpy as np
import runtime as rt
import attack4 as A
from core3 import staged, exclusive
from core4 import exclusive_cell, _clean
from core5 import X5Session, gate_decision
from numerics4 import require_replay_close

VARIANTS = ('full', 'member')


class X6Session(X5Session):
    def __init__(self, c, freeze_to=None):
        super().__init__(c, freeze_to=freeze_to)
        self.v5 = Path(c['v5_out'])
        self.v5_hash = c['v5_config_sha256']
        self.old_k, self.new_k = list(c['old_attack_seeds']), list(c['new_attack_seeds'])
        self._new_j00 = {}

    # -------------------------------------------------------------- v5 inputs
    def v5_dir(self, rel, cell):
        d = self.v5 / rel
        rt.verify_hashes({str(d / 'complete.json'): self.c['input_sha256'][str(d / 'complete.json')]})
        rt.completed(d, self.v5_hash, cell)
        return d

    def member_rows(self, fixed):
        return fixed['membership'][fixed['shadow_id'] == A.AFFECTED] == 1

    # ------------------------------------------------------------------- smoke
    def smoke6(self, dest):
        """Copy the v5 r1 table, replay old attacks exactly, train the new attack seeds on every full
        baseline, gate them (CE quality gate, inherited gamma_A* must qualify), freeze the extended panel."""
        import torch
        dest = Path(dest)
        c = self.c
        src = self.v5_dir('smoke', dict(kind='xsmoke'))
        for f in ('fixed_rows.npz', 'frozen.json', 'gate.json'):
            shutil.copy2(src / f, dest / ('v5_' + f if f != 'fixed_rows.npz' else f))
        if rt.read(dest / 'designs.json') != rt.read(src / 'designs.json'):
            raise ValueError('Regenerated designs differ from the v5 r1 designs.json')
        with np.load(dest / 'fixed_rows.npz') as z:
            fixed = {k: z[k] for k in z.files}
        fr5 = rt.read(dest / 'v5_frozen.json')
        primary, gstar = fr5['qualified_classes'], fr5['attack_gamma_star']
        v5_base = {}
        for s in c['seeds']:
            for r in rt.read(self.v5_dir(f'base/seed{s}', dict(kind='base', seed=s)) / 'result.json')['rows']:
                v5_base[(s, r['attack_seed'], r['cls'])] = r['metrics']['cross_entropy']
        checks, rows = {}, []
        # 1) exact replay of the old attacks (one per primary class, first seed): same GPU type and code
        s0 = c['seeds'][0]
        p0 = self.v3_probs(self.full_name(s0))
        rep = {}
        for cls in primary:
            out, _ = self.condition(fixed, p0, self.old_k[0], cls)
            rep[str(cls)] = require_replay_close(f'Old attack replay, seed {s0} attack {self.old_k[0]} class {cls}',
                                                 out['metrics']['cross_entropy'], v5_base[(s0, self.old_k[0], cls)], c['j00_tol'])
        checks['old_attack_replay_max_abs'] = rep
        # 2) new attack seeds on every full baseline: metrics, Hessian spectrum, qualification at gamma_A*
        for s in c['seeds']:
            p3 = self.v3_probs(self.full_name(s))
            for k in self.new_k:
                for cls in c['classes']:
                    out, _ = self.condition(fixed, p3, k, cls, adjoint_gammas=c['attack_gammas'])
                    adj = out['adjoint']
                    per = adj['per_gamma']
                    rows.append(dict(seed=s, attack_seed=k, cls=cls, metrics=out['metrics'],
                                     qualified_gamma=A.qualified_gamma(per, c['attack_solve_tol']),
                                     hessian_eig_min=adj['hessian_eig_min'], hessian_eig_max=adj['hessian_eig_max'],
                                     train_grad_norm=adj['train_grad_norm'],
                                     per_gamma={g: {k2: v for k2, v in e.items() if k2 != 'v'} for g, e in per.items()}))
        rt.write(dest / 'attack_baselines_new_seeds.json', _clean(dict(rows=rows)))
        gate = gate_decision(rows, primary, c['j00_gate'])
        gate['gamma_star_qualified_on_new_seeds'] = all(r['qualified_gamma'][gstar] for r in rows if r['cls'] in primary)
        gate['old_attack_replay_max_abs'] = rep
        rt.write(dest / 'gate.json', _clean(gate))
        if not gate['gamma_star_qualified_on_new_seeds']:
            raise ValueError(f'Inherited gamma_A*={gstar} does not qualify for every new primary attack; stop and discuss')
        if not gate['passed']:
            rt.write(self.root / f'smoke_gate_failed_{dest.name.lstrip(".")}.json', _clean(gate))
            raise ValueError('CE quality gate failed on the new attack seeds; stop and discuss. '
                             + str({k: v['J00_max'] for k, v in gate['per_class'].items()}))
        frozen = dict(fr5, attack_seeds=self.old_k + self.new_k, old_attack_seeds=self.old_k, new_attack_seeds=self.new_k,
                      inherited_from=str(self.v5),
                      rule=fr5['rule'] + ' v6: classes, omega and gamma_A* inherited from v5 r1; gamma_A* re-qualified on the new seeds.')
        rt.write(dest / 'frozen.json', frozen)
        rt.write(dest / 'result.json', _clean(dict(checks=checks, gate_passed=gate['passed'])))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -------------------------------------------------------------------- base
    @exclusive
    def base6(self, seed, tag):
        """V (all attack gammas) and h / h_member at gamma_A* for all six attack seeds.
        Old seeds: V and h from v5 r1 (h replayed for one class), h_member computed. New seeds: everything."""
        import torch
        cell = dict(kind='base6', seed=seed)
        final, tmp = staged(self.root, Path('base') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, cell)
            return 'skipped'
        start = time.monotonic()
        c, fr, fixed = self.c, self.frozen(), self.fixed()
        gA, gstar = c['attack_gammas'], fr['attack_gamma_star']
        gi = gA.index(gstar)
        member = self.member_rows(fixed)
        d5 = self.v5_dir(f'base/seed{seed}', dict(kind='base', seed=seed))
        with np.load(d5 / 'v.npz') as z:
            V_old = z['v']
            if list(z['attack_seeds']) != self.old_k or list(z['attack_gammas']) != gA:
                raise ValueError('v5 r1 base arrays have a different attack-seed / gamma layout')
        with np.load(d5 / 'h.npz') as z:
            H_old = z['h']
        model = self.model(self.full_name(seed), seed)
        p3 = self.v3_probs(self.full_name(seed))
        nK, nC = len(self.old_k) + len(self.new_k), len(c['classes'])
        V = np.zeros((nK,) + V_old.shape[1:])
        V[:len(self.old_k)] = V_old
        H = np.zeros((nK, nC, H_old.shape[-1]))
        Hm = np.zeros_like(H)
        H[:len(self.old_k)] = H_old
        rows = []
        for ki, k in enumerate(self.new_k, start=len(self.old_k)):
            for ci, cls in enumerate(c['classes']):
                out, _ = self.condition(fixed, p3, k, cls, adjoint_gammas=gA)
                for gj, g in enumerate(gA):
                    V[ki, ci, gj] = out['v'][g]
                rows.append(dict(attack_seed=k, cls=cls, metrics=out['metrics'], adjoint=out['adjoint']))
            hk = self.h_vectors(model, fixed, {cls: V[ki, ci, gi] for ci, cls in enumerate(c['classes'])})
            H[ki] = torch.stack([hk[cls] for cls in c['classes']]).cpu().numpy()
        replay = None
        for ki, k in enumerate(self.old_k + self.new_k):
            hm = self.h_vectors(model, fixed, {cls: V[ki, ci, gi] * member[:, None] for ci, cls in enumerate(c['classes'])})
            Hm[ki] = torch.stack([hm[cls] for cls in c['classes']]).cpu().numpy()
            if ki == 0:   # replay v5 r1's full h for the first old seed and the first primary class
                cls = fr['qualified_classes'][0]
                ci = c['classes'].index(cls)
                hf = self.h_vectors(model, fixed, {cls: V[0, ci, gi]})[cls].cpu().numpy()
                replay = require_replay_close(f'h replay vs v5 r1, seed {seed} attack {k} class {cls}', hf, H_old[0, ci],
                                              1e-10 * max(1., float(np.abs(H_old[0, ci]).max())))
        np.savez_compressed(tmp / 'v.npz', v=V, attack_gammas=np.asarray(gA), classes=np.asarray(c['classes']),
                            attack_seeds=np.asarray(self.old_k + self.new_k))
        np.savez(tmp / 'h.npz', h=H, h_member=Hm, gamma_star=gstar)
        rt.write(tmp / 'result.json', _clean(dict(seed=seed, attack_gamma_star=gstar, rows=rows, h_replay_max_abs=replay,
                                                  member_rows=int(member.sum()))))
        rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'

    def base6_arrays(self, seed):
        cache = self.__dict__.setdefault('_base6', {})
        if seed not in cache:
            cache.clear()
            d = self.root / 'base' / f'seed{seed}'
            rt.completed(d, self.h, dict(kind='base6', seed=seed))
            with np.load(d / 'v.npz') as z:
                V = z['v']
            with np.load(d / 'h.npz') as z:
                H, Hm = z['h'], z['h_member']
            J00 = {(r['attack_seed'], r['cls']): r['metrics']['cross_entropy'] for r in rt.read(d / 'result.json')['rows']}
            cache[seed] = (V, H, Hm, J00)
        return cache[seed]

    def new_j00(self, seed, fixed, J00_base):
        if seed not in self._new_j00:
            p3 = self.v3_probs(self.full_name(seed))
            out = {}
            for k in self.new_k:
                for cls in self.c['classes']:
                    r, _ = self.condition(fixed, p3, k, cls)
                    ce = r['metrics']['cross_entropy']
                    require_replay_close(f'Pack J00 vs base, seed {seed} attack {k} class {cls}', ce, J00_base[(k, cls)],
                                         self.c['j00_tol'])
                    out[(k, cls)] = ce
            self._new_j00 = {seed: out}
        return self._new_j00[seed]

    # ------------------------------------------------------------------- truth
    @exclusive_cell
    def truth6(self, cell, tag):
        """Rows for all six attack seeds. Old seeds: J00 / J10 / L1 / L2-lin copied from v5 r1 (one J10
        replayed here); new seeds: trained here. Member-only L1 and L2-lin for every seed."""
        import torch
        final, tmp = staged(self.root, Path('truth') / cell['name'], tag)
        tcell = dict(cell, kind='truth6')
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, tcell)
            return 'skipped'
        start = time.monotonic()
        c, fixed, fr = self.c, self.fixed(), self.frozen()
        s, j = cell['seed'], cell['subset']
        V, H, Hm, J00_base = self.base6_arrays(s)
        J00_new = self.new_j00(s, fixed, J00_base)
        old = {(r['attack_seed'], r['cls']): r for r in rt.read(self.v5_dir(f'truth/{cell["name"]}', cell) / 'result.json')['rows']}
        p_full = self.v3_probs(self.full_name(s))
        p_sub = self.v3_probs(self.subset_name(j, s))
        model = self.model(self.full_name(s), s)
        dth = (self.flat_params(self.model(self.subset_name(j, s), s)) - self.flat_params(model)).cpu().numpy()
        dP = p_sub - p_full
        member = self.member_rows(fixed)
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        gA = c['attack_gammas']
        # exact replay of one old J10 (first old seed, first primary class) on this device
        k0, c0 = self.old_k[0], fr['qualified_classes'][0]
        out, _ = self.condition(fixed, p_sub, k0, c0)
        replay = require_replay_close(f'J10 replay vs v5 r1, {cell["name"]} attack {k0} class {c0}',
                                      out['metrics']['cross_entropy'], old[(k0, c0)]['J10'], c['j00_tol'])
        rows = []
        for ki, k in enumerate(self.old_k + self.new_k):
            for ci, cls in enumerate(c['classes']):
                sel = cls3 == cls
                if k in self.old_k:
                    o = old[(k, cls)]
                    r = dict(attack_seed=k, cls=cls, source='v5r1', J00=o['J00'], J10=o['J10'], L1=o['L1'], L2_lin=o['L2_lin'])
                else:
                    j10, _ = self.condition(fixed, p_sub, k, cls)
                    r = dict(attack_seed=k, cls=cls, source='v6', J00=J00_new[(k, cls)], J10=j10['metrics']['cross_entropy'],
                             L1={f'{g:g}': float((V[ki, ci, gi][sel] * dP[sel]).sum()) for gi, g in enumerate(gA)},
                             L2_lin=float(H[ki, ci] @ dth))
                r['L1_member'] = {f'{g:g}': float((V[ki, ci, gi][sel & member] * dP[sel & member]).sum()) for gi, g in enumerate(gA)}
                r['L2_lin_member'] = float(Hm[ki, ci] @ dth)
                if k in self.old_k:   # the copied v5 r1 L1 / L2-lin must equal what v6's arrays give
                    r['L1_check_abs'] = abs(float((V[ki, ci, gA.index(fr['attack_gamma_star'])][sel] * dP[sel]).sum())
                                            - o['L1'][f"{fr['attack_gamma_star']:g}"])
                    r['L2_lin_check_abs'] = abs(float(H[ki, ci] @ dth) - o['L2_lin'])
                rows.append(r)
        rt.write(tmp / 'result.json', _clean(dict(cell=tcell, rows=rows, j10_replay_abs=replay)))
        rt.finish(tmp, c, tcell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'

    # ------------------------------------------------------------------- score
    @exclusive
    def score6(self, seed, tag):
        """Multi-shift Stage-1 IF for two right-hand sides over all six attack seeds:
        hbar_full = sum_c omega_c mean_k h_kc and hbar_member (same with member-only h), plus proxies.
        Replays v5 r1's 3-seed hbar first."""
        import torch
        import solver
        from analysis_stats import spearman
        from imgdesign import matched_prediction
        cell = dict(kind='score6', seed=seed)
        final, tmp = staged(self.root, Path('score') / f'seed{seed}', tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, cell)
            return 'skipped'
        start = time.monotonic()
        c, fr = self.c, self.frozen()
        _, H, Hm, _ = self.base6_arrays(seed)
        w = [(ci, fr['omega'][str(cls)]) for ci, cls in enumerate(c['classes']) if cls in fr['qualified_classes']]
        hbar = {'full': sum(om * H[:, ci].mean(0) for ci, om in w), 'member': sum(om * Hm[:, ci].mean(0) for ci, om in w)}
        nold = len(self.old_k)
        with np.load(self.v5_dir(f'score/seed{seed}', dict(kind='xscore', seed=seed)) / 'scores.npz') as z:
            hbar_v5 = z['hbar']
        hbar_old = sum(om * H[:nold, ci].mean(0) for ci, om in w)
        replay = require_replay_close(f'3-seed hbar replay vs v5 r1, seed {seed}', hbar_old, hbar_v5,
                                      1e-10 * max(1., float(np.abs(hbar_v5).max())))
        model = self.model(self.full_name(seed), seed)
        G = self.image_gradients(model)
        hvp = self.score.make_hvp(model, self.ctx['X'], self.ctx['y'], self.ctx['train'], self.dev,
                                  c['weight_decay'], 0., c['batch'])
        probe = self.dc.lanczos_probe(hvp, G.shape[1], c['probe_steps'], 20261015 + seed, self.dev, torch.float64)
        arrays, out = dict(train_rows=self.ctx['train']), dict(seed=seed, hbar_replay_vs_v5_max_abs=replay, variants={})
        for v in VARIANTS:
            hv = torch.as_tensor(hbar[v], dtype=torch.float64, device=self.dev)
            proxy = -(G @ hv).cpu().numpy()
            arrays[f'{v}_hbar'] = hbar[v]
            arrays[f'{v}_proxy'] = proxy
            arrays[f'{v}_pred_proxy'] = matched_prediction(self.M, proxy)
            ms = solver.multishift_lanczos(hvp, hv, c['gammas'], c['lanczos_max_steps'], c['lanczos_est_tol'],
                                           dot=lambda a, b: float(a @ b))
            gates = []
            for g in c['gammas']:
                x = ms['x'].get(g)
                true = None if x is None else float((hvp(x) + g * x - hv).norm() / hv.norm())
                true = true if true is not None and np.isfinite(true) else None
                q = solver.qualify(g, true, ms['ritz_min'], solver.probe_screen(probe, g), c['cg_fail_tol'])
                q.update(estimated_rel_residual=ms['final_estimated_rel_residual'][g], spearman_if_vs_proxy_images=None)
                if q['qualified']:
                    tau = -(G @ x).cpu().numpy()
                    if not np.isfinite(tau).all():
                        q['qualified'] = False
                        q['reasons'].append('non-finite influence scores')
                    else:
                        arrays[f'{v}_tau_gamma{g:g}'] = tau
                        arrays[f'{v}_pred_gamma{g:g}'] = matched_prediction(self.M, tau)
                        q['spearman_if_vs_proxy_images'] = spearman(tau, proxy)
                gates.append(q)
            rt.csv_write(tmp / f'lanczos_history_{v}.csv', ms['history'])
            out['variants'][v] = dict(gammas=gates, hbar_norm=float(np.linalg.norm(hbar[v])),
                                      lanczos=dict(steps=ms['steps'], ritz_min=ms['ritz_min'], ritz_max=ms['ritz_max']))
        out['cos_hbar_full_member'] = float(hbar['full'] @ hbar['member'] /
                                            max(np.linalg.norm(hbar['full']) * np.linalg.norm(hbar['member']), 1e-300))
        out['random_probe'] = dict(steps=probe['steps'], endpoints=probe['endpoints'])
        np.savez_compressed(tmp / 'scores.npz', **arrays)
        rt.write(tmp / 'result.json', _clean(out))
        rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        return 'done'
