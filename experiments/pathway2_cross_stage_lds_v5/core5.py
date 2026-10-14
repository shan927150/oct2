"""GPU work for cross-stage LDS v5 (revision r1, after the 2026-10-14 review).

v4 (cross_stage_lds_v4_stable_20261013T044559507358835Z) used five stable-recipe shadows and the
legacy target. The attack (trained on shadows only) reached AUC 0.90 on target queries but CE
0.95-1.50 on the primary classes. v5 retrains one CNN, the target, with the same stable recipe on
its original split. The target is query-side only: it never enters attack training, so attack
parameters, training gradients and Hessians are unchanged; q, v, h, L1, L2-lin and L3 change.

What this module adds on top of the v4 session (imported byte-for-byte):
- Background shadows 0/1/2/4 are reused from the v4 run after hash checks and an fp64 replay of
  their interface; only the target is trained.
- Three query sets on the same attack table: stable target (primary), legacy target through the
  v4 path (q_x_legacy, must reproduce v4's fixed_rows exactly) and the legacy checkpoint through
  the fp64 path of the stable target (q_x_legacy_fp64, isolates the precision path). Every
  trained attack is evaluated on all three, so old and new target are paired by construction.
- Smoke: a prespecified CE quality gate (not a calibration test), the exact forward/reverse
  identity, per-class / per-membership p_true profiles, prior-CE baselines.
- Truth cell (overrides v4): v4 fields plus legacy-target CE, the projection-only control
  J(A(Pi(P0))), and an L2-retrain variant that linearizes logits instead of probabilities.
- Hybrid cells: deleted-only, retained-only, nonmember-only and deleted+nonmember, primary classes.
- Interpolation panel: P(t) = P0 + t (Pj - P0) on a fixed small panel, J(t) - J(0) against t * L1.
"""
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import runtime as rt
import attack4 as A
from core3 import staged
from core4 import XSession, exclusive_cell, _clean
from numerics4 import require_replay_close

LN2 = math.log(2.)
HYBRIDS = {'deleted': ('deleted',), 'retained': ('retained',), 'nonmember': ('nonmember',),
           'deleted_nonmember': ('deleted', 'nonmember')}
ALT_TARGETS = ('legacy', 'legacy_fp64')
PROJ_FLOOR = 1e-6   # same floor as linearized_predictions (07_cross_stage_score_ladder.py)


# ----------------------------------------------------------------------------- pure helpers
def calibration(prob, y, bins=10):
    """Binary probability quality of the member probability: ECE (equal-width bins), confidently-wrong rate."""
    prob, y = np.asarray(prob, dtype=np.float64), np.asarray(y)
    edges = np.linspace(0., 1., bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = sum(abs(prob[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    wrong = np.where(y == 1, 1. - prob, prob)
    return dict(ece=float(ece), confident_wrong_rate=float((wrong >= .99).mean()),
                mean_member_prob=float(prob.mean()), member_fraction=float((y == 1).mean()))


def prior_ce(y):
    """CE of the constant prediction equal to the member fraction (the no-information baseline)."""
    pi = float(np.mean(np.asarray(y) == 1))
    return 0. if pi in (0., 1.) else float(-(pi * math.log(pi) + (1 - pi) * math.log(1 - pi)))


def p_true_profile(x, classes, membership, by_class=True):
    """Saturation of the true-class probability, overall and per OCT class, per membership."""
    x, classes, membership = np.asarray(x), np.asarray(classes), np.asarray(membership)
    pt = x[np.arange(len(x)), classes]

    def block(sel):
        out = {}
        for name, msel in (('member', sel & (membership == 1)), ('nonmember', sel & (membership == 0))):
            v = pt[msel]
            out[name] = dict(n=int(msel.sum()), mean=float(v.mean()) if len(v) else None,
                             frac_below_0_99=float((v < .99).mean()) if len(v) else None,
                             frac_below_0_999=float((v < .999).mean()) if len(v) else None,
                             quantiles=[float(q) for q in np.quantile(v, [.01, .1, .5])] if len(v) else None)
        return out
    out = dict(all=block(np.ones(len(x), dtype=bool)))
    if by_class:
        out['by_class'] = {str(int(c)): block(classes == c) for c in np.unique(classes)}
    return out


def gate_decision(rows, primary, threshold):
    """Prespecified CE quality gate: every (seed, attack seed) J00 of every primary class < threshold.
    It screens for confidently wrong attacks; it is not a calibration test."""
    per = {}
    for cls in primary:
        sel = [r for r in rows if r['cls'] == cls]
        if not sel:
            raise ValueError(f'No full-baseline attack for primary class {cls}')
        j = [r['metrics']['cross_entropy'] for r in sel]
        mean = lambda key: float(np.mean([r['metrics'][key] for r in sel])) if all(key in r['metrics'] for r in sel) else None
        per[str(cls)] = dict(J00_max=float(max(j)), J00_mean=float(np.mean(j)), n=len(j), auc_mean=mean('auc'),
                             brier_mean=mean('brier'), ece_mean=mean('ece'), confident_wrong_mean=mean('confident_wrong_rate'),
                             prior_ce=sel[0]['metrics'].get('prior_ce'),
                             legacy_J00_mean=mean('cross_entropy_legacy'), legacy_fp64_J00_mean=mean('cross_entropy_legacy_fp64'),
                             passed=bool(max(j) < threshold))
    return dict(threshold=threshold, name='CE quality gate',
                rule='every primary-class J00 on every full baseline < threshold; a quality screen, not a calibration test',
                per_class=per, passed=all(v['passed'] for v in per.values()))


def project(p, floor=PROJ_FLOOR):
    """The projection used by L2-retrain: clamp to >= floor, renormalise (not a Euclidean projection)."""
    q = np.clip(p, floor, None)
    return q / q.sum(1, keepdims=True)


def projection_stats(raw):
    proj = project(raw)
    return dict(rows_negative=float((raw.min(1) < 0).mean()), rows_modified=float((np.abs(proj - raw) > 1e-12).any(1).mean()),
                mean_l1_change=float(np.abs(proj - raw).sum(1).mean()))


class X5Session(XSession):
    def __init__(self, c, freeze_to=None):
        super().__init__(c, freeze_to=freeze_to)
        self._j00_full = {}
        self._proj0 = {}

    # ------------------------------------------------------------------ attacks
    def condition(self, fixed, p3, k, cls, adjoint_gammas=None):
        """v4 condition (stable-target queries, v4 adjoint) plus the same attack evaluated on the
        alternative target query sets and quality metrics. CE/AUC of the primary target are unchanged."""
        out, model = super().condition(fixed, p3, k, cls, adjoint_gammas=adjoint_gammas)
        te = fixed['q_classes'] == cls
        qm = fixed['q_membership'][te]
        prob, _ = self.pilot.predict_attack_outputs(model, fixed['q_x'][te])
        out['metrics'].update(calibration(prob, qm), prior_ce=prior_ce(qm))
        for key in ALT_TARGETS:
            if f'q_x_{key}' in fixed:
                m = A.evaluate(self.pilot, model, fixed[f'q_x_{key}'][te], qm)
                out['metrics'].update({f'{name}_{key}': m[name] for name in ('cross_entropy', 'auc', 'brier', 'balanced_accuracy')})
                prob, _ = self.pilot.predict_attack_outputs(model, fixed[f'q_x_{key}'][te])
                out['metrics'].update({f'{name}_{key}': v for name, v in calibration(prob, qm).items()})
        return out, model

    # ----------------------------------------------------- background (reuse)
    def _interface_fp64(self, model, rows):
        import torch
        model.double().eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(rows), self.c['batch']):
                x = torch.as_tensor(self.ctx['X'][rows[i:i + self.c['batch']]], device=self.dev, dtype=torch.float64)
                probs.append(model(x).softmax(1).cpu().numpy())
        return np.concatenate(probs)

    def reuse_background_shadow(self, sid, tag):
        """Copy a v4 background shadow after hash checks; replay its interface from the checkpoint."""
        import torch
        with rt.lock(self.root / '.locks' / f'background_shadow_{sid}.lock'):
            final, tmp = staged(self.root, Path('background') / f'shadow{sid}', tag)
            cell = dict(kind='background', shadow=sid)
            if final.exists():
                shutil.rmtree(tmp)
                rt.completed(final, self.h, cell)
                return 'skipped'
            start = time.monotonic()
            src = Path(self.c['background_source'][str(sid)])
            files = ('final.pt', 'interface.npz', 'result.json', 'complete.json')
            rt.verify_hashes({str(src / f): self.c['input_sha256'][str(src / f)] for f in files})
            for f in ('final.pt', 'interface.npz', 'result.json'):
                shutil.copy2(src / f, tmp / f)
            state = torch.load(tmp / 'final.pt', map_location='cpu', weights_only=False)
            seed = self.c['fixed_shadow_seed'] + sid
            if state['seed'] != seed or state['epochs'] != self.c['epochs'] or state['recipe'] != self.c['recipe']:
                raise ValueError(f'Reused shadow{sid} checkpoint has a different seed/epochs/recipe')
            split = rt.read(Path(self.c['full']) / 'splits/fresh_patient_split.json')['shadow_models'][sid]
            rows = np.r_[np.asarray(split['train_idx'], dtype=np.int64), np.asarray(split['test_idx'], dtype=np.int64)]
            with np.load(tmp / 'interface.npz') as z:
                if not np.array_equal(z['raw_index'], rows):
                    raise ValueError(f'Reused shadow{sid} interface rows differ from the split')
                stored = np.asarray(z['raw'])
            replay = self._interface_fp64(self.initialize(seed, state['raw']), rows)
            err = require_replay_close(f'Reused shadow{sid} interface replay', replay, stored, self.c['v3_prediction_tol'])
            rt.write(tmp / 'provenance.json', dict(source=str(src), source_sha256={f: rt.sha(src / f) for f in files},
                                                   interface_replay_max_abs=err, rule='hash-checked copy + fp64 replay'))
            rt.finish(tmp, self.c, cell, elapsed_seconds=time.monotonic() - start)
            os.rename(tmp, final)
            return 'reused'

    # ------------------------------------------------------------ stable target
    def background_target(self, tag):
        """Train the target once with the stable recipe on its original split (legacy target seed)."""
        import torch
        with rt.lock(self.root / '.locks' / 'background_target.lock'):
            final, tmp = staged(self.root, Path('background') / 'target', tag)
            cell = dict(kind='target')
            if final.exists():
                shutil.rmtree(tmp)
                rt.completed(final, self.h, cell)
                return 'skipped'
            start = time.monotonic()
            split = rt.read(Path(self.c['full']) / 'splits/fresh_patient_split.json')
            train = np.asarray(split['target_train_idx'], dtype=np.int64)
            rows = np.r_[train, np.asarray(split['target_test_idx'], dtype=np.int64)]
            seed = self.c['target_seed']
            orders = self.pilot.make_epoch_orders(train, self.c['epochs'], seed + 700000)
            model, ema, _, curve = self.stable.train(self.initialize(seed), self.ctx['X'], self.ctx['y'], orders,
                                                    np.array([], dtype=np.int64), 1., **self.c['recipe'])
            torch.save(dict(raw=model.state_dict(), ema=ema.state_dict(), seed=seed,
                            epochs=self.c['epochs'], recipe=self.c['recipe']), tmp / 'final.pt')
            rt.csv_write(tmp / 'curve.csv', curve)
            p = self._interface_fp64(model, rows)
            np.savez_compressed(tmp / 'interface.npz', raw=p, raw_index=rows)
            y = self.ctx['y']
            member = np.r_[np.ones(len(train), dtype=np.int64), np.zeros(len(rows) - len(train), dtype=np.int64)]
            rt.write(tmp / 'result.json', _clean(dict(
                seed=seed, epochs=self.c['epochs'], dropout=0., train_images=len(train), test_images=len(rows) - len(train),
                train_accuracy=float((p[:len(train)].argmax(1) == y[train]).mean()),
                test_accuracy=float((p[len(train):].argmax(1) == y[rows[len(train):]]).mean()),
                p_true=p_true_profile(p, y[rows], member))))
            rt.finish(tmp, self.c, cell, elapsed_seconds=time.monotonic() - start)
            os.rename(tmp, final)
            return 'done'

    def build_fixed(self):
        """v4 attack table; primary queries from the stable target; legacy queries kept on two paths."""
        fixed = super().build_fixed()                       # q_x = legacy target through the v4 path
        fixed['q_x_legacy'] = fixed['q_x'].copy()
        n_in, n_cls = self.ctx['X'].shape[1], int(self.ctx['y'].max()) + 1
        legacy = self.pilot.load_model(Path(self.c['full']) / 'checkpoints/target_fixed.pt', n_in, 128, n_cls)
        fixed['q_x_legacy_fp64'] = self._interface_fp64(legacy, fixed['q_raw_index']).astype(np.float32)
        d = self.root / 'background' / 'target'
        rt.completed(d, self.h, dict(kind='target'))
        with np.load(d / 'interface.npz') as z:
            p, rows = np.asarray(z['raw']), np.asarray(z['raw_index'])
        if not np.array_equal(rows, fixed['q_raw_index']):
            raise ValueError('Stable-target query rows differ from the legacy target query rows')
        if not np.isfinite(p).all():
            raise FloatingPointError('Non-finite stable-target probabilities')
        fixed['q_x'] = p.astype(np.float32)
        return fixed

    # ------------------------------------------------------------------- smoke
    def xsmoke(self, dest):
        """v4 smoke on the v5 table, then: v4 table replay, CE quality gate, exact JVP identity, profiles."""
        dest = Path(dest)
        super().xsmoke(dest)
        c = self.c
        v4_rows = {(r['seed'], r['attack_seed'], r['cls']): r for r in rt.read(dest / 'attack_baselines.json')['rows']}
        frozen = rt.read(dest / 'frozen.json')
        with np.load(dest / 'fixed_rows.npz') as z:
            fx = {k: z[k] for k in z.files}
        # The v4 smoke evaluates its baselines on the primary queries only; retrain the same attacks
        # (deterministic, replay-checked) through condition() to score them on every target.
        rows = []
        for s in c['seeds']:
            p3 = self.v3_probs(self.full_name(s))
            for k in c['attack_seeds']:
                for cls in c['classes']:
                    out, _ = self.condition(fx, p3, k, cls)
                    require_replay_close(f'Smoke baseline replay, seed {s} attack {k} class {cls}', out['metrics']['cross_entropy'],
                                         v4_rows[(s, k, cls)]['metrics']['cross_entropy'], c['j00_tol'])
                    rows.append(dict(seed=s, attack_seed=k, cls=cls, metrics=out['metrics']))
        rt.write(dest / 'attack_baselines_all_targets.json', _clean(dict(rows=rows)))
        gate = gate_decision(rows, frozen['qualified_classes'], c['j00_gate'])
        gate['v4_table_replay'] = self.v4_table_replay(fx)
        sid = fx['shadow_id']
        prof = {f'shadow{s}': p_true_profile(fx['x'][sid == s], fx['classes'][sid == s], fx['membership'][sid == s])
                for s in sorted(set(sid.tolist())) if s != A.AFFECTED}
        m3, c3 = fx['membership'][sid == A.AFFECTED], fx['classes'][sid == A.AFFECTED]
        for s in c['seeds']:
            prof[f'shadow3_full_seed{s}'] = p_true_profile(self.v3_probs(self.full_name(s)), c3, m3)
        for key in ('q_x', 'q_x_legacy', 'q_x_legacy_fp64'):
            prof[{'q_x': 'target_stable', 'q_x_legacy': 'target_legacy', 'q_x_legacy_fp64': 'target_legacy_fp64'}[key]] = \
                p_true_profile(fx[key], fx['q_classes'], fx['q_membership'])
        gate['p_true_profile'] = prof
        gate['legacy_precision_path_max_abs'] = float(np.abs(fx['q_x_legacy'].astype(np.float64)
                                                             - fx['q_x_legacy_fp64'].astype(np.float64)).max())
        gate['jvp_reverse_identity'] = self.jvp_identity(fx, frozen)
        rt.write(dest / 'gate.json', _clean(gate))
        if gate['v4_table_replay'].get('passed') is False:
            raise ValueError(f'v4 attack table / legacy queries not reproduced: {gate["v4_table_replay"]}')
        if gate['jvp_reverse_identity']['relative_error'] > c['adjoint_identity_rel_tol']:
            raise ValueError(f'Forward JVP and reverse-mode h disagree: {gate["jvp_reverse_identity"]}')
        if not gate['passed']:
            rt.write(self.root / f'smoke_gate_failed_{dest.name.lstrip(".")}.json', _clean(gate))
            raise ValueError('CE quality gate failed (primary-class J00 not below threshold); stop and discuss. '
                             + str({k: v['J00_max'] for k, v in gate['per_class'].items()}))

    def v4_table_replay(self, fx):
        """The shadow rows and legacy queries must equal the v4 run's fixed_rows exactly."""
        src = self.c.get('v4_out')
        if not src:
            return dict(checked=False)
        path = Path(src) / 'smoke/fixed_rows.npz'
        rt.verify_hashes({str(path): self.c['input_sha256'][str(path)]})
        with np.load(path) as z:
            v4 = {k: z[k] for k in z.files}
        sid = fx['shadow_id'] != A.AFFECTED
        checks = dict(x_background=bool(np.array_equal(fx['x'][sid], v4['x'][v4['shadow_id'] != A.AFFECTED])),
                      membership=bool(np.array_equal(fx['membership'], v4['membership'])),
                      classes=bool(np.array_equal(fx['classes'], v4['classes'])),
                      q_x_legacy=bool(np.array_equal(fx['q_x_legacy'], v4['q_x'])),
                      q_rows=bool(np.array_equal(fx['q_raw_index'], v4['q_raw_index'])))
        return dict(checked=True, source=str(path), **checks, passed=all(checks.values()))

    def jvp_identity(self, fixed, frozen):
        """Exact (no finite differences): sum_r v_r . (J dtheta)_r == h . dtheta, first seed and subset."""
        c = self.c
        s, k, gstar = c['seeds'][0], c['attack_seeds'][0], frozen['attack_gamma_star']
        p3 = self.v3_probs(self.full_name(s))
        v_by_class, v_all = {}, np.zeros((len(self.rows3), 4))
        for cls in frozen['qualified_classes']:
            out, _ = self.condition(fixed, p3, k, cls, adjoint_gammas=[gstar])
            v_by_class[cls] = out['v'][gstar] * frozen['omega'][str(cls)]
            v_all += v_by_class[cls]
        model = self.model(self.full_name(s), s)
        h = sum(self.h_vectors(model, fixed, v_by_class).values())
        dtheta = self.flat_params(self.model(self.subset_name(0, s), s)) - self.flat_params(model)
        lin = self.score.linearized_predictions(model, self.ctx['X'][self.rows3], dtheta, self.dev, c['batch'])
        p0, _ = self.predict(model)
        terms = (v_all * (lin['raw'] - p0)).sum(1)
        fwd, rev = float(terms.sum()), float(h @ dtheta)
        rel = abs(fwd - rev) / max(abs(rev), float(np.abs(terms).sum()), 1e-300)
        return dict(h_dot_dtheta=rev, v_dot_jvp=fwd, relative_error=rel, tolerance=c['adjoint_identity_rel_tol'])

    # -------------------------------------------------------- per-seed caches
    def baseline_metrics(self, seed, fixed):
        """Pack-local full-baseline attacks (same device as J10), replay-checked against base, both targets."""
        if seed not in self._j00_full:
            base = {(r['attack_seed'], r['cls']): r['metrics'] for r in rt.read(self.root / 'base' / f'seed{seed}/result.json')['rows']}
            p3 = self.v3_probs(self.full_name(seed))
            out = {}
            for k in self.c['attack_seeds']:
                for cls in self.c['classes']:
                    r, _ = self.condition(fixed, p3, k, cls)
                    for key in ('cross_entropy', 'cross_entropy_legacy'):
                        require_replay_close(f'Pack {key} vs base, seed {seed} attack {k} class {cls}',
                                             r['metrics'][key], base[(k, cls)][key], self.c['j00_tol'])
                    out[(k, cls)] = r['metrics']
            self._j00_full = {seed: out}
        return self._j00_full[seed]

    def projection_baseline(self, seed, fixed):
        """J(A_train(Pi(P0))): the attack retrained after only the L2-retrain projection, zero parameter step."""
        if seed not in self._proj0:
            p0 = project(self.v3_probs(self.full_name(seed)))
            out = {}
            for k in self.c['attack_seeds']:
                for cls in self.c['classes']:
                    r, _ = self.condition(fixed, p0, k, cls)
                    out[(k, cls)] = r['metrics']
            self._proj0 = {seed: out}
        return self._proj0[seed]

    def linearized_logits(self, model, dtheta):
        """softmax(z0 + J_z dtheta): a first-order step in logit space, always a valid probability vector."""
        import torch
        from torch.func import functional_call, jvp
        model.eval()
        names = [n for n, _ in model.named_parameters()]
        base = {n: p.detach() for n, p in model.named_parameters()}
        tangent = dict(zip(names, self.score.unflatten(dtheta, [base[n] for n in names])))
        X, rows, out = self.ctx['X'], self.rows3, []
        for s in range(0, len(rows), self.c['batch']):
            xb = torch.as_tensor(X[rows[s:s + self.c['batch']]], dtype=torch.float64, device=self.dev)
            z0, dz = jvp(lambda P: functional_call(model, P, (xb,)), (base,), (tangent,))
            out.append(torch.softmax(z0 + dz, 1).detach().cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------------- truth
    @exclusive_cell
    def truth_cell(self, cell, tag):
        """v4 truth cell (same fields and meaning) plus legacy-target CE, the projection-only control
        and a logit-linearized L2-retrain. One attack per condition, evaluated on every target."""
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
        B = self.baseline_metrics(s, fixed)
        P0 = self.projection_baseline(s, fixed)
        p_full = self.v3_probs(self.full_name(s))
        p_sub = self.v3_probs(self.subset_name(j, s))
        model = self.model(self.full_name(s), s)
        theta = self.flat_params(model)
        dtheta = self.flat_params(self.model(self.subset_name(j, s), s)) - theta
        lin = self.score.linearized_predictions(model, self.ctx['X'][self.rows3], dtheta, self.dev, c['batch'])
        p_hat, p_logit = lin['projected'], self.linearized_logits(model, dtheta)
        dP = p_sub - p_full
        dth = dtheta.cpu().numpy()
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        rows = []
        for ki, k in enumerate(c['attack_seeds']):
            for ci, cls in enumerate(c['classes']):
                sel = cls3 == cls
                j10, _ = self.condition(fixed, p_sub, k, cls)
                jhat, _ = self.condition(fixed, p_hat, k, cls)
                jlog, _ = self.condition(fixed, p_logit, k, cls)
                b, pr = B[(k, cls)], P0[(k, cls)]
                rows.append(dict(attack_seed=k, cls=cls, J00=b['cross_entropy'], J10=j10['metrics']['cross_entropy'],
                                 J00_minus_base=b['cross_entropy'] - J00_base[(k, cls)],
                                 J_L2_retrain=jhat['metrics']['cross_entropy'],
                                 auc10=j10['metrics']['auc'], auc_L2_retrain=jhat['metrics']['auc'],
                                 L1={f'{g:g}': float((V[ki, ci, gi][sel] * dP[sel]).sum()) for gi, g in enumerate(c['attack_gammas'])},
                                 L2_lin=float(H[ki, ci] @ dth),
                                 J_proj0=pr['cross_entropy'], J_L2_logit=jlog['metrics']['cross_entropy'],
                                 J00_legacy=b['cross_entropy_legacy'], J10_legacy=j10['metrics']['cross_entropy_legacy'],
                                 J_L2_retrain_legacy=jhat['metrics']['cross_entropy_legacy'],
                                 J_proj0_legacy=pr['cross_entropy_legacy'], J_L2_logit_legacy=jlog['metrics']['cross_entropy_legacy'],
                                 auc00_legacy=b['auc_legacy'], auc10_legacy=j10['metrics']['auc_legacy'],
                                 brier00=b['brier'], brier10=j10['metrics']['brier']))
        member = fixed['membership'][fixed['shadow_id'] == A.AFFECTED] == 1
        deleted = np.r_[self.M[j] == 0, np.zeros(len(self.rows3) - self.N, dtype=bool)]
        groups = dict(deleted=deleted, retained=member & ~deleted, nonmember=~member)
        drift = {g: dict(js_true=float(A.js(p_full[msk], p_sub[msk]).mean()),
                         js_linearized=float(A.js(p_full[msk], p_hat[msk]).mean()),
                         js_true_vs_linearized=float(A.js(p_sub[msk], p_hat[msk]).mean()),
                         js_logit_linearized=float(A.js(p_full[msk], p_logit[msk]).mean()),
                         js_true_vs_logit_linearized=float(A.js(p_sub[msk], p_logit[msk]).mean())) for g, msk in groups.items()}
        rt.write(tmp / 'result.json', _clean(dict(cell=cell, rows=rows, dtheta_norm=float(np.linalg.norm(dth)),
            jvp=dict(row_clip_rate=lin['row_clip_rate'], mean_negative_mass=lin['mean_negative_mass'],
                     mean_projection_l1=lin['mean_projection_l1'], projection=projection_stats(lin['raw']),
                     projection_at_P0=projection_stats(p_full)), drift=drift)))
        rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'

    # ------------------------------------------------------------------ hybrid
    @exclusive_cell
    def hybrid_cell(self, cell, tag):
        """J with only some shadow-3 row groups at their subset values (primary classes), and L1 by group."""
        import torch
        hcell = dict(name=cell['name'], kind='hybrid', subset=cell['subset'], seed=cell['seed'])
        final, tmp = staged(self.root, Path('hybrid') / cell['name'], tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, hcell)
            return 'skipped'
        start = time.monotonic()
        c, fixed, fr = self.c, self.fixed(), self.frozen()
        s, j = cell['seed'], cell['subset']
        V, _, _ = self.base_arrays(s)
        B = self.baseline_metrics(s, fixed)
        p_full = self.v3_probs(self.full_name(s))
        p_sub = self.v3_probs(self.subset_name(j, s))
        dP = p_sub - p_full
        member = fixed['membership'][fixed['shadow_id'] == A.AFFECTED] == 1
        deleted = np.r_[self.M[j] == 0, np.zeros(len(self.rows3) - self.N, dtype=bool)]
        groups = dict(deleted=deleted, retained=member & ~deleted, nonmember=~member)
        if (groups['deleted'] & ~member).any() or groups['deleted'].sum() + groups['retained'].sum() != member.sum():
            raise ValueError('Row groups do not partition the shadow-3 interface')
        tables = {}
        for name, parts in HYBRIDS.items():
            p = p_full.copy()
            for g in parts:
                p[groups[g]] = p_sub[groups[g]]
            tables[name] = p
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        rows = []
        for ki, k in enumerate(c['attack_seeds']):
            for ci, cls in enumerate(c['classes']):
                sel = cls3 == cls
                r = dict(attack_seed=k, cls=cls, J00=B[(k, cls)]['cross_entropy'], J00_legacy=B[(k, cls)]['cross_entropy_legacy'],
                         L1_by_group={g: {f'{gA:g}': float((V[ki, ci, gi][sel & msk] * dP[sel & msk]).sum())
                                          for gi, gA in enumerate(c['attack_gammas'])} for g, msk in groups.items()})
                if cls in fr['qualified_classes']:
                    r['J'], r['J_legacy'] = {}, {}
                    for name, p in tables.items():
                        out, _ = self.condition(fixed, p, k, cls)
                        r['J'][name] = out['metrics']['cross_entropy']
                        r['J_legacy'][name] = out['metrics']['cross_entropy_legacy']
                rows.append(r)
        rt.write(tmp / 'result.json', _clean(dict(cell=hcell, rows=rows, hybrids={k: list(v) for k, v in HYBRIDS.items()},
                 group_sizes={g: int(m.sum()) for g, m in groups.items()})))
        rt.finish(tmp, c, hcell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'

    # ----------------------------------------------------------- interpolation
    def interp_panel(self, tag):
        """P(t) = P0 + t (Pj - P0) on the fixed panel; same attack init and batch order for every t.
        Compares J(t) - J(0) with t * L1 for both targets (legacy adjoint computed here at P0)."""
        import torch
        cell = dict(kind='interp')
        with rt.lock(self.root / '.locks' / 'interp.lock'):
            final, tmp = staged(self.root, Path('interp'), tag)
            if final.exists():
                shutil.rmtree(tmp)
                rt.completed(final, self.h, cell)
                return 'skipped'
            start = time.monotonic()
            c, fixed, fr = self.c, self.fixed(), self.frozen()
            P, gA = c['interp'], c['attack_gammas']
            cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
            n_block = int((fixed['shadow_id'] == A.AFFECTED).sum())
            rows = []
            for s in P['seeds']:
                V, _, J00_base = self.base_arrays(s)
                p0 = self.v3_probs(self.full_name(s))
                x0 = A.assemble(fixed, p0)
                for ki, k in enumerate(c['attack_seeds']):
                    for cls in fr['qualified_classes']:
                        ci = c['classes'].index(cls)
                        b, model0 = self.condition(fixed, p0, k, cls)
                        require_replay_close(f'Interp J(0) vs base, seed {s} attack {k} class {cls}',
                                             b['metrics']['cross_entropy'], J00_base[(k, cls)], c['j00_tol'])
                        tr, aff, where = A.class_rows(fixed, cls)
                        te = fixed['q_classes'] == cls
                        adj = A.adjoint(model0, x0[tr], fixed['membership'][tr], fixed['q_x_legacy'][te],
                                        fixed['q_membership'][te], gA, self.dev)
                        v_leg = {}
                        for g, e in adj['per_gamma'].items():
                            v = np.zeros((n_block, 4))
                            v[where] = e['v'][aff]
                            v_leg[g] = v
                        sel = cls3 == cls
                        for j in P['subsets']:
                            dP = self.v3_probs(self.subset_name(j, s)) - p0
                            L1 = {f'{g:g}': float((V[ki, ci, gi][sel] * dP[sel]).sum()) for gi, g in enumerate(gA)}
                            L1_leg = {f'{g:g}': float((v_leg[float(g)][sel] * dP[sel]).sum()) for g in gA}
                            for t in P['ts']:
                                pt = p0 + t * dP
                                eff = (pt.astype(np.float32).astype(np.float64) - p0.astype(np.float32).astype(np.float64))[sel]
                                out, _ = self.condition(fixed, pt, k, cls)
                                rows.append(dict(seed=s, attack_seed=k, cls=cls, subset=j, t=t,
                                                 dJ=out['metrics']['cross_entropy'] - b['metrics']['cross_entropy'],
                                                 dJ_legacy=out['metrics']['cross_entropy_legacy'] - b['metrics']['cross_entropy_legacy'],
                                                 L1=L1, L1_legacy=L1_leg,
                                                 dP_norm=float(t * np.linalg.norm(dP[sel])), dP_norm_float32=float(np.linalg.norm(eff))))
            rt.write(tmp / 'result.json', _clean(dict(panel=P, attack_gamma_star=fr['attack_gamma_star'], rows=rows)))
            rt.finish(tmp, c, cell, elapsed_seconds=time.monotonic() - start)
            os.rename(tmp, final)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return 'done'
