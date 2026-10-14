"""GPU work for cross-stage LDS v5: v4 with a recipe-matched target, a calibration gate and hybrid cells.

v4 (cross_stage_lds_v4_stable_20261013T044559507358835Z) showed that with five stable-recipe shadows
and the legacy target, shadow members have p_true ~ 1 while 12-25% of target members have p_true < 0.99.
The attack learned a very steep boundary, its CE on target queries was 0.95-1.71 (above ln 2) at AUC 0.90,
and deleting half of shadow-3 moved the CE by -0.74 on average with a between-subset SD of only 0.0044,
while the AUC barely moved. v5 changes one factor: the target is retrained with the same stable recipe
on its original split. Everything else (designs, Stage-1 models, attack, seeds, damping grids, primary
classes, rungs) is the v4 code, imported byte-for-byte.

Additions:
- background_target: the stable-recipe target (seed = legacy target_seed), queries from its fp64 softmax.
- build_fixed: v4 table, target queries replaced by the stable target (row identity checked).
- attack metrics gain calibration fields (ECE, confidently-wrong rate); CE/AUC are unchanged.
- xsmoke: after the v4 smoke, a prespecified gate (every primary-class J00 below ln 2 on all full
  baselines) and an exact forward/reverse identity v . (J dtheta) = h . dtheta. Gate failure stops the run.
- hybrid_cell: J with only the deleted rows (or only the shadow-3 nonmember rows) set to their subset
  values, and L1 split by row group. Diagnostic, not part of the primary ladder.
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

LN2 = math.log(2.)
HYBRID_GROUPS = ('deleted', 'nonmember')


def calibration(prob, y, bins=10):
    """Binary calibration of the member probability: ECE (equal-width bins), confidently wrong rate."""
    prob, y = np.asarray(prob, dtype=np.float64), np.asarray(y)
    edges = np.linspace(0., 1., bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = sum(abs(prob[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    wrong = np.where(y == 1, 1. - prob, prob)
    return dict(ece=float(ece), confident_wrong_rate=float((wrong >= .99).mean()),
                mean_member_prob=float(prob.mean()), member_fraction=float((y == 1).mean()))


_EVALUATE = A.evaluate


def evaluate_with_calibration(pilot, model, qx, qm):
    """v4 metrics (CE, AUC, ... unchanged) plus calibration of the same probabilities."""
    m = _EVALUATE(pilot, model, qx, qm)
    prob, _ = pilot.predict_attack_outputs(model, qx)
    m.update(calibration(prob, qm))
    return m


def p_true_profile(x, classes, membership):
    """Saturation of the true-class probability, the quantity whose mismatch broke v4."""
    pt = np.asarray(x)[np.arange(len(x)), np.asarray(classes)]
    out = {}
    for name, sel in (('member', membership == 1), ('nonmember', membership == 0)):
        v = pt[sel]
        out[name] = dict(n=int(sel.sum()), mean=float(v.mean()) if len(v) else None,
                         frac_below_0_99=float((v < .99).mean()) if len(v) else None,
                         frac_below_0_999=float((v < .999).mean()) if len(v) else None)
    return out


def gate_decision(rows, primary, threshold):
    """Prespecified: every (seed, attack seed) J00 of every primary class is below the threshold."""
    per = {}
    for cls in primary:
        j = [r['metrics']['cross_entropy'] for r in rows if r['cls'] == cls]
        if not j:
            raise ValueError(f'No full-baseline attack for primary class {cls}')
        per[str(cls)] = dict(J00_max=float(max(j)), J00_mean=float(np.mean(j)), n=len(j),
                             auc_mean=float(np.mean([r['metrics']['auc'] for r in rows if r['cls'] == cls])),
                             ece_mean=float(np.mean([r['metrics']['ece'] for r in rows if r['cls'] == cls])),
                             confident_wrong_mean=float(np.mean([r['metrics']['confident_wrong_rate']
                                                                 for r in rows if r['cls'] == cls])),
                             passed=bool(max(j) < threshold))
    return dict(threshold=threshold, rule='every primary-class J00 on every full baseline < threshold',
                per_class=per, passed=all(v['passed'] for v in per.values()))


class X5Session(XSession):
    def __init__(self, c, freeze_to=None):
        super().__init__(c, freeze_to=freeze_to)
        if A.evaluate is not evaluate_with_calibration:
            A.evaluate = evaluate_with_calibration

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
            model.double().eval()
            probs = []
            with torch.no_grad():
                for i in range(0, len(rows), self.c['batch']):
                    x = torch.as_tensor(self.ctx['X'][rows[i:i + self.c['batch']]], device=self.dev, dtype=torch.float64)
                    probs.append(model(x).softmax(1).cpu().numpy())
            p = np.concatenate(probs)
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

    def target_queries(self):
        d = self.root / 'background' / 'target'
        rt.completed(d, self.h, dict(kind='target'))
        with np.load(d / 'interface.npz') as z:
            return np.asarray(z['raw']), np.asarray(z['raw_index'])

    def build_fixed(self):
        """v4 attack table (stable shadows 0/1/2/4); target queries from the stable target."""
        fixed = super().build_fixed()
        p, rows = self.target_queries()
        if not np.array_equal(rows, fixed['q_raw_index']):
            raise ValueError('Stable-target query rows differ from the legacy target query rows')
        if not np.isfinite(p).all():
            raise FloatingPointError('Non-finite stable-target probabilities')
        fixed['q_x'] = p.astype(np.float32)   # float32 like every other attack input
        return fixed

    # ------------------------------------------------------------------- smoke
    def xsmoke(self, dest):
        """v4 smoke on the v5 table, then the prespecified calibration gate and the exact JVP identity."""
        dest = Path(dest)
        super().xsmoke(dest)
        c = self.c
        rows = rt.read(dest / 'attack_baselines.json')['rows']
        frozen = rt.read(dest / 'frozen.json')
        gate = gate_decision(rows, frozen['qualified_classes'], c['j00_gate'])
        with np.load(dest / 'fixed_rows.npz') as z:
            fx = {k: z[k] for k in z.files}
        profile = {f'shadow{s}': p_true_profile(fx['x'][fx['shadow_id'] == s], fx['classes'][fx['shadow_id'] == s],
                                                fx['membership'][fx['shadow_id'] == s])
                   for s in sorted(set(fx['shadow_id'].tolist())) if s != A.AFFECTED}
        p3 = self.v3_probs(self.full_name(c['seeds'][0]))
        m3 = fx['membership'][fx['shadow_id'] == A.AFFECTED]
        profile[f'shadow3_full_seed{c["seeds"][0]}'] = p_true_profile(p3, fx['classes'][fx['shadow_id'] == A.AFFECTED], m3)
        profile['target'] = p_true_profile(fx['q_x'], fx['q_classes'], fx['q_membership'])
        gate['p_true_profile'] = profile
        gate['jvp_reverse_identity'] = self.jvp_identity(fx, frozen)
        rt.write(dest / 'gate.json', _clean(gate))
        if gate['jvp_reverse_identity']['relative_error'] > c['adjoint_identity_rel_tol']:
            raise ValueError(f'Forward JVP and reverse-mode h disagree: {gate["jvp_reverse_identity"]}')
        if not gate['passed']:
            rt.write(self.root / f'smoke_gate_failed_{dest.name.lstrip(".")}.json', _clean(gate))
            raise ValueError('Calibration gate failed (primary-class J00 not below ln 2); stop and discuss. '
                             + str({k: v['J00_max'] for k, v in gate['per_class'].items()}))

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

    # ------------------------------------------------------------------ hybrid
    @exclusive_cell
    def hybrid_cell(self, cell, tag):
        """J with only one row group of shadow-3 moved to its subset value; L1 split by row group."""
        import torch
        hcell = dict(name=cell['name'], kind='hybrid', subset=cell['subset'], seed=cell['seed'])
        final, tmp = staged(self.root, Path('hybrid') / cell['name'], tag)
        if final.exists():
            shutil.rmtree(tmp)
            rt.completed(final, self.h, hcell)
            return 'skipped'
        start = time.monotonic()
        c, fixed = self.c, self.fixed()
        s, j = cell['seed'], cell['subset']
        V, _, J00_base = self.base_arrays(s)
        J00 = self.j00(s, fixed, J00_base)
        p_full = self.v3_probs(self.full_name(s))
        p_sub = self.v3_probs(self.subset_name(j, s))
        dP = p_sub - p_full
        member = fixed['membership'][fixed['shadow_id'] == A.AFFECTED] == 1
        deleted = np.r_[self.M[j] == 0, np.zeros(len(self.rows3) - self.N, dtype=bool)]
        groups = dict(deleted=deleted, retained=member & ~deleted, nonmember=~member)
        if (groups['deleted'] & ~member).any() or groups['deleted'].sum() + groups['retained'].sum() != member.sum():
            raise ValueError('Row groups do not partition the shadow-3 interface')
        hybrids = {}
        for g in HYBRID_GROUPS:
            p = p_full.copy()
            p[groups[g]] = p_sub[groups[g]]
            hybrids[g] = p
        cls3 = fixed['classes'][fixed['shadow_id'] == A.AFFECTED]
        rows = []
        for ki, k in enumerate(c['attack_seeds']):
            for ci, cls in enumerate(c['classes']):
                sel = cls3 == cls
                r = dict(attack_seed=k, cls=cls, J00=J00[(k, cls)])
                for g, p in hybrids.items():
                    out, _ = self.condition(fixed, p, k, cls)
                    r[f'J_{g}'] = out['metrics']['cross_entropy']
                    r[f'auc_{g}'] = out['metrics']['auc']
                for g, msk in groups.items():
                    r[f'L1_{g}'] = {f'{gA:g}': float((V[ki, ci, gi][sel & msk] * dP[sel & msk]).sum())
                                    for gi, gA in enumerate(c['attack_gammas'])}
                rows.append(r)
        rt.write(tmp / 'result.json', _clean(dict(cell=hcell, rows=rows,
                 group_sizes={g: int(m.sum()) for g, m in groups.items()})))
        rt.finish(tmp, c, hcell, elapsed_seconds=time.monotonic() - start)
        os.rename(tmp, final)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 'done'
