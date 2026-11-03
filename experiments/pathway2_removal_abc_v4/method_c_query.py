#!/usr/bin/env python3
"""Method C: remove target attack queries. The attack training table and models are unchanged.

For class c the attack objective is J_Q = mean_{i in Q_c} l_i (cross-entropy of target query i). Because
J_Q and its gradient q are averages over queries, and v = -B^T (H + gamma I)^{-1} q with H from the
training table only, both the truth and the prediction split exactly into per-query terms:

    truth_ij = l_i(J10 model of subset j) - l_i(J00 model)
    score_ij = -g_i^T (H + gamma I)^{-1} w_j,   w_j = (d grad_phi L / dP) . dP_j,   g_i = grad_phi l_i
    mean_i truth_ij = J10_j - J00,  mean_i score_ij = sum_r v_r . dp_{j,r}

worker (GPU, one task per Stage-1 seed): replays the stored v6 attacks (same table, same seeds, exact
replay), saves l00, l10 and score for every query, and checks that their query means reproduce the stored
J00, J10 and the stored per-class bridge score. No new attack condition is trained.

analyze (CPU, can run anywhere): removes query sets S, recomputes truth and prediction on Q \\ S and the
LDS over the same 32 subsets, against matched random query removals.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

import common as C

NAMES = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
VARIANTS = ('if', 'true', 'row3')        # prediction built from bridge rows 1, 2, 3
REFERENCE = 0.26173020527859236         # interface v1r1 bridge row 1 (gamma 0.03, panel 42-44)


# ------------------------------------------------------------------ per-query adjoint (torch)
def query_scores(model, P_tr, M_tr, P_q, M_q, gamma, deltas, device):
    """Per-query losses' gradients pushed through the attack adjoint.

    deltas: {name: [J, len(P_tr), 4]} changes of the training-table rows. Returns {name: [n_q, J]},
    plus the dense-solve diagnostics. The query mean of each matrix equals sum_r v_r . delta_r of
    attack4.adjoint with the same model, table, queries and gamma."""
    import torch
    import torch.nn.functional as F
    m = copy.deepcopy(model).double().eval()
    params = list(m.parameters())
    for p in params:
        p.requires_grad_(True)
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    P = torch.as_tensor(np.asarray(P_tr, dtype=np.float64), device=device).requires_grad_(True)
    M = torch.as_tensor(np.asarray(M_tr), dtype=torch.long, device=device)
    Pq = torch.as_tensor(np.asarray(P_q, dtype=np.float64), device=device)
    Mq = torch.as_tensor(np.asarray(M_q), dtype=torch.long, device=device)
    L = F.cross_entropy(m(P), M)
    gL = flat(torch.autograd.grad(L, params, create_graph=True))
    d = gL.numel()
    H = torch.empty((d, d), dtype=torch.float64, device=device)
    for k in range(d):
        H[k] = flat(torch.autograd.grad(gL[k], params, retain_graph=True))
    H = .5 * (H + H.T)
    A = H + float(gamma) * torch.eye(d, dtype=torch.float64, device=device)
    lq = F.cross_entropy(m(Pq), Mq, reduction='none')
    G = torch.stack([flat(torch.autograd.grad(lq[i], params, retain_graph=True)) for i in range(len(lq))])
    z = torch.zeros(d, dtype=torch.float64, device=device, requires_grad=True)
    vjp = torch.autograd.grad((gL * z).sum(), P, create_graph=True)[0]          # B^T z, linear in z
    out = {}
    for name, D in deltas.items():
        Dt = torch.as_tensor(np.asarray(D, dtype=np.float64), device=device)
        W = torch.stack([torch.autograd.grad((vjp * Dt[j]).sum(), z, retain_graph=True)[0]
                         for j in range(Dt.shape[0])], 1)                       # [d, J] = B D_j
        X = torch.linalg.solve(A, W)
        out[name] = (-(G @ X)).detach().cpu().numpy()
    q = G.mean(0)
    u = torch.linalg.solve(A, q)
    res = float((A @ u - q).norm() / q.norm())
    return out, dict(solve_rel_residual=res, damped_eig_min=float(torch.linalg.eigvalsh(H).min()) + float(gamma),
                     n_query=int(len(lq)))


def per_query_loss(pilot, model, qx, qm):
    metrics, prob, logp = pilot.evaluate_attack_queries(model, qx, qm)
    l = -logp[np.arange(len(qm)), qm].astype(np.float64)
    return l, float(metrics['cross_entropy']), prob


# ------------------------------------------------------------------ worker
def worker(a):
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    seed = C.PANEL[a.task]
    final = dest / f'seed{seed}.npz'
    if final.exists():
        raise FileExistsError(final)
    start = time.monotonic()
    c, data, ai, fixed = C.load_context(a.repo, a.out)
    # load_context first imports the frozen checkout's runtime, which adds the
    # other experiment directories to sys.path. Do not depend on PYTHONPATH.
    import attack4 as A
    from core_if import ISession
    session = ISession(c)
    fr = data['fr6']
    seeds = [int(k) for k in fr['attack_seeds']]
    gamma_a = float(fr['attack_gamma_star'])
    qualified = [int(x) for x in fr['qualified_classes']]
    classes = [int(x) for x in a.classes.split(',')]
    J = len(data['M'])
    dp = {'if': C.if_dp(c, data, ai, seed)}
    dp['true'] = C.true_dp(c, data, seed)
    dp['row3'] = C.row3_dp(c, data, seed, dp['if'])
    pfull = data['full'][seed]
    xfull = A.assemble(fixed, pfull)
    xsub = [A.assemble(fixed, ai.v3_interface(c, f'{c["design_name"]}_s{j:03d}_seed{seed}', data['rows']['raw_index']))
            for j in range(J)]
    V6 = C.class_v_per_seed(c, data, seed)
    stored = {}
    for j in range(J):
        for r in data['v6_truth'][(j, seed)]:
            stored[(j, int(r['cls']), int(r['attack_seed']))] = r
    arrays, meta = {}, dict(seed=seed, attack_seeds=seeds, gamma_a=gamma_a, classes=classes, qualified=qualified,
                            omega={str(k): v for k, v in C.omega(data).items()}, checks={})
    for cls in classes:
        tr, aff, where = A.class_rows(fixed, cls)
        te = np.flatnonzero(fixed['q_classes'] == cls)
        qx, qm = fixed['q_x'][te], fixed['q_membership'][te].astype(int)
        D = {}
        for name, x in dp.items():
            Dn = np.zeros((J, len(tr), x.shape[-1]))
            Dn[:, aff] = x[:, where]
            D[name] = Dn
        K = len(seeds)
        l00 = np.zeros((K, len(te))); p00 = np.zeros((K, len(te))); l10 = np.zeros((K, J, len(te)))
        score = {n: np.zeros((K, len(te), J)) for n in dp}
        ck = dict(J00_max_abs=0., J10_max_abs=0., score_if_max_rel=0., solve_rel_residual_max=0.)
        for ki, k in enumerate(seeds):
            model = A.train(session.pilot, session.hp, xfull[tr], fixed['membership'][tr], k)
            l00[ki], ce, p00[ki] = per_query_loss(session.pilot, model, qx, qm)
            ref = stored[(0, cls, k)]['J00']
            if not np.isfinite(l00[ki]).all() or not np.isfinite(p00[ki]).all() or not np.isfinite([ce, ref]).all():
                raise ValueError(f'non-finite J00 or per-query output for class {cls}, attack seed {k}')
            ck['J00_max_abs'] = max(ck['J00_max_abs'], abs(ce - ref), abs(l00[ki].mean() - ref))
            s, diag = query_scores(model, xfull[tr], fixed['membership'][tr], qx, qm, gamma_a, D, session.dev)
            for n in dp:
                score[n][ki] = s[n]
            ref_score = np.einsum('jrk,rk->j', dp['if'], V6[cls][ki])
            if not all(np.isfinite(s[n]).all() for n in dp) or not np.isfinite(ref_score).all() or not np.isfinite(diag['solve_rel_residual']):
                raise ValueError(f'non-finite query score or solve for class {cls}, attack seed {k}')
            rel = float(np.max(np.abs(s['if'].mean(0) - ref_score)) / max(np.abs(ref_score).max(), 1e-300))
            ck['score_if_max_rel'] = max(ck['score_if_max_rel'], rel)
            ck['solve_rel_residual_max'] = max(ck['solve_rel_residual_max'], diag['solve_rel_residual'])
            for j in range(J):
                mj = A.train(session.pilot, session.hp, xsub[j][tr], fixed['membership'][tr], k)
                l10[ki, j], ce, _ = per_query_loss(session.pilot, mj, qx, qm)
                ref = stored[(j, cls, k)]['J10']
                if not np.isfinite(l10[ki, j]).all() or not np.isfinite([ce, ref]).all():
                    raise ValueError(f'non-finite J10 for class {cls}, attack seed {k}, subset {j}')
                ck['J10_max_abs'] = max(ck['J10_max_abs'], abs(ce - ref), abs(l10[ki, j].mean() - ref))
            print(f'seed {seed} class {cls} attack {k} done  {ck}', flush=True)
        meta['checks'][str(cls)] = ck
        if ck['J00_max_abs'] > a.replay_tol or ck['J10_max_abs'] > a.replay_tol:
            raise ValueError(f'attack replay differs from stored v6 truth for class {cls}: {ck}')
        if cls in qualified and ck['score_if_max_rel'] > a.score_tol:
            raise ValueError(f'per-query scores do not add up to the stored bridge score for class {cls}: {ck}')
        p = f'c{cls}_'
        arrays.update({p + 'query_index': te, p + 'membership': qm, p + 'q_x': qx, p + 'l00': l00, p + 'p00': p00,
                       p + 'l10': l10, **{p + 'score_' + n: v for n, v in score.items()}})
    meta['elapsed_seconds'] = time.monotonic() - start
    tmp = dest / f'.seed{seed}.{os.getpid()}.npz'
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, final)
    (dest / f'seed{seed}_checks.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('COMPLETE', final, flush=True)


# ------------------------------------------------------------------ analysis
def load_results(dest):
    """Per class: membership, q_x, l00 [S,K,n], p00, d [S,K,n,J] (l10 - l00), score_<v> [S,K,n,J]."""
    out, meta0, checks = {}, None, {}
    for s in C.PANEL:
        with np.load(Path(dest) / f'seed{s}.npz') as z:
            meta = json.loads(str(z['meta']))
            if int(meta['seed']) != s:
                raise ValueError(f'{s}: result declares seed {meta["seed"]}')
            if meta0 is None:
                meta0 = meta
            else:
                for key in ('classes', 'qualified', 'omega', 'attack_seeds', 'gamma_a'):
                    if meta.get(key) != meta0.get(key):
                        raise ValueError(f'{s}: result metadata differs: {key}')
            checks[str(s)] = meta.get('checks', {})
            for cls in meta['classes']:
                p = f'c{cls}_'
                e = out.setdefault(cls, dict(membership=z[p + 'membership'], q_x=z[p + 'q_x'], query_index=z[p + 'query_index'],
                                             l00=[], p00=[], d=[], **{'score_' + v: [] for v in VARIANTS}))
                if not np.array_equal(e['query_index'], z[p + 'query_index']):
                    raise ValueError('query order differs between Stage-1 seeds')
                if not np.array_equal(e['membership'], z[p + 'membership']) or not np.array_equal(e['q_x'], z[p + 'q_x']):
                    raise ValueError('query labels or inputs differ between Stage-1 seeds')
                e['l00'].append(z[p + 'l00']); e['p00'].append(z[p + 'p00'])
                e['d'].append(np.transpose(z[p + 'l10'], (0, 2, 1)) - z[p + 'l00'][:, :, None])
                for v in VARIANTS:
                    e['score_' + v].append(z[p + 'score_' + v])
    for e in out.values():
        for k in ['l00', 'p00', 'd'] + ['score_' + v for v in VARIANTS]:
            e[k] = np.stack(e[k])
    meta0 = dict(meta0, checks_by_seed=checks)
    return out, meta0


class Pool:
    """All queries of the target classes, concatenated, with per-query means for a chosen set of attack seeds."""

    def __init__(self, res, classes, seed_idx=None):
        self.classes = classes
        parts = [res[c] for c in classes]
        sl = slice(None) if seed_idx is None else seed_idx
        self.cls = np.concatenate([np.full(len(p['membership']), c) for c, p in zip(classes, parts)])
        self.member = np.concatenate([p['membership'] for p in parts])
        self.d = np.concatenate([p['d'][:, sl].mean((0, 1)) for p in parts])                # [n, J]
        self.s = {v: np.concatenate([p['score_' + v][:, sl].mean((0, 1)) for p in parts]) for v in VARIANTS}
        self.loss = np.concatenate([p['l00'].mean((0, 1)) for p in parts])
        self.conf = np.concatenate([np.abs(p['p00'].mean((0, 1)) - .5) for p in parts])
        self.tconf = np.concatenate([p['q_x'].max(1) for p in parts])
        self.strata = self.cls * 2 + self.member

    def target(self, keep, omega=None):
        """keep: [..., n] bool. omega None -> image-equal weights recomputed on the kept queries (pooled mean);
        otherwise fixed class weights. Returns (score {v: [..., J]}, truth [..., J])."""
        keep = np.asarray(keep, float)
        if omega is None:
            w = keep / keep.sum(-1, keepdims=True)
        else:
            w = np.zeros_like(keep)
            for c in self.classes:
                m = (self.cls == c)
                n = (keep * m).sum(-1, keepdims=True)
                w = w + np.where(n > 0, keep * m * omega[c] / np.maximum(n, 1), 0)
        return {v: w @ self.s[v] for v in VARIANTS}, w @ self.d


def drop_within_strata(pool, feature, frac, highest):
    """Remove the top `frac` of each (class, membership) stratum by `feature` (highest or lowest)."""
    keep = np.ones(len(feature), bool)
    for st in np.unique(pool.strata):
        idx = np.flatnonzero(pool.strata == st)
        n = int(round(frac * len(idx)))
        if n:
            order = np.argsort(-feature[idx] if highest else feature[idx], kind='stable')
            keep[idx[order[:n]]] = False
    return keep


def random_keeps(pool, keep, draws, rng, strata):
    """Random removals with the same number removed in each stratum."""
    out = np.ones((draws, len(keep)), bool)
    for st in np.unique(strata):
        idx = np.flatnonzero(strata == st)
        n = int((~keep[idx]).sum())
        if n == 0:
            continue
        if n == len(idx):
            out[:, idx] = False
            continue
        pick = np.argpartition(rng.random((draws, len(idx))), n - 1, axis=1)[:, :n]
        out[np.arange(draws)[:, None], idx[pick]] = False
    return out


def evaluate(pool, keep, draws, rng, strata, omega, label, extra, rows):
    sc, tr = pool.target(keep)
    row = dict(rule=label, **extra, n_kept=int(keep.sum()), n_removed=int((~keep).sum()),
               removed_member=int(((~keep) & (pool.member == 1)).sum()),
               **{f'lds_{v}': C.spearman(sc[v], tr) for v in VARIANTS})
    if omega is not None:
        scf, trf = pool.target(keep, omega)
        row['lds_if_fixed_omega'] = C.spearman(scf['if'], trf)
    for c in pool.classes:
        mk = keep & (pool.cls == c)
        if mk.any():
            s1, t1 = pool.target(mk)
            row[f'lds_if_{NAMES[c]}'] = C.spearman(s1['if'], t1)
    if draws and strata is not None and (~keep).any():
        rk = random_keeps(pool, keep, draws, rng, strata)
        rs, rt = pool.target(rk)
        rl = C.spearman_rows(rs['if'], rt)
        row.update(random_mean=float(rl.mean()), random_p025=float(np.quantile(rl, .025)),
                   random_p975=float(np.quantile(rl, .975)), frac_random_ge=float(np.mean(rl >= row['lds_if'])))
    rows.append(row)
    return row


def analyze(a):
    res, meta = load_results(a.dest)
    om = {int(k): v for k, v in meta['omega'].items()}
    qualified = [int(x) for x in meta['qualified']]
    rng = np.random.default_rng(a.seed)
    rows = []
    pool = Pool(res, qualified)
    full = evaluate(pool, np.ones(len(pool.cls), bool), 0, rng, None, om, 'none', {}, rows)
    checks = dict(lds_if_full=full['lds_if'], reference=REFERENCE)
    if abs(full['lds_if'] - REFERENCE) > 1e-9:
        raise ValueError(f'full query set does not reproduce bridge row 1: {checks}')
    # class: one target class at a time (= old E1); CNV / NORMAL attacks did not pass the quality gate
    for c in sorted(res):
        pc = Pool(res, [c])
        evaluate(pc, np.ones(len(pc.cls), bool), 0, rng, None, None, f'class_only_{NAMES[c]}',
                 dict(qualified=c in qualified), rows)
    # membership of the target queries, within the qualified target
    for keep_member, name in [(1, 'members_only'), (0, 'nonmembers_only')]:
        evaluate(pool, pool.member == keep_member, a.draws, rng, pool.cls, om, name, {}, rows)
    # difficulty defined by the J00 attack alone (no truth), removed within (class, membership) strata
    feats = dict(high_loss=(pool.loss, True), low_attack_conf=(pool.conf, False), low_target_conf=(pool.tconf, False))
    for fname, (feat, highest) in feats.items():
        for frac in a.fracs:
            evaluate(pool, drop_within_strata(pool, feat, frac, highest), a.draws, rng, pool.strata, om,
                     f'drop_{fname}', dict(frac=frac), rows)
    # "badly predicted" queries: per-query agreement over subsets, cross-fitted over attack seeds
    K = len(meta['attack_seeds'])
    halves = [np.arange(K // 2), np.arange(K // 2, K)]
    for direction, (sel, ev) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        ps, pe = Pool(res, qualified, sel), Pool(res, qualified, ev)
        agree = C.spearman_rows(ps.s['if'], ps.d)
        agree = np.where(np.isfinite(agree), agree, 0.)
        evaluate(pe, np.ones(len(pe.cls), bool), 0, rng, None, om, 'crossfit_none', dict(direction=direction), rows)
        for frac in a.fracs:
            keep = drop_within_strata(ps, agree, frac, highest=False)
            evaluate(pe, keep, a.draws, rng, pe.strata, om, 'crossfit_drop_low_agreement',
                     dict(direction=direction, frac=frac), rows)
    agree = C.spearman_rows(pool.s['if'], pool.d)
    agree = np.where(np.isfinite(agree), agree, 0.)
    for frac in a.fracs:      # same seeds select and evaluate: Oracle, descriptive only
        evaluate(pool, drop_within_strata(pool, agree, frac, highest=False), a.draws, rng, pool.strata, om,
                 'oracle_drop_low_agreement', dict(frac=frac), rows)
    fields = ['rule', 'qualified', 'direction', 'frac', 'n_kept', 'n_removed', 'removed_member', 'lds_if', 'lds_true',
              'lds_row3', 'lds_if_fixed_omega'] + [f'lds_if_{NAMES[c]}' for c in sorted(res)] + \
             ['random_mean', 'random_p025', 'random_p975', 'frac_random_ge']
    tmp = Path(a.dest) / f'.c_results.{os.getpid()}.csv'
    try:
        C.writecsv(tmp, rows, fields)
        os.replace(tmp, Path(a.dest) / 'c_results.csv')
    finally:
        tmp.unlink(missing_ok=True)
    Path(a.dest, 'c_checks.json').write_text(json.dumps(dict(checks, worker_checks_by_seed=meta['checks_by_seed']), indent=2) + '\n')
    for r in rows:
        rr = f"  random {r['random_mean']:.3f} [{r['random_p025']:.3f}, {r['random_p975']:.3f}]" if 'random_mean' in r else ''
        tag = ' '.join(f'{k}={r[k]}' for k in ('direction', 'frac') if k in r)
        print(f"{r['rule']:30s} {tag:22s} kept={r['n_kept']:5d} LDS_if={r['lds_if']:.3f}{rr}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('worker')
    w.add_argument('--repo', required=True); w.add_argument('--out', required=True); w.add_argument('--dest', required=True)
    w.add_argument('--task', type=int, required=True, help='0, 1, 2 -> Stage-1 seed 42, 43, 44')
    w.add_argument('--classes', default='0,1,2,3', help='attack classes to replay (DME=1, DRUSEN=2 are the target)')
    w.add_argument('--replay-tol', type=float, default=1e-12)
    w.add_argument('--score-tol', type=float, default=1e-6)
    w.set_defaults(fn=worker)
    q = sub.add_parser('analyze')
    q.add_argument('--dest', required=True)
    q.add_argument('--draws', type=int, default=1000)
    q.add_argument('--seed', type=int, default=20261102)
    q.add_argument('--fracs', type=lambda s: [float(x) for x in s.split(',')], default=[.1, .25, .5])
    q.set_defaults(fn=analyze)
    a = p.parse_args()
    if a.cmd == 'worker' and not 0 <= a.task < len(C.PANEL):
        p.error('task must be 0, 1 or 2')
    a.fn(a)


if __name__ == '__main__':
    main()
