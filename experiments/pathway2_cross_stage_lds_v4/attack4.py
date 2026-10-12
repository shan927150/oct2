"""Stage-2 pieces for the cross-stage LDS ladder (torch/numpy only; no I/O).

Attack topology is the legacy Shokri one: per OCT class c, an MLP 4->64->2 is trained on the
interface rows of class c from every shadow, and evaluated on the fixed target queries of class
c. Only the affected shadow's rows (shadow 3) change between conditions; membership labels are
frozen (value pathway, Eq. 83).

Adjoint (Eq. 11-21, 75-76), per class c, with the attack Hessian built densely in float64:
    q_c = grad_phi J_{Q,c},  (H_{A,c} + gamma_A I) u_c = q_c,  v_r = -B_r^T u_c = -d/dp_r [u_c . grad_phi L_{A,c}]
so that v_r = dJ_{Q,c}/dp_r (unweighted). The ladder multiplies by omega_c afterwards.
"""
import copy
import numpy as np

AFFECTED = 3


def assemble(fixed, p_affected):
    """Attack training table with the affected shadow's 4 probabilities replaced (float32, like legacy)."""
    x = fixed['x'].copy()
    mask = fixed['shadow_id'] == AFFECTED
    p = np.asarray(p_affected)
    if p.shape != (int(mask.sum()), x.shape[1]):
        raise ValueError('Affected-shadow block has the wrong shape')
    if not np.isfinite(p).all():
        raise FloatingPointError('Non-finite affected-shadow probabilities')
    x[mask] = p.astype(np.float32)
    return x


def class_rows(fixed, cls):
    """Indices (into the attack table) of class-c rows, and the affected-shadow positions among them."""
    tr = np.flatnonzero(fixed['classes'] == cls)
    aff = fixed['shadow_id'][tr] == AFFECTED
    # position of each affected row inside the 4343-row shadow-3 block
    block = np.flatnonzero(fixed['shadow_id'] == AFFECTED)
    where = np.searchsorted(block, tr[aff])
    if not np.array_equal(block[where], tr[aff]):
        raise ValueError('Affected rows are not contiguous with the shadow-3 block')
    return tr, aff, where


def train(pilot, hp, x, membership, seed):
    return pilot.train_attack_model_deterministic(x, membership, int(seed), hp['epochs'], hp['lr'], hp['batch'],
                                                  n_hidden=hp['hidden'], deterministic=hp['deterministic'])


def evaluate(pilot, model, qx, qm):
    metrics, _, _ = pilot.evaluate_attack_queries(model, qx, qm)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in metrics.items()}


def class_condition(pilot, hp, fixed, x, cls, seed, adjoint_gammas=None, device=None):
    """Train the class-c attack on table x, evaluate on class-c target queries; optional adjoint."""
    tr, aff, where = class_rows(fixed, cls)
    te = fixed['q_classes'] == cls
    model = train(pilot, hp, x[tr], fixed['membership'][tr], seed)
    out = dict(metrics=evaluate(pilot, model, fixed['q_x'][te], fixed['q_membership'][te]))
    if adjoint_gammas is not None:
        adj = adjoint(model, x[tr], fixed['membership'][tr], fixed['q_x'][te], fixed['q_membership'][te],
                      adjoint_gammas, device)
        # v rows of the affected shadow, scattered into the full 4343-row block (zeros elsewhere)
        n_block = int((fixed['shadow_id'] == AFFECTED).sum())
        vs = {}
        for g, e in adj['per_gamma'].items():
            v = np.zeros((n_block, x.shape[1]))
            v[where] = e.pop('v')[aff]
            vs[g] = v
        out.update(adjoint={k: val for k, val in adj.items()}, v=vs)
    return out, model


def adjoint(model, P_tr, M_tr, P_q, M_q, gammas, device):
    """Dense float64 attack Hessian, all damping values from one factorisation per gamma."""
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
    Jq = F.cross_entropy(m(Pq), Mq)
    q = flat(torch.autograd.grad(Jq, params))
    L = F.cross_entropy(m(P), M)
    gL = flat(torch.autograd.grad(L, params, create_graph=True))
    d = gL.numel()
    H = torch.empty((d, d), dtype=torch.float64, device=device)
    for k in range(d):
        H[k] = flat(torch.autograd.grad(gL[k], params, retain_graph=True))
    H = .5 * (H + H.T)
    eig = torch.linalg.eigvalsh(H)
    eye = torch.eye(d, dtype=torch.float64, device=device)
    per = {}
    for g in gammas:
        A = H + float(g) * eye
        try:
            u = torch.linalg.solve(A, q)
            ok = bool(torch.isfinite(u).all())
        except RuntimeError:
            u, ok = torch.zeros_like(q), False
        res = float((A @ u - q).norm() / q.norm()) if ok else None
        s = (gL * u.detach()).sum()
        V = torch.autograd.grad(s, P, retain_graph=True)[0]
        dmin, dmax = float(eig.min()) + float(g), float(eig.max()) + float(g)
        per[float(g)] = dict(v=(-V).detach().cpu().numpy(), solve_rel_residual=res, solved=ok,
                             damped_eig_min=dmin, condition_number=(dmax / dmin) if dmin > 0 else None)
    return dict(per_gamma=per, hessian_eig_min=float(eig.min()), hessian_eig_max=float(eig.max()),
                train_grad_norm=float(gL.detach().norm()), q_norm=float(q.norm()), J_query_fp64=float(Jq.detach()))


def qualified_gamma(per_gamma, tol):
    return {g: bool(e['solved'] and e['damped_eig_min'] > 0 and e['solve_rel_residual'] is not None
                    and e['solve_rel_residual'] <= tol) for g, e in per_gamma.items()}


def class_gate(metrics, auc_min, bacc_min):
    return bool(metrics['auc'] >= auc_min and metrics['balanced_accuracy'] >= bacc_min)


def js(p, q):
    p, q = np.clip(p, 1e-12, 1.), np.clip(q, 1e-12, 1.)
    mid = .5 * (p + q)
    return .5 * ((p * np.log(p / mid)).sum(1) + (q * np.log(q / mid)).sum(1))


def weights(q_classes, qualified):
    """omega_c = |Q_c| / sum_{c' qualified} |Q_c'| (image-equal weighting of target queries)."""
    counts = {int(c): int((q_classes == c).sum()) for c in qualified}
    total = sum(counts.values())
    if not qualified or total == 0:
        raise ValueError('No qualified attack class')
    return {c: n / total for c, n in counts.items()}
