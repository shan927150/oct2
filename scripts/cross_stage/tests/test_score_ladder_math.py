"""Finite-difference checks for the implicit-differentiation pieces used by 07.

Run:  python scripts/cross_stage/tests/test_score_ladder_math.py

Test A (Stage 2, Eq. 16-20): v_j = dJ_Q/dp_j from attack_implicit_v() must match a
central finite difference of J_Q(phi*(P + eps e_j)) where phi* is re-optimised to
convergence (full-batch, strongly convex via small L2) after perturbing row j.

Test B (Stage 1, Eq. 56-61): (1/n) w^T g_i with w = (H+gamma I)^{-1} h must match
the change of h^T theta* when training example i is removed and theta* is re-optimised
to convergence, for a small MLP with an L2 term (so the Hessian is PD).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ladder", HERE.parent / "07_cross_stage_score_ladder.py")
ladder = importlib.util.module_from_spec(spec)
sys.modules["ladder"] = ladder
spec.loader.exec_module(ladder)

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device("cpu")
torch.set_default_dtype(torch.float64)


def fit_to_convergence(model, loss_fn, iters=400):
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=iters, tolerance_grad=1e-12,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    opt.step(closure)
    return model


def test_attack_stage():
    N, d_in, wd = 60, 4, 1e-2
    P = torch.softmax(torch.randn(N, d_in), dim=1)
    M = (torch.rand(N) < torch.sigmoid(4 * (P[:, 0] - 0.4))).long()
    Pq = torch.softmax(torch.randn(40, d_in), dim=1)
    Mq = (torch.rand(40) < torch.sigmoid(4 * (Pq[:, 0] - 0.4))).long()
    att = nn.Sequential(nn.Linear(d_in, 8), nn.Tanh(), nn.Linear(8, 2))

    def train_loss(model, P_):
        l2 = sum((p ** 2).sum() for p in model.parameters())
        return F.cross_entropy(model(P_), M) + 0.5 * wd * l2

    def J(model):
        with torch.no_grad():
            return float(F.cross_entropy(model(Pq), Mq))

    fit_to_convergence(att, lambda: train_loss(att, P))
    # note: attack_implicit_v builds the Hessian of plain CE; add wd via damping
    res = ladder.attack_implicit_v(att, P.numpy(), M.numpy(), Pq.numpy(), Mq.numpy(), damping=wd, device=DEV)
    v = res["v"]
    errs = []
    for j in [0, 7, 23]:
        for k in range(d_in):
            eps = 1e-4
            fd = []
            for sgn in (+1, -1):
                Pp = P.clone(); Pp[j, k] += sgn * eps
                m2 = nn.Sequential(nn.Linear(d_in, 8), nn.Tanh(), nn.Linear(8, 2))
                m2.load_state_dict(att.state_dict())
                fit_to_convergence(m2, lambda: train_loss(m2, Pp))
                fd.append(J(m2))
            fd_val = (fd[0] - fd[1]) / (2 * eps)
            errs.append((v[j, k], fd_val))
    v_imp = np.array([e[0] for e in errs]); v_fd = np.array([e[1] for e in errs])
    rel = np.abs(v_imp - v_fd).max() / (np.abs(v_fd).max() + 1e-12)
    print(f"[A] attack implicit v vs finite difference: max rel err = {rel:.2e}; "
          f"corr = {np.corrcoef(v_imp, v_fd)[0,1]:.6f}; |grad L|={res['train_grad_norm']:.1e}")
    assert rel < 2e-2, rel


def test_stage1():
    n, d_in, wd, gamma = 80, 6, 5e-2, 0.0
    Xn = torch.randn(n, d_in)
    yn = (Xn[:, 0] + 0.5 * Xn[:, 1] > 0).long()
    net = nn.Sequential(nn.Linear(d_in, 6), nn.Tanh(), nn.Linear(6, 2))
    idx = np.arange(n)

    def loss_fn(model, keep):
        l2 = sum((p ** 2).sum() for p in model.parameters())
        return F.cross_entropy(model(Xn[keep]), yn[keep]) + 0.5 * wd * l2

    fit_to_convergence(net, lambda: loss_fn(net, idx))
    theta0 = ladder.flatten([p.detach() for p in ladder.flat_params(net)])
    h = torch.randn_like(theta0)  # arbitrary downstream direction h_s
    hvp = ladder.make_hvp(net, Xn.numpy(), yn.numpy(), idx, DEV, wd, gamma, batch=n)
    cg = ladder.conjugate_gradient(hvp, h, iters=200, tol=1e-10)
    w = cg["x"]
    G = ladder.per_image_grads(net, Xn.numpy(), yn.numpy(), [3, 11, 40], DEV)
    pred = (G @ w / n).numpy()          # Eq. 61: predicted change in h^T theta after removal
    actual = []
    for i in [3, 11, 40]:
        m2 = nn.Sequential(nn.Linear(d_in, 6), nn.Tanh(), nn.Linear(6, 2))
        m2.load_state_dict(net.state_dict())
        keep = np.array([k for k in idx if k != i])
        # removal = mean over n-1 examples; Eq. 61 uses eps=-1/n (first order), so compare
        # against the exact (n-1)-mean refit, which is what the derivation approximates.
        fit_to_convergence(m2, lambda: loss_fn(m2, keep))
        theta1 = ladder.flatten([p.detach() for p in ladder.flat_params(m2)])
        actual.append(float(h @ (theta1 - theta0)))
    actual = np.array(actual)
    print(f"[B] stage-1 implicit (1/n) w^T g_i vs refit h^T dtheta: pred={pred}, actual={actual}, "
          f"CG iters={cg['iters']} res={cg['final_rel_residual']:.1e}")
    rel = np.abs(pred - actual).max() / (np.abs(actual).max() + 1e-12)
    assert rel < 0.15, rel  # first-order in 1/n; n=80 -> O(1%) plus refit tolerance




def test_detach_equivalence_nonstationary():
    """Blocker-A check: v must not change whether u is detached or not, even far from
    a stationary point (H and q are built without create_graph, so u is graph-free)."""
    torch.manual_seed(1)
    att = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))  # untrained: |grad L| large
    P = torch.softmax(torch.randn(50, 4), 1).numpy(); M = (np.random.rand(50) < 0.5).astype(int)
    Pq = torch.softmax(torch.randn(30, 4), 1).numpy(); Mq = (np.random.rand(30) < 0.5).astype(int)
    res = ladder.attack_implicit_v(att, P, M, Pq, Mq, 1e-3, DEV)
    # manual version that differentiates a *non-detached* u through an explicit graph
    from torch.func import functional_call
    names = [n for n, _ in att.named_parameters()]
    params = [p.detach().clone().requires_grad_(True) for _, p in att.named_parameters()]
    pd = dict(zip(names, params)); Pt = torch.tensor(P).requires_grad_(True)
    L = F.cross_entropy(functional_call(att, pd, (Pt,)), torch.tensor(M))
    gL = ladder.flatten(torch.autograd.grad(L, params, create_graph=True))
    q = ladder.flatten(torch.autograd.grad(
        F.cross_entropy(functional_call(att, pd, (torch.tensor(Pq),)), torch.tensor(Mq)), params))
    d = gL.numel()
    rows = []
    for k in range(d):
        g = torch.autograd.grad(gL[k], params, retain_graph=True, create_graph=True, allow_unused=True)
        rows.append(ladder.flatten([gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]))
    H = torch.stack(rows); H = 0.5 * (H + H.T)                      # graph-carrying Hessian
    u_graph = torch.linalg.solve(H + 1e-3 * torch.eye(d), q)         # NOT detached
    u_const = u_graph.detach()
    v_right = -torch.autograd.grad((gL * u_const).sum(), Pt, retain_graph=True)[0].numpy()
    v_wrong = -torch.autograd.grad((gL * u_graph).sum(), Pt)[0].numpy()
    err_impl = np.abs(res["v"] - v_right).max() / (np.abs(v_right).max() + 1e-12)
    extra = np.abs(v_wrong - v_right).max() / (np.abs(v_right).max() + 1e-12)
    print(f"[C] |grad L|={res['train_grad_norm']:.2e}; implementation vs detached reference rel err = {err_impl:.1e}; "
          f"size of the spurious (dU/dP)^T gL term if u were NOT detached = {extra:.2e}")
    assert err_impl < 1e-10, err_impl


def test_cg_curvature_gate():
    """CG must flag an indefinite operator instead of returning garbage silently."""
    A = torch.diag(torch.tensor([2.0, 1.0, -0.5]))
    res = ladder.conjugate_gradient(lambda v: A @ v, torch.tensor([1.0, 1.0, 1.0]), 20, 1e-8)
    assert res["nonpositive_curvature"] is True or res["min_rayleigh_quotient"] <= 0
    B = torch.diag(torch.tensor([2.0, 1.0, 0.5]))
    res2 = ladder.conjugate_gradient(lambda v: B @ v, torch.tensor([1.0, 1.0, 1.0]), 20, 1e-10)
    assert not res2["nonpositive_curvature"] and res2["final_rel_residual"] < 1e-8
    lz = ladder.lanczos_extreme_eigs(lambda v: B @ v, 3, 3, DEV, torch.float64)
    assert abs(lz["lambda_min_est"] - 0.5) < 1e-6 and abs(lz["lambda_max_est"] - 2.0) < 1e-6
    print(f"[D] CG curvature gate OK; Lanczos eigs {lz['lambda_min_est']:.3f}/{lz['lambda_max_est']:.3f}")


def test_fixed_mask_rng_alignment():
    """fixed_mask deletion consumes exactly the baseline RNG stream (dropout aligned) and is
    independent of the removed row's content; filter_rechunk does not have these properties."""
    import importlib.util as iu
    spec5 = iu.spec_from_file_location("pilot05", HERE.parent / "05_end_to_end_patient_loo_pilot.py")
    pilot = iu.module_from_spec(spec5); sys.modules["pilot05"] = pilot; spec5.loader.exec_module(pilot)
    torch.set_default_dtype(torch.float32)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 1, 16, 16)).astype(np.float32); y = rng.integers(0, 4, size=40)
    orders = pilot.make_epoch_orders(np.arange(40), 2, 123)
    kw = dict(eval_indices=np.arange(40), seed=7, n_hidden=8, lr=1e-3, batch_size=16,
              weight_decay=0.0, deterministic=True)

    def run(mode, excl, X_=X):
        m, _ = pilot.train_classifier_from_orders(X_, y, orders, excluded_indices=excl, deletion_mode=mode, **kw)
        return ladder.flatten([p.detach() for p in m.parameters()]).clone(), torch.get_rng_state().clone()

    base, rng_base = run("fixed_mask", [])
    base_f, _ = run("filter_rechunk", [])
    assert torch.equal(base, base_f), "no-deletion path must be identical in both modes"
    loo, rng_loo = run("fixed_mask", [3, 17])
    assert torch.equal(rng_base, rng_loo), "fixed_mask must leave the RNG stream aligned with baseline"
    X2 = X.copy(); X2[[3, 17]] = rng.normal(size=(2, 1, 16, 16)).astype(np.float32)
    loo2, _ = run("fixed_mask", [3, 17], X2)
    assert torch.equal(loo, loo2), "fixed_mask result must not depend on the removed rows' content"
    assert not torch.equal(loo, base), "deletion must change the model"
    _, rng_filter = run("filter_rechunk", [3, 17])
    print(f"[E] fixed_mask: RNG aligned, content-independent; filter_rechunk RNG aligned with baseline: "
          f"{torch.equal(rng_base, rng_filter)}")


if __name__ == "__main__":
    test_attack_stage()
    test_stage1()
    test_detach_equivalence_nonstationary()
    test_cg_curvature_gate()
    test_fixed_mask_rng_alignment()
    print("all checks passed")
