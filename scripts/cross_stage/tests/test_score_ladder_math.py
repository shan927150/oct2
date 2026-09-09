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


if __name__ == "__main__":
    test_attack_stage()
    test_stage1()
    print("all checks passed")
