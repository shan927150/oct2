"""Solver/target checks that run on CPU (unit tests) and on the GPU (smoke)."""
import numpy as np
import solver
import targets as T


def torch_solver_checks(device='cpu', seed=3):
    import torch
    g = torch.Generator().manual_seed(seed)
    n = 60
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    eig = torch.linspace(-0.02, 4.5, n, dtype=torch.float64)  # mildly indefinite like the CNN
    A = (Q * eig) @ Q.T
    A = A.to(device)
    b = torch.randn(n, generator=g, dtype=torch.float64).to(device)
    gammas = [1., .3, .1, .03]
    ms = solver.multishift_lanczos(lambda v: A @ v, b, gammas, n, 1e-12, dot=lambda a, c: float(a @ c))
    out = {}
    for gm in gammas:
        exact = torch.linalg.solve(A + gm * torch.eye(n, dtype=torch.float64, device=device), b)
        err = float((ms['x'][gm] - exact).norm() / exact.norm())
        res = float((A @ ms['x'][gm] + gm * ms['x'][gm] - b).norm() / b.norm())
        out[f'gamma{gm:g}'] = dict(relative_error_vs_dense=err, true_rel_residual=res)
        if err > 1e-8:
            raise AssertionError(f'multi-shift solve wrong at gamma={gm}: {err}')
    # Early-stopped multi-shift iterate equals plain CG after the same number of steps.
    k = 6
    ms6 = solver.multishift_lanczos(lambda v: A @ v + 0. * v, b, [1.], k, 0., dot=lambda a, c: float(a @ c))
    x, r = torch.zeros_like(b), b.clone()
    p, rs = r.clone(), r @ r
    M = A + torch.eye(n, dtype=torch.float64, device=device)
    for _ in range(k):
        Ap = M @ p
        a = rs / (p @ Ap)
        x, r = x + a * p, r - a * Ap
        rs_new = r @ r
        p, rs = r + rs_new / rs * p, rs_new
    cg_err = float((ms6['x'][1.] - x).norm() / x.norm())
    if cg_err > 1e-8:
        raise AssertionError(f'Galerkin iterate differs from CG: {cg_err}')
    out['equals_cg_after_6_steps_rel_error'] = cg_err
    out['ritz_min'] = ms['ritz_min']
    return out


def target_fd_checks(seed=5):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(7, 4))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = rng.integers(0, 4, 7)
    w = rng.random(7)
    d = rng.normal(size=p.shape)
    out = {}
    for t in T.TARGETS:
        _, v = T.value_and_v(t, p, labels, w)
        eps = 1e-6
        fd = (T.value_and_v(t, p + eps * d, labels, w)[0] - T.value_and_v(t, p - eps * d, labels, w)[0]) / (2 * eps)
        out[t] = abs(fd - float((v * d).sum())) / max(1e-12, abs(fd))
        if out[t] > 1e-6:
            raise AssertionError(f'target derivative wrong: {t}')
    return out
