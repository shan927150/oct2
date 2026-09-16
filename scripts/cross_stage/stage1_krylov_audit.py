"""Shared Lanczos basis with explicitly verified shifted least-squares solves.

This implements the useful shift-invariance idea from stage1_krylov_probe.py.
Projected residuals are estimates, spectral masses are quadrature estimates,
and numerical qualification is distinct from an influence-model assumption.
"""
from __future__ import annotations
import math
import torch
import stage1_diagnostic_core as core


class CountedOperator:
    def __init__(self, operator):
        self.operator, self.calls = operator, 0

    def __call__(self, x):
        self.calls += 1
        return self.operator(x)


def lanczos_from_rhs(matvec, b, steps):
    if steps < 1 or b.ndim != 1 or not bool(torch.isfinite(b).all()):
        raise ValueError("Require positive steps and a finite vector RHS")
    bnorm = core.norm(b)
    if not bnorm:
        return {"Q": b.reshape(-1, 1)[:, :0], "diag": [], "betas": [],
                "bnorm": 0., "orthogonality_error": 0., "rhs_zero": True}
    q, previous, previous_beta = b / bnorm, torch.zeros_like(b), 0.
    vectors, diagonal, betas = [], [], []
    for _ in range(min(steps, b.numel())):
        vectors.append(q)
        Aq = matvec(q)
        if not bool(torch.isfinite(Aq).all()):
            raise RuntimeError("Nonfinite Hessian-vector product")
        alpha = float(q.double() @ Aq.double())
        z = Aq - alpha * q - previous_beta * previous
        for _pass in range(2):
            for v in vectors:
                z = z - (v.double() @ z.double()).to(z.dtype) * v
        beta = core.norm(z)
        diagonal.append(alpha)
        betas.append(beta)
        if beta <= 1e-12 * max(1., core.norm(Aq)):
            break
        previous, q, previous_beta = q, z / beta, beta
    Q = torch.stack(vectors, dim=1)
    gram = Q.double().T @ Q.double()
    error = float((gram - torch.eye(gram.shape[0], device=gram.device)).abs().max())
    return {"Q": Q, "diag": diagonal, "betas": betas, "bnorm": bnorm,
            "orthogonality_error": error, "rhs_zero": False}


def projected_matrix(krylov, requested_steps):
    m = min(requested_steps, len(krylov["diag"]))
    T = torch.diag(torch.tensor(krylov["diag"][:m], dtype=torch.float64))
    if m > 1:
        off = torch.tensor(krylov["betas"][:m - 1], dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    return T


def solve_shift(matvec, b, krylov, gamma, requested_steps):
    """MINRES-style projected least squares, then an actual HVP residual.

    All small matrix algebra is float64 on CPU. The coefficient vector is
    explicitly transferred to Q.device before reconstructing a solution.
    The finite-precision Lanczos relation is never treated as exact.
    """
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Require finite nonnegative gamma")
    if krylov["rhs_zero"]:
        return {"x": torch.zeros_like(b), "Hx": torch.zeros_like(b), "steps": 0,
                "rhs_zero": True, "true_relative_residual": None,
                "projected_relative_residual": None, "finite": True}
    T = projected_matrix(krylov, requested_steps)
    m = T.shape[0]
    extended = torch.zeros((m + 1, m), dtype=torch.float64)
    extended[:m] = T + gamma * torch.eye(m, dtype=torch.float64)
    extended[m, m - 1] = krylov["betas"][m - 1]
    rhs = torch.zeros(m + 1, dtype=torch.float64)
    rhs[0] = krylov["bnorm"]
    z = torch.linalg.lstsq(extended, rhs, rcond=1e-12, driver="gelsd").solution
    Q = krylov["Q"][:, :m]
    x = (Q.double() @ z.to(device=Q.device)).to(dtype=Q.dtype)
    Hx = matvec(x)
    residual = Hx.double() + gamma * x.double() - b.double()
    sv = torch.linalg.svdvals(extended)
    shifted_eig = torch.linalg.eigvalsh(extended[:m])
    true = core.norm(residual) / krylov["bnorm"]
    projected = core.norm(extended @ z - rhs) / krylov["bnorm"]
    return {"x": x, "Hx": Hx, "steps": m, "rhs_zero": False,
            "true_relative_residual": true, "projected_relative_residual": projected,
            "residual_disagreement": abs(true - projected),
            "projected_min_eig": float(shifted_eig.min()),
            "projected_min_abs_eig": float(shifted_eig.abs().min()),
            "projected_condition_estimate": float(sv.max() / sv.min()) if float(sv.min()) > 0 else None,
            "finite": bool(torch.isfinite(x).all() and torch.isfinite(Hx).all()) and math.isfinite(true)}


def qualify_pair(short, long, orthogonality_error, residual_tol, stability_tol):
    scale = core.norm(long["x"])
    change = core.norm(long["x"] - short["x"]) / scale if scale else None
    residual_ok = all(r["finite"] and r["true_relative_residual"] is not None
                      and r["true_relative_residual"] <= residual_tol for r in (short, long))
    stable = change is not None and math.isfinite(change) and change <= stability_tol
    passed = bool(residual_ok and stable and orthogonality_error <= 1e-3)
    return {"linear_solve_qualified": passed, "two_depth_residuals_passed": bool(residual_ok),
            "relative_depth_change": change, "depth_stability_passed": bool(stable),
            "orthogonality_error": orthogonality_error,
            "qualification_scope": "Numerical solve check, not an SPD or forward-error certificate"}


def weighted_spectrum(krylov, requested_steps, gammas):
    if krylov["rhs_zero"]:
        return {"rhs_zero": True}
    T = projected_matrix(krylov, requested_steps)
    nodes, vectors = torch.linalg.eigh(T)
    weights = vectors[0].square()
    out = {"steps": len(nodes), "ritz_values": nodes.tolist(), "rhs_weights": weights.tolist(),
           "mass_estimates": {}, "signed_resolvent_estimates": {},
           "note": "Finite quadrature estimates. No certified CDF interval or full spectral distribution. "
                   "A signed resolvent near one does not establish x = b/gamma."}
    for threshold in (0., .01, .03, .1, .3, .7, 1., 2., 5.):
        out["mass_estimates"][str(threshold)] = float(weights[nodes < threshold].sum())
    for gamma in gammas:
        shifted = nodes + gamma
        gap = float(shifted.abs().min())
        safe = gap > 1e-12 * max(1., float(shifted.abs().max()))
        out["signed_resolvent_estimates"][str(gamma)] = {
            "value": float((weights * gamma / shifted).sum()) if safe else None,
            "projected_pole_distance": gap,
            "warning": "Projected pole distance is not a lower bound on the true spectral gap"}
    return out


def reverse_residual(matvec, d, b, alpha, gammas):
    Hd = matvec(d).double()
    dv, rhs = d.double(), alpha * b.double()
    dd, rn = float(dv @ dv), core.norm(rhs)
    implied = float(dv @ (rhs - Hd)) / dd if dd else None

    def residual(gamma):
        error = core.norm(Hd + gamma * dv - rhs)
        denominator = core.norm(Hd) + abs(gamma) * core.norm(dv) + rn
        return {"absolute": error, "relative_to_rhs": error / rn if rn else None,
                "scaled_equation_residual": error / denominator if denominator else None}

    nonnegative = max(0., implied) if implied is not None else None
    return {"implied_gamma_unconstrained": implied, "best_nonnegative_gamma": nonnegative,
            "unconstrained_residual": residual(implied) if implied is not None else None,
            "nonnegative_residual": residual(nonnegative) if nonnegative is not None else None,
            "grid_residuals": {str(g): residual(g) for g in gammas},
            "true_norm": core.norm(d), "rhs_norm": rn, "Hd_norm": core.norm(Hd),
            "d_rayleigh": float(dv @ Hd) / dd if dd else None,
            "cosine_Hd_rhs": core.cosine(Hd, rhs),
            "interpretation": "Equation consistency at this finite alpha and eval objective. "
                              "Does not identify chaos or prove convergence of the dropout objective."}
