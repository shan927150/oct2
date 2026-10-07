"""Multi-shift Lanczos/Galerkin solve of (H + gamma I) x = b for several gamma.

The Krylov space K_k(H + gamma I, b) = K_k(H, b) does not depend on gamma, so
one Lanczos run with b as start vector gives, for every gamma, the same iterate
that CG would produce after k steps (when T_k + gamma I is positive definite):
    x_gamma = ||b|| Q_k (T_k + gamma I)^{-1} e_1,
    ||(H + gamma I) x_gamma - b|| = beta_k |e_k^T y_gamma|  (exact arithmetic).
Full re-orthogonalisation (two passes) keeps Q_k orthonormal in floating point.
The caller must still recompute the TRUE residual with explicit matvecs, and
treat min Ritz(T_k) + gamma <= 0 as unqualified for the positive-curvature solve.
Failure of this screen does not prove that the mathematical system is singular.
The smallest Ritz value decreases monotonically with k, so checking the final
T_k covers every earlier CG step.
"""
import numpy as np


def _solve_small(T, gamma, rhs_norm):
    k = len(T)
    e1 = np.zeros(k)
    e1[0] = rhs_norm
    return np.linalg.solve(T + gamma * np.eye(k), e1)


def multishift_lanczos(matvec, b, gammas, max_steps, tol, dot=None, reorth_passes=2):
    """matvec: undamped operator. tol: estimated relative residual for every gamma.

    Works with torch or numpy vectors (only +, -, scalar *, dot and norm are used).
    Returns basis-free results: solutions x[gamma], tridiagonal, residual history.
    """
    gammas = [float(g) for g in gammas]
    if (not gammas or len(set(gammas)) != len(gammas)
            or any(not np.isfinite(g) or g <= 0 for g in gammas)
            or max_steps < 1 or not np.isfinite(tol) or tol < 0):
        raise ValueError('Invalid gamma panel or Lanczos budget/tolerance')
    if dot is None:
        dot = lambda a, c: float((a * c).sum())
    beta0 = dot(b, b) ** .5
    if not np.isfinite(beta0) or beta0 == 0:
        raise ValueError('Right-hand side must be finite and non-zero')
    q = b / beta0
    basis, alphas, betas = [], [], []
    q_prev, beta_prev = None, 0.
    history = []
    breakdown = False
    for step in range(1, max_steps + 1):
        basis.append(q)
        z = matvec(q)
        alpha = dot(q, z)
        if not np.isfinite(alpha):
            raise FloatingPointError('Non-finite Lanczos coefficient')
        z = z - alpha * q
        if q_prev is not None:
            z = z - beta_prev * q_prev
        for _ in range(reorth_passes):
            for v in basis:
                z = z - dot(v, z) * v
        alphas.append(alpha)
        beta = dot(z, z) ** .5
        if not np.isfinite(beta):
            raise FloatingPointError('Non-finite Lanczos residual norm')
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        est = {}
        for g in gammas:
            try:
                y = _solve_small(T, g, beta0)
                est[g] = float(beta * abs(y[-1]) / beta0)
            except np.linalg.LinAlgError:
                est[g] = float('inf')
        history.append(dict(step=step, beta=beta, ritz_min=float(np.linalg.eigvalsh(T)[0]),
                            **{f'est_rel_residual_gamma{g:g}': est[g] if np.isfinite(est[g]) else None for g in gammas}))
        if beta <= 1e-12 * max(1., abs(alpha)):
            breakdown = True  # invariant subspace: Galerkin solution is exact
            break
        if all(e <= tol for e in est.values()):
            break
        betas.append(beta)
        q_prev, q, beta_prev = q, z / beta, beta
    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:k - 1], 1) + np.diag(betas[:k - 1], -1)
    ritz = np.linalg.eigvalsh(T)
    solutions, errors = {}, {}
    for g in gammas:
        try:
            y = _solve_small(T, g, beta0)
            if not np.isfinite(y).all():
                raise FloatingPointError('non-finite projected solution')
            x = float(y[0]) * basis[0]
            for c, v in zip(y[1:], basis[1:]):
                x = x + float(c) * v
            if not np.isfinite(dot(x, x)):
                raise FloatingPointError('non-finite solution norm')
            solutions[g] = x
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            errors[g] = str(exc)  # A failed shift never discards other shifts.
    return dict(x=solutions, errors=errors, steps=k, ritz_min=float(ritz[0]), ritz_max=float(ritz[-1]),
                ritz_values=ritz.tolist(), alphas=alphas, betas=betas[:k - 1],
                history=history, invariant_subspace=breakdown, rhs_norm=beta0,
                final_estimated_rel_residual={g: history[-1][f'est_rel_residual_gamma{g:g}'] for g in gammas})


def probe_screen(probe, gamma):
    """Random-start Lanczos screen (legacy lanczos_probe output) at one gamma.

    Gate: min Ritz + gamma must exceed the absolute Ritz residual. The endpoint
    residual tolerance is reported, not gated: a short probe rarely converges
    the low end, and an unconverged positive Ritz value is not an SPD proof.
    """
    low = probe['endpoints']['min']
    return dict(gamma=gamma, damped_min_ritz=low['value'] + gamma, min_ritz_residual_abs=low['residual_abs'],
                positive=bool(low['value'] + gamma > low['residual_abs']),
                min_endpoint_residual_scaled=low['residual_scaled'],
                not_spd_certificate=True)


def qualify(gamma, true_rel_residual, krylov_ritz_min, screen, fail_tol):
    """Per-gamma numerical gate; failure is not a proof of noninvertibility."""
    reasons = []
    if true_rel_residual is None or not np.isfinite(true_rel_residual) or true_rel_residual > fail_tol:
        reasons.append('true relative residual above tolerance')
    if not krylov_ritz_min + gamma > 0:
        reasons.append('non-positive curvature in the solve Krylov space')
    if screen is not None and not screen['positive']:
        reasons.append('random-probe Ritz value not positive after damping')
    return dict(gamma=gamma, qualified=not reasons, reasons=reasons,
                true_rel_residual=true_rel_residual, damped_krylov_ritz_min=krylov_ritz_min + gamma)
