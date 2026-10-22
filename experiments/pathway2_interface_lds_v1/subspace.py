"""Galerkin solves of (H + gamma I) x = b for many right-hand sides on one shared, growing subspace.

The basis Q (rows, orthonormal up to roundoff after two-pass Gram-Schmidt) and HQ = H Q are stored.
For a right-hand side b and every damping gamma:
    y = (Q H Q^T + gamma Q Q^T)^{-1} Q b,   x = Q^T y,   r = b - HQ^T y - gamma Q^T y.
The residual is exact given HQ, so no extra matvec is needed to check it. While some gamma misses its
tolerance, the residual with the largest ratio residual / tolerance is orthogonalised against Q and
appended (one matvec per new direction).

From an empty basis this is Lanczos / CG: the Galerkin residuals of all gammas lie in K_{k+1}(H, b) and
are orthogonal to K_k(H, b), so they are collinear and multi-shift Lanczos is recovered. Later right-hand
sides start from every direction found so far; only what they need beyond it costs matvecs. The smallest
Ritz value of the projected pencil can only decrease as the basis grows (interlacing), so the value at the
end of a seed bounds every solve made on a smaller basis.

Works on any torch device and dtype; the small projected systems are solved in float64 on the same device.
"""
import math
import torch


class SharedGalerkin:
    def __init__(self, matvec, dim, max_dim, dtype=torch.float64, device='cpu'):
        if max_dim < 1 or dim < 1:
            raise ValueError('Invalid subspace dimensions')
        self.matvec = matvec
        max_dim = min(max_dim, dim)
        self.Q = torch.zeros((max_dim, dim), dtype=dtype, device=device)
        self.HQ = torch.zeros((max_dim, dim), dtype=dtype, device=device)
        self.A = torch.zeros((max_dim, max_dim), dtype=torch.float64, device=device)   # Q H Q^T
        self.G = torch.zeros((max_dim, max_dim), dtype=torch.float64, device=device)   # Q Q^T
        self.m = 0
        self.matvecs = 0

    # ------------------------------------------------------------------ basis
    def _append(self, v):
        """Orthogonalise v against Q twice; append it with its matvec. Returns False if v is in span(Q)."""
        if self.m >= self.Q.shape[0]:
            return False
        norm0 = float(v.norm())
        if not norm0 > 0 or not torch.isfinite(v).all():
            return False
        for _ in range(2):
            if self.m:
                v = v - (self.Q[:self.m] @ v) @ self.Q[:self.m]
        norm = float(v.norm())
        if not norm > 1e-12 * norm0:
            return False
        q = v / norm
        hq = self.matvec(q)
        self.matvecs += 1
        if not torch.isfinite(hq).all():
            raise FloatingPointError('Non-finite matvec')
        m = self.m
        self.Q[m], self.HQ[m] = q, hq
        a = (self.Q[:m + 1] @ hq).to(torch.float64)
        b = (self.Q[:m + 1] @ q).to(torch.float64)
        self.A[m, :m + 1] = a
        self.A[:m + 1, m] = a
        self.G[m, :m + 1] = b
        self.G[:m + 1, m] = b
        self.m = m + 1
        return True

    def _project(self, b, gammas):
        m = self.m
        A = .5 * (self.A[:m, :m] + self.A[:m, :m].T)
        Gm = .5 * (self.G[:m, :m] + self.G[:m, :m].T)
        c = (self.Q[:m] @ b).to(torch.float64)
        out = {}
        self.projection_failures = []
        for g in gammas:
            try:
                y = torch.linalg.solve(A + g * Gm, c)
            except torch.linalg.LinAlgError:
                self.projection_failures.append(g)
                out[g] = (torch.zeros_like(b), b.clone())
                continue
            yd = y.to(self.Q.dtype)
            x = yd @ self.Q[:m]
            r = b - yd @ self.HQ[:m] - g * x
            if not torch.isfinite(x).all() or not torch.isfinite(r).all():
                self.projection_failures.append(g)
                x, r = torch.zeros_like(b), b.clone()
            out[g] = (x, r)
        return out

    def ritz_min(self):
        """Smallest Ritz value of the pencil (Q H Q^T, Q Q^T) on the current basis."""
        m = self.m
        if not m:
            return None
        A = .5 * (self.A[:m, :m] + self.A[:m, :m].T)
        Gm = .5 * (self.G[:m, :m] + self.G[:m, :m].T)
        L = torch.linalg.cholesky(Gm)
        Li = torch.linalg.solve_triangular(L, torch.eye(m, dtype=A.dtype, device=A.device), upper=False)
        return float(torch.linalg.eigvalsh(Li @ A @ Li.T)[0])

    # ------------------------------------------------------------------ solve
    def solve(self, b, gammas, tols, max_new):
        """x[gamma] with exact Galerkin residuals; expands the shared basis as needed (at most max_new)."""
        gammas = [float(g) for g in gammas]
        if (not gammas or len(set(gammas)) != len(gammas)
                or any(not math.isfinite(g) or g <= 0 for g in gammas)
                or len(tols) != len(gammas) or any(not math.isfinite(t) or t <= 0 for t in tols)
                or max_new < 1):
            raise ValueError('Distinct finite positive gammas and tolerances, and max_new >= 1, are required')
        if b.shape != (self.Q.shape[1],) or not torch.isfinite(b).all():
            raise ValueError('Right-hand side must be finite with the expected dimension')
        bn = float(b.norm())
        if bn == 0:
            return dict(x={g: torch.zeros_like(b) for g in gammas}, rel_residual={g: 0. for g in gammas},
                        converged=True, new_vectors=0, dim=self.m, stalled=False, capped=False,
                        projection_failures=[])
        start, stalled, capped = self.matvecs, False, False
        if self.m == 0:
            self._append(b.clone())
        while True:
            sol = self._project(b, gammas)
            rel = {g: float(r.norm()) / bn for g, (_, r) in sol.items()}
            ratio = {g: rel[g] / t for g, t in zip(gammas, tols)}
            worst = max(ratio, key=ratio.get)
            if ratio[worst] <= 1:
                break
            if self.matvecs - start >= max_new or self.m >= self.Q.shape[0]:
                capped = True
                break
            # A singular auxiliary shift must not prevent other shifts from expanding their basis.
            candidates = sorted((g for g in gammas if ratio[g] > 1), key=ratio.get, reverse=True)
            if not any(self._append(sol[g][1].clone()) for g in candidates):
                stalled = True
                break
        return dict(x={g: sol[g][0] for g in gammas}, rel_residual=rel, converged=all(v <= 1 for v in ratio.values()),
                    new_vectors=self.matvecs - start, dim=self.m, stalled=stalled, capped=capped,
                    projection_failures=list(self.projection_failures))
