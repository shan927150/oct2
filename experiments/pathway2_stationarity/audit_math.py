"""Vector Monte Carlo moments, including covariance for paired g - alpha*b.

NumPy only. Trace covariance is enough to estimate E||mean - E mean||^2;
it does not give a confidence region or a calibrated significance test.
"""
import math
import numpy as np


class VectorMoments:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, vectors):
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite [vector, parameter] array')
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
        if x.shape != self.mean.shape:
            raise ValueError('Vector shape changed')
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += np.einsum('ip,jp->ij', delta, x - self.mean)

    def se(self, weights):
        if self.n < 2:
            raise ValueError('At least two MC draws required')
        w = np.asarray(weights, dtype=np.float64)
        variance = float(w @ self.m2 @ w) / (self.n * (self.n - 1))
        return math.sqrt(max(0., variance))


def norm(x):
    return float(np.linalg.norm(x))


def ratio(a, b):
    return a / b if b > 0 else None


def cosine(a, b):
    denom = norm(a) * norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1., 1.)) if denom else None


def summary_rows(vectors, patients, moments=None):
    """Rows use b_p = sum(patient gradients)/N; L2 occurs only in g."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[0] != 1 + len(patients) or not np.isfinite(v).all():
        raise ValueError('Invalid vectors/patient panel')
    basis = np.eye(len(v))
    g, ng = v[0], norm(v[0])
    se_g = moments.se(basis[0]) if moments else 0.
    rows = []
    for j, patient in enumerate(patients, 1):
        b, nb = v[j], norm(v[j])
        se_b = moments.se(basis[j]) if moments else 0.
        for alpha in (.1, 1.):
            r = g - alpha * b
            se_r = moments.se(basis[0] - alpha * basis[j]) if moments else 0.
            rows.append(dict(patient_id=patient, alpha=alpha,
                g_norm=ng, b_norm=nb, alpha_b_norm=alpha * nb,
                residual_over_perturbation=ratio(ng, alpha * nb),
                cosine_g_b=cosine(g, b), deleted_objective_grad_norm=norm(r),
                g_vector_se=se_g, b_vector_se=se_b, deleted_grad_vector_se=se_r,
                g_se_over_norm=ratio(se_g, ng), b_se_over_norm=ratio(se_b, nb),
                # Signed estimates can be negative when MC noise dominates.
                g_norm_sq_minus_mc_variance=ng * ng - se_g * se_g,
                b_norm_sq_minus_mc_variance=nb * nb - se_b * se_b))
    return rows
