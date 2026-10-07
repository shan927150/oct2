"""Seed-curve, deleter-LDS, size-control and per-query statistics.

Statistical unit = subset (never subset x seed). Seed draws are shared across
predictors inside one k so predictor curves are paired comparisons.
"""
import numpy as np
from analysis_stats import spearman, evaluate, crossfit_ridge  # LDS v1, hashed in config


# ---------------------------------------------------------------- predictions
def deleter_prediction(masks, tau):
    """Task 2: sum over DELETED patients of the predicted deletion effect -tau_p.

    tau is the upweight score, so deleting p changes Y by -tau_p (alpha=1).
    Identity: deleter = retained_sum - sum(tau); identical ranks/LDS.
    """
    M = np.asarray(masks, dtype=float)
    return -(1. - M) @ np.asarray(tau, dtype=float)


def retained_sum(masks, tau):
    return np.asarray(masks, dtype=float) @ np.asarray(tau, dtype=float)


def matched_prediction(masks, tau, images_per_patient):
    """First-order effect with the actual subset-mode normalization
    delta_w_jp = (N/N_j) m_jp - 1 (mean CE over retained images)."""
    M = np.asarray(masks, dtype=float)
    n = np.asarray(images_per_patient, dtype=float)
    N, Nj = n.sum(), M @ n
    return ((N / Nj)[:, None] * M - 1.) @ np.asarray(tau, dtype=float)


def size_baseline(n_images):
    """Prespecified naive predictor: fewer retained images -> larger heldout loss."""
    return -np.asarray(n_images, dtype=float)


# --------------------------------------------------------------- size control
def residualize(x, z):
    x, z = np.asarray(x, float), np.asarray(z, float)
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _collinear(resid, x):
    x = np.asarray(x, float)
    return np.linalg.norm(resid) <= 1e-9 * max(np.linalg.norm(x - x.mean()), 1e-300)


def residual_spearman(pred, truth, z):
    """Spearman after removing the linear effect of z from BOTH sides
    (the convention behind the 0.119 number in the 2026-10-05 analysis).
    Undefined (None) when the predictor is itself a linear function of z."""
    if np.ptp(np.asarray(z, float)) == 0:
        return spearman(pred, truth)
    rp = residualize(pred, z)
    if _collinear(rp, pred):
        return None
    return spearman(rp, residualize(truth, z))


def partial_spearman(pred, truth, z):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(pred), rankdata(truth), rankdata(z)
    if np.ptp(rz) == 0:
        return spearman(pred, truth)
    a, b = residualize(rx, rz), residualize(ry, rz)
    if _collinear(a, rx):
        return None
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else None


def lds_block(pred, truth, n_images, seed=20261006, draws=2000):
    """One LDS cell with the size controls that must accompany every LDS."""
    out = evaluate(pred, truth, seed=seed, draws=draws)
    out['residual_spearman_given_images'] = residual_spearman(pred, truth, n_images)
    out['partial_rank_spearman_given_images'] = partial_spearman(pred, truth, n_images)
    return out


def critical_spearman(m, alpha=.05, draws=20000, seed=7):
    """One-sided permutation critical value for Spearman with m subsets."""
    rng = np.random.default_rng(seed)
    x = np.arange(m)
    null = [spearman(x, rng.permutation(m)) for _ in range(draws)]
    return float(np.quantile(null, 1 - alpha))


# ----------------------------------------------------------------- seed curves
def spearman_brown(r1, k):
    k = np.asarray(k, float)
    return k * r1 / (1 + (k - 1) * r1)


def variance_components(Y):
    """Two-way (subset x seed) decomposition with additive seed intercepts removed."""
    Y = np.asarray(Y, float)
    m, S = Y.shape
    centered = Y - Y.mean(0, keepdims=True)
    resid = centered - centered.mean(1, keepdims=True)
    noise = float((resid ** 2).sum() / ((m - 1) * (S - 1)))
    between = float(np.var(Y.mean(1), ddof=1))
    signal = max(0., between - noise / S)
    return dict(noise_variance=noise, signal_variance=signal, m=m, seeds=S)


def se_curve(Y, ks, draws=500, seed=20261006):
    """Standard error of the k-seed subset mean, analytic and resampled."""
    Y = np.asarray(Y, float)
    m, S = Y.shape
    sd = Y.std(1, ddof=1)
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        fpc = np.sqrt((S - k) / (S - 1)) if S > 1 else 0.
        analytic = float(np.mean(sd / np.sqrt(k)))
        means = np.stack([Y[:, rng.choice(S, k, replace=False)].mean(1) for _ in range(draws)])
        rows.append(dict(k=int(k), analytic_se_mean_over_subsets=analytic,
                         analytic_se_with_finite_population_correction=float(analytic * fpc),
                         resampled_sd_of_k_seed_mean=float(means.std(0, ddof=1).mean())))
    return rows


def reliability_curve(Y, ks, draws=500, seed=20261006):
    """Correlation between two DISJOINT k-seed means (k <= S/2), with the
    Spearman-Brown prediction from k=1 and the variance-component (ICC) curve."""
    Y = np.asarray(Y, float)
    m, S = Y.shape
    rng = np.random.default_rng(seed)
    vc = variance_components(Y)
    rows = []
    for k in ks:
        row = dict(k=int(k))
        if 2 * k <= S:
            rs, ps = [], []
            for _ in range(draws):
                perm = rng.permutation(S)
                a, b = Y[:, perm[:k]].mean(1), Y[:, perm[k:2 * k]].mean(1)
                r = spearman(a, b)
                if r is not None:
                    rs.append(r)
                ps.append(float(np.corrcoef(a, b)[0, 1]))
            row.update(split_spearman_mean=float(np.mean(rs)) if rs else None,
                       split_spearman_q025=float(np.quantile(rs, .025)) if rs else None,
                       split_spearman_q975=float(np.quantile(rs, .975)) if rs else None,
                       split_pearson_mean=float(np.nanmean(ps)))
        tot = vc['signal_variance'] + vc['noise_variance'] / k
        row['icc_reliability'] = vc['signal_variance'] / tot if tot > 0 else None
        rows.append(row)
    r1 = rows[0].get('split_spearman_mean') if rows and rows[0]['k'] == 1 else None
    for row in rows:
        row['spearman_brown_from_k1'] = float(spearman_brown(r1, row['k'])) if r1 is not None else None
    return dict(rows=rows, variance_components=vc,
                note='Split-half Spearman with few subsets is noisy; ICC uses all seeds.')


def lds_curve(truth, predictors, ks, draws=300, seed=20261006):
    """truth: [m, S] seed-level Delta Y. predictors: name -> f(seed_indices) -> [m].

    Truth is the mean over a random k-seed draw; every predictor sees the SAME draw.
    """
    truth = np.asarray(truth, float)
    m, S = truth.shape
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        n = 1 if k == S else draws
        sets = [np.arange(S) if k == S else rng.choice(S, k, replace=False) for _ in range(n)]
        for name, f in predictors.items():
            vals = [spearman(f(s), truth[:, s].mean(1)) for s in sets]
            vals = [v for v in vals if v is not None]
            rows.append(dict(k=int(k), predictor=name, draws=len(vals),
                             mean=float(np.mean(vals)) if vals else None,
                             q025=float(np.quantile(vals, .025)) if vals else None,
                             q975=float(np.quantile(vals, .975)) if vals else None))
    return rows


# ------------------------------------------------------------- per-query LDS
def _standardized_ranks(A):
    from scipy.stats import rankdata
    R = rankdata(A, axis=0)
    R = R - R.mean(0, keepdims=True)
    n = np.linalg.norm(R, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return R / n, n > 0


def per_query_lds(pred, truth, draws=1000, seed=20261006):
    """TRAK-style: Spearman across subsets for every query unit, then average.

    pred, truth: [m, Q]. Null: one subset permutation applied to all queries
    jointly (keeps the cross-query dependence). Constant columns -> undefined."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    m, Q = truth.shape
    zp, okp = _standardized_ranks(pred)
    zt, okt = _standardized_ranks(truth)
    ok = okp & okt
    per = np.where(ok, (zp * zt).sum(0), np.nan)
    observed = float(np.nanmean(per))
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        null[i] = np.nanmean(np.where(ok, (zp * zt[rng.permutation(m)]).sum(0), np.nan))
    return dict(per_query=per.tolist(), mean=observed, median=float(np.nanmedian(per)),
                fraction_positive=float(np.mean(per[ok] > 0)) if ok.any() else None, defined=int(ok.sum()),
                n_queries=Q, n_subsets=m,
                permutation_p_one_sided=float((1 + np.sum(null >= observed)) / (draws + 1)),
                null_mean=float(np.mean(null)), null_q95=float(np.quantile(null, .95)))
