"""Subset-level statistics. Shared seed columns are never pseudo-replicated rows."""
import numpy as np
from scipy.stats import rankdata


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    x, y = rankdata(x), rankdata(y)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator else None


def evaluate(pred, target, seed=20260929, draws=2000):
    pred, target = np.asarray(pred), np.asarray(target)
    rho = spearman(pred, target)
    if rho is None:
        return dict(spearman=None, reason='constant or nonfinite prediction/target')
    rng = np.random.default_rng(seed)
    boots, null = [], []
    for _ in range(draws):
        i = rng.integers(len(pred), size=len(pred))
        r = spearman(pred[i], target[i])
        if r is not None:
            boots.append(r)
        null.append(spearman(pred, rng.permutation(target)))
    # Slope/intercept describe calibration; no fitting is used for LDS or error.
    slope = float(np.cov(pred, target, ddof=1)[0,1]/np.var(pred, ddof=1))
    scale = float(np.linalg.norm(target))
    return dict(spearman=rho, bootstrap_ci_conditional_on_seed_panel=np.quantile(boots,[.025,.975]).tolist(),
                permutation_p_one_sided=(1+sum(x >= rho for x in null))/(draws+1),
                slope_truth_on_prediction=slope,
                intercept=float(target.mean()-slope*pred.mean()),
                uncalibrated_rmse=float(np.sqrt(np.mean((pred-target)**2))),
                relative_error=float(np.linalg.norm(pred-target)/scale) if scale else None,
                n_subsets=len(pred), valid_bootstraps=len(boots))


def reliability(y):
    """Descriptive shared-seed repeatability; k=3 is not a variance proof."""
    y = np.asarray(y, dtype=float)
    m, k = y.shape
    pairwise = [dict(seed_column_a=i,seed_column_b=j,spearman=spearman(y[:,i],y[:,j]))
                for i in range(k) for j in range(i+1,k)]
    centered = y-y.mean(0,keepdims=True)
    # Random-effects descriptive decomposition removes additive seed intercepts.
    residual = centered-centered.mean(1,keepdims=True)
    noise = float((residual**2).sum()/((m-1)*(k-1)))
    between = float(np.var(y.mean(1),ddof=1))
    signal = max(0.,between-noise/k)
    return dict(seed_pairwise_rank_correlations=pairwise,
                subset_by_seed_interaction_variance=noise,
                estimated_subset_signal_variance=signal,
                reliability_of_seed_mean=signal/between if between else None,
                leave_one_seed_out_mean_correlations=[spearman(y.mean(1),np.delete(y,i,axis=1).mean(1)) for i in range(k)],
                caveat='Descriptive with shared seeds; no claim of a Spearman upper bound.')


def crossfit_ridge(masks, target):
    """Nested leave-one-subset-out ridge. Never fit on the scored subset.

    Diagnostic comparator with m << patient count, NOT an oracle/upper bound.
    Hyperparameters chosen only on inner training folds; group = subset id.
    """
    X, y = np.asarray(masks,float), np.asarray(target,float)
    if len(y) < 8:
        return None
    def fit_predict(train, test, alpha):
        xm, ym = X[train].mean(0), y[train].mean()
        A = X[train]-xm
        coef = A.T@np.linalg.solve(A@A.T+alpha*np.eye(len(train)), y[train]-ym)
        return (X[test]-xm)@coef+ym
    grid = [.1, 1., 10., 100.]
    out = np.zeros(len(y))
    for i in range(len(y)):
        outer = np.asarray([j for j in range(len(y)) if j != i])
        loss = []
        for alpha in grid:
            err = [float((fit_predict(outer[outer!=j],[j],alpha)[0]-y[j])**2) for j in outer]
            loss.append(np.mean(err))
        out[i] = fit_predict(outer,[i],grid[int(np.argmin(loss))])[0]
    return out
