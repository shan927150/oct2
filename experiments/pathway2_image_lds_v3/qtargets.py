"""Query targets normalised by IMAGE count (mentor 2026-10-09), plus the frozen
patient-macro weighting as a secondary target.

For one query image with logits z, p = softmax(z) and label y:
  CE        l = -log p_y              u = dl/dz = p - e_y
  1 - p_y   l = 1 - p_y               u = p_y (p - e_y)
  Brier     l = sum_c (p_c - e_c)^2   u = 2 [p*(p - e) - p <p, p - e>]
The parameter gradient of l is J_z(theta)^T u, so per-image query gradients for
all three targets come from one logit Jacobian per query image.
"""
import numpy as np

TARGETS = ('oneminusp', 'brier', 'ce')
WEIGHTINGS = ('img', 'macro')


def _prep(log_probs, labels):
    lp = np.asarray(log_probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if lp.ndim != 2 or len(lp) != len(y) or not np.isfinite(lp).all():
        raise FloatingPointError('Invalid log probabilities')
    if np.any(y < 0) or np.any(y >= lp.shape[1]):
        raise ValueError('Invalid labels')
    p = np.exp(lp)
    e = np.zeros_like(p)
    e[np.arange(len(y)), y] = 1.
    return lp, y, p, e


def per_image_loss(name, log_probs, labels):
    lp, y, p, e = _prep(log_probs, labels)
    rows = np.arange(len(y))
    if name == 'ce':
        return -lp[rows, y]
    if name == 'oneminusp':
        return 1. - p[rows, y]
    if name == 'brier':
        return ((p - e) ** 2).sum(1)
    raise ValueError('Unknown target ' + str(name))


def logit_derivative(name, log_probs, labels):
    """u_i = d loss_i / d logits_i, shape [n, C]."""
    lp, y, p, e = _prep(log_probs, labels)
    rows = np.arange(len(y))
    if name == 'ce':
        return p - e
    if name == 'oneminusp':
        return p[rows, y][:, None] * (p - e)
    if name == 'brier':
        d = p - e
        return 2. * (p * d - p * (p * d).sum(1, keepdims=True))
    raise ValueError('Unknown target ' + str(name))


def image_weights(n_query):
    return np.full(n_query, 1. / n_query)


def values(log_probs_test, labels_test, macro_weights, units):
    """Image-mean Y, patient-macro Y, per-image and per-query-patient losses."""
    w_img = image_weights(len(labels_test))
    out = {}
    for t in TARGETS:
        loss = per_image_loss(t, log_probs_test, labels_test)
        out[t] = dict(Y_img=float(w_img @ loss), Y_macro=float(np.asarray(macro_weights) @ loss),
                      Yi=loss, Yq=np.asarray([loss[u].mean() for u in units]))
    return out
