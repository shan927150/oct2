"""Stage-1 query targets Y(q) and their exact interface derivative v = dY/dq.

Every target is a weighted sum of per-image losses on the frozen heldout rows.
The macro target uses the frozen class-balanced patient-macro weights of the
LDS v1 design; per-query targets use the within-patient mean (weights 1/n_q).
No clipping: a non-positive probability under CE raises instead of being hidden.
"""
import numpy as np

TARGETS = ('oneminusp', 'brier', 'ce')


def _check(probs, labels, weights):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if probs.ndim != 2 or len(probs) != len(labels) or len(labels) != len(weights):
        raise ValueError('probs/labels/weights shape mismatch')
    if not np.isfinite(probs).all():
        raise FloatingPointError('Non-finite probabilities')
    return probs, labels, weights


def per_image(name, probs, labels):
    probs, labels, _ = _check(probs, labels, np.zeros(len(labels)))
    rows = np.arange(len(labels))
    py = probs[rows, labels]
    if name == 'oneminusp':
        return 1. - py
    if name == 'brier':
        onehot = np.zeros_like(probs)
        onehot[rows, labels] = 1.
        return ((probs - onehot) ** 2).sum(1)
    if name == 'ce':
        if np.any(py <= 0):
            raise FloatingPointError('CE probability is not positive')
        return -np.log(py)
    raise ValueError('Unknown target ' + str(name))


def value_and_v(name, probs, labels, weights):
    """Return Y = sum_i w_i loss_i(q_i) and v_i = w_i d loss_i / d q_i."""
    probs, labels, weights = _check(probs, labels, weights)
    rows = np.arange(len(labels))
    v = np.zeros_like(probs)
    if name == 'oneminusp':
        v[rows, labels] = -weights
    elif name == 'brier':
        onehot = np.zeros_like(probs)
        onehot[rows, labels] = 1.
        v = 2. * weights[:, None] * (probs - onehot)
    elif name == 'ce':
        py = probs[rows, labels]
        if np.any(py <= 0):
            raise FloatingPointError('CE probability is not positive')
        v[rows, labels] = -weights / py
    else:
        raise ValueError('Unknown target ' + str(name))
    return float(weights @ per_image(name, probs, labels)), v


def query_units(design):
    """Per-query-patient positions inside the frozen heldout row order.

    Returns (patient ids, list of position arrays, macro coefficient per patient)
    with sum_q coef_q * mean_{i in q} loss_i == macro Y exactly.
    """
    test = np.asarray(design['query_rows'], dtype=np.int64)
    position = {int(r): i for i, r in enumerate(test)}
    classes = [p['oct_class'] for p in design['query_patients']]
    per_class = {c: classes.count(c) for c in set(classes)}
    ids, units, coef = [], [], []
    for p in design['query_patients']:
        ids.append(int(p['patient']))
        units.append(np.asarray([position[int(r)] for r in p['rows']], dtype=np.int64))
        coef.append(1. / (len(per_class) * per_class[p['oct_class']]))
    return np.asarray(ids), units, np.asarray(coef)


def per_query_values(name, probs_test, labels_test, units):
    """Within-patient mean loss for every query patient (shape [Q])."""
    loss = per_image(name, probs_test, labels_test)
    return np.asarray([loss[u].mean() for u in units])


def all_values(probs_test, labels_test, weights, units):
    """Macro Y and per-query Y_q for every target, from stored probabilities."""
    out = {}
    for name in TARGETS:
        Y, _ = value_and_v(name, probs_test, labels_test, weights)
        out[name] = dict(Y=Y, Yq=per_query_values(name, probs_test, labels_test, units))
    return out
