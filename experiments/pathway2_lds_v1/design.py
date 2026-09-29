"""Patient masks and the precise scalar estimand; no training imports."""
import hashlib
import numpy as np


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def patient_table(indices, groups, labels):
    table = []
    for pid in sorted(set(map(int, groups[indices]))):
        rows = indices[groups[indices] == pid]
        classes = np.unique(labels[rows])
        if len(classes) != 1:
            raise ValueError('Mixed-class patient is outside this design: ' + str(pid))
        table.append(dict(patient=pid, oct_class=int(classes[0]), rows=rows.tolist()))
    return table


def make_design(train, test, groups, labels, m, retain, subset_seed):
    """Uniform fixed-count patient subsets, independently sampled, duplicates rejected.

    No complement pairing or class stratification: same distribution as a uniform
    patient-cluster extension of LDS. Reject duplicate sets, not bad outcomes.
    """
    train, test = np.asarray(train), np.asarray(test)
    t, q = patient_table(train, groups, labels), patient_table(test, groups, labels)
    if set(x['patient'] for x in t) & set(x['patient'] for x in q):
        raise ValueError('Training/query patients overlap')
    if set(x['oct_class'] for x in q) != {0, 1, 2, 3}:
        raise ValueError('All four heldout classes are required')
    n = len(t)
    keep = round(n * retain)
    if not 0 < keep < n:
        raise ValueError('Invalid retained patient count')
    import math
    if m > math.comb(n, keep):
        raise ValueError('Too many distinct subsets')
    rng = np.random.default_rng(subset_seed)
    masks, seen = [], set()
    while len(masks) < m:
        selected = tuple(sorted(rng.choice(n, keep, replace=False).tolist()))
        if selected in seen:
            continue
        seen.add(selected)
        mask = np.zeros(n, dtype=np.int8)
        mask[list(selected)] = 1
        masks.append(mask.tolist())
    # Scalar Y = equal-class, equal-patient, within-patient mean heldout CE.
    # This is one prespecified aggregate query, not the paper's mean per-image LDS.
    weight_by_row = {}
    for cls in range(4):
        patients = [x for x in q if x['oct_class'] == cls]
        for p in patients:
            for row in p['rows']:
                weight_by_row[row] = 1. / (4 * len(patients) * len(p['rows']))
    weights = np.asarray([weight_by_row[int(row)] for row in test])
    if not np.isclose(weights.sum(), 1.):
        raise ValueError('Query weights do not sum to one')
    return dict(train_patients=t, query_patients=q, query_rows=test.tolist(),
                query_weights=weights.tolist(), masks=masks,
                group_sha256=array_sha(groups), label_sha256=array_sha(labels),
                subset_seed=subset_seed, retained_patient_count=keep,
                estimand='class-balanced patient-macro heldout CE; raw endpoint')


def intervention(d, subset, mode, alpha, n):
    """Return removed rows and objective weight differences from full-data mean CE.

    subset objective: mean over retained images + SAME L2 (filter/rechunk).
    fixed_mask objective: original-N CE weights 1-alpha outside S + SAME L2.
    b_p is always sum patient gradients / original N.
    """
    mask = np.asarray(d['masks'][subset], dtype=float)
    removed = [r for keep, p in zip(mask, d['train_patients']) if not keep for r in p['rows']]
    if mode == 'fixed_mask':
        delta_weight = -alpha * (1 - mask)
    elif mode == 'subset':
        if alpha != 1:
            raise ValueError('Subset mode requires alpha=1')
        retained = n - len(removed)
        if retained <= 0:
            raise ValueError('Empty subset')
        delta_weight = n / retained * mask - 1
    else:
        raise ValueError('Unknown intervention mode')
    return np.asarray(removed, dtype=np.int64), delta_weight


def output_and_v(probs, labels, weights):
    """Exact CE and dY/dP; fail on unrepresentable probabilities, never clip silently."""
    probs = np.asarray(probs, dtype=np.float64)
    py = probs[np.arange(len(labels)), labels]
    if not np.isfinite(probs).all() or np.any(py <= 0):
        raise FloatingPointError('CE probabilities are not finite and positive')
    v = np.zeros_like(probs)
    v[np.arange(len(labels)), labels] = -np.asarray(weights) / py
    return float(-np.asarray(weights) @ np.log(py)), v
