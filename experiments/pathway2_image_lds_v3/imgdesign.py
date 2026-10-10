"""Image-level random subsets (standard TRAK-style LDS, mentor 2026-10-09).

Each subset keeps exactly K = floor(N * retain) training IMAGES chosen uniformly
without replacement, ignoring patients. Every subset therefore has the same
size, the same optimizer steps per epoch and the same loss denominator, so a
size-only predictor (-N_j) is constant and cannot rank subsets.
Masks are indexed by POSITION in ctx['train'] (the frozen raw-index order).
Selection uses only the seed; it never looks at outcomes.
"""
import math
import numpy as np


def image_design(n_train, m, retain, seed):
    if not 0 < retain < 1 or m < 3:
        raise ValueError('Need 0 < retain < 1 and m >= 3')
    keep = int(math.floor(n_train * retain))
    if not 0 < keep < n_train:
        raise ValueError('Invalid retained image count')
    rng = np.random.default_rng(seed)
    masks, seen = [], set()
    while len(masks) < m:
        idx = np.sort(rng.choice(n_train, keep, replace=False))
        key = idx.tobytes()
        if key in seen:
            continue
        seen.add(key)
        mask = np.zeros(n_train, dtype=np.int8)
        mask[idx] = 1
        masks.append(mask.tolist())
    return dict(kind='image_random', m=m, retain=retain, seed=seed, retained_images=keep, masks=masks)


def features(masks, train_labels, train_groups):
    """Descriptive covariates per subset (never used for selection)."""
    M = np.asarray(masks, dtype=float)
    labels, groups = np.asarray(train_labels), np.asarray(train_groups)
    out = dict(n_images=M.sum(1))
    for c in sorted(set(labels.tolist())):
        out[f'class{c}_images'] = M[:, labels == c].sum(1)
    pids = sorted(set(groups.tolist()))
    present = np.stack([M[:, groups == p].sum(1) > 0 for p in pids], 1)
    whole = np.stack([M[:, groups == p].sum(1) == (groups == p).sum() for p in pids], 1)
    out['patients_with_any_retained_image'] = present.sum(1)
    out['patients_fully_removed'] = (~present).sum(1)
    out['patients_fully_retained'] = whole.sum(1)
    return out


def composition_matrix(masks, train_labels):
    M = np.asarray(masks, dtype=float)
    labels = np.asarray(train_labels)
    keys = sorted(set(labels.tolist()))
    return np.stack([M[:, labels == c].sum(1) for c in keys], 1), [f'class{c}_images' for c in keys]


def deleter_prediction(masks, tau):
    """Sum over DELETED images of the predicted deletion effect -tau_i (tau = upweight score)."""
    return -(1. - np.asarray(masks, dtype=float)) @ np.asarray(tau, dtype=float)


def matched_prediction(masks, tau):
    """First-order effect with the actual subset objective (mean CE over K retained
    images): delta_w_i = (N/K) m_i - 1. With K constant this is an affine map of the
    deletion sum, so its ranking is identical; reported only as an identity check."""
    M = np.asarray(masks, dtype=float)
    N, K = M.shape[1], M.sum(1)
    return ((N / K)[:, None] * M - 1.) @ np.asarray(tau, dtype=float)


def aggregation(train_groups, patient_ids):
    """A[p, i] = 1 when training image i belongs to patient p, so A @ tau_img = patient score."""
    groups = np.asarray(train_groups)
    return np.stack([(groups == p).astype(float) for p in patient_ids])
