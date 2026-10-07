"""Patient-subset distributions. Selection depends only on patient class and
image counts, never on any outcome. Every design is regenerated from its seed
inside GPU jobs and must equal the frozen designs.json written by the smoke.

pilot8   the frozen LDS v1 masks (8 x 74/148, uniform, subset_seed=20260929)
eq32     class-stratified retained patient counts AND total/per-class image
         counts inside a prespecified tolerance (rejection sampling, so the
         design is uniform over the constrained set)
smalldel delete a fixed small number of patients uniformly (closer to the local
         regime of IF; optional, needs mentor sign-off before submission)
"""
import math
import numpy as np


def class_counts(train_patients, keep_total):
    """Largest-remainder allocation of retained patients to classes."""
    classes = sorted(set(p['oct_class'] for p in train_patients))
    n = {c: sum(p['oct_class'] == c for p in train_patients) for c in classes}
    total = len(train_patients)
    raw = {c: n[c] * keep_total / total for c in classes}
    k = {c: int(math.floor(raw[c])) for c in classes}
    order = sorted(classes, key=lambda c: (raw[c] - k[c], c), reverse=True)
    for c in order[:keep_total - sum(k.values())]:
        k[c] += 1
    if any(not 0 < k[c] < n[c] for c in classes):
        raise ValueError('Stratified allocation leaves a class empty or complete')
    return classes, n, k


def equal_size_stratified(train_patients, m, retain, seed, total_tol_frac=.02,
                          class_tol_frac=.10, batch=20000, max_draws=5_000_000,
                          diagnostic_pool=4000):
    images = np.asarray([len(p['rows']) for p in train_patients], dtype=np.int64)
    cls = np.asarray([p['oct_class'] for p in train_patients])
    keep_total = round(len(train_patients) * retain)
    classes, n, k = class_counts(train_patients, keep_total)
    members = {c: np.flatnonzero(cls == c) for c in classes}
    target_c = {c: images[members[c]].sum() * k[c] / n[c] for c in classes}
    target = sum(target_c.values())
    tol = max(1., total_tol_frac * target)
    tol_c = {c: max(1., class_tol_frac * target_c[c]) for c in classes}
    rng = np.random.default_rng(seed)

    def draw(rng, size):
        masks = np.zeros((size, len(train_patients)), dtype=np.int8)
        for c in classes:
            keys = rng.random((size, n[c]))
            pick = np.argpartition(keys, k[c] - 1, axis=1)[:, :k[c]]
            masks[np.arange(size)[:, None], members[c][pick]] = 1
        ok = np.abs(masks @ images - target) <= tol
        for c in classes:
            ok &= np.abs(masks[:, members[c]] @ images[members[c]] - target_c[c]) <= tol_c[c]
        return masks[ok]

    chosen, seen, draws = [], set(), 0
    while len(chosen) < m:
        if draws >= max_draws:
            raise ValueError(f'eq design infeasible: {len(chosen)}/{m} after {draws} draws. '
                             'Loosen --eq-total-tol/--eq-class-tol at prepare (never after seeing outcomes).')
        for mask in draw(rng, batch):
            key = mask.tobytes()
            if key not in seen and len(chosen) < m:
                seen.add(key)
                chosen.append(mask.tolist())
        draws += batch
    # Inclusion-probability diagnostic from an independent stream (not used for selection).
    diag_rng = np.random.default_rng(seed + 1)
    pool, pool_draws = [], 0
    while sum(len(x) for x in pool) < diagnostic_pool and pool_draws < max_draws:
        pool.append(draw(diag_rng, batch))
        pool_draws += batch
    pool = np.concatenate(pool) if pool else np.zeros((0, len(train_patients)))
    acceptance = len(pool) / pool_draws if pool_draws else None
    inclusion = pool.mean(0) if len(pool) else np.full(len(train_patients), np.nan)
    return dict(masks=chosen, kind='eq', seed=seed, retain=retain,
                stratified_counts={str(c): int(k[c]) for c in classes},
                target_total_images=float(target), total_tolerance=float(tol),
                target_class_images={str(c): float(target_c[c]) for c in classes},
                class_tolerance={str(c): float(tol_c[c]) for c in classes},
                diagnostics=dict(acceptance_rate=acceptance, selection_draws=draws,
                                 inclusion_probability_min=float(np.min(inclusion)),
                                 inclusion_probability_max=float(np.max(inclusion)),
                                 inclusion_probability=inclusion.tolist(),
                                 note='Constrained-uniform design: heavy patients can have inclusion != retain.'))


def small_deletion(train_patients, m, n_delete, seed):
    n = len(train_patients)
    if not 0 < n_delete < n or m > math.comb(n, n_delete):
        raise ValueError('Invalid small-deletion design')
    rng = np.random.default_rng(seed)
    chosen, seen = [], set()
    while len(chosen) < m:
        drop = tuple(sorted(rng.choice(n, n_delete, replace=False).tolist()))
        if drop in seen:
            continue
        seen.add(drop)
        mask = np.ones(n, dtype=np.int8)
        mask[list(drop)] = 0
        chosen.append(mask.tolist())
    return dict(masks=chosen, kind='smalldel', seed=seed, n_delete=n_delete)


def features(masks, train_patients):
    """Prespecified naive covariates of each subset: images and class composition."""
    masks = np.asarray(masks, dtype=float)
    images = np.asarray([len(p['rows']) for p in train_patients], dtype=float)
    cls = np.asarray([p['oct_class'] for p in train_patients])
    out = dict(n_images=masks @ images, n_patients=masks.sum(1))
    for c in sorted(set(cls.tolist())):
        out[f'class{c}_patients'] = masks[:, cls == c].sum(1)
        out[f'class{c}_images'] = masks[:, cls == c] @ images[cls == c]
    return out


def composition_matrix(masks, train_patients):
    f = features(masks, train_patients)
    keys = sorted(k for k in f if k.startswith('class'))
    return np.stack([f[k] for k in keys], 1), keys


def build_all(base, spec):
    """base = LDS v1 make_design(...) output; spec = frozen config['designs']."""
    out = {}
    for name, s in spec.items():
        if s['kind'] == 'pilot':
            d = dict(masks=base['masks'], kind='pilot', seed=base['subset_seed'], retain=.5)
        elif s['kind'] == 'eq':
            d = equal_size_stratified(base['train_patients'], s['m'], s['retain'], s['seed'],
                                      s['total_tol_frac'], s['class_tol_frac'])
        elif s['kind'] == 'smalldel':
            d = small_deletion(base['train_patients'], s['m'], s['n_delete'], s['seed'])
        else:
            raise ValueError('Unknown design kind')
        if len(d['masks']) != s['m']:
            raise ValueError('Design size mismatch: ' + name)
        f = features(d['masks'], base['train_patients'])
        d['n_images'] = f['n_images'].tolist()
        d['n_images_range'] = [float(f['n_images'].min()), float(f['n_images'].max())]
        out[name] = d
    return out
