"""Read-only full-data and patient gradients with shared dropout masks."""
from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def preserved(model, seed=None):
    modes = [(m, m.training) for m in model.modules()]
    py, nrng = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        if seed is not None:
            torch.manual_seed(seed)
            if gpu:
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(nrng)
        torch.set_rng_state(cpu)
        if gpu:
            torch.cuda.set_rng_state_all(gpu)
        for module, mode in modes:
            module.training = mode


def snapshot(model, X, y, indices, patient_rows, batch, wd,
             dropout=False, seed=None, check_identity=False):
    """No optimizer, backward(), p.grad writes, shuffling or model saves.

    A patient's CE is divided by ORIGINAL N, never by patient size or N-n_p.
    g and all b_p share the logits and masks of each batch. The optional direct
    weighted-loss backprop checks g-alpha*b against the actual fixed-mask loss.
    """
    params = list(model.parameters())
    if not params or not all(p.requires_grad for p in params):
        raise ValueError('All model parameters must be differentiable')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('BatchNorm is outside the frozen OCT CNN contract')
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if n < 1 or batch < 1 or len(np.unique(indices)) != n or wd < 0:
        raise ValueError('Invalid fixed training set, batch or L2')
    patients = list(patient_rows)
    for pid, rows in patient_rows.items():
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0 or len(np.unique(rows)) != len(rows) or not np.isin(rows, indices).all():
            raise ValueError(f'Invalid patient rows: {pid}')
    device, dtype = params[0].device, params[0].dtype
    size = sum(p.numel() for p in params)
    result = torch.zeros((1 + len(patients), size), dtype=torch.float64, device=device)
    direct_keys = [(j, a) for j in range(1, len(patients) + 1) for a in (.1, 1.)]
    direct = {key: torch.zeros(size, dtype=torch.float64, device=device)
              for key in direct_keys} if check_identity else {}
    ce = 0.
    patient_ce = {p: 0. for p in patients}
    with preserved(model, seed), torch.enable_grad():
        model.train(dropout)
        for start in range(0, n, batch):
            idx = indices[start:start + batch]
            xb = torch.as_tensor(X[idx], device=device, dtype=dtype)
            yb = torch.as_tensor(y[idx], device=device, dtype=torch.long)
            losses = F.cross_entropy(model(xb), yb, reduction='none')
            ce += float((losses.sum() / n).detach())
            masks = {j: torch.as_tensor(np.isin(idx, patient_rows[p]), device=device)
                     for j, p in enumerate(patients, 1)}
            jobs = [(0, losses.sum() / n)]
            for j, p in enumerate(patients, 1):
                if bool(masks[j].any()):
                    loss = losses[masks[j]].sum() / n
                    patient_ce[p] += float(loss.detach())
                    jobs.append((j, loss))
            if check_identity:
                for j, alpha in direct_keys:
                    weights = 1. - alpha * masks[j].to(dtype)
                    jobs.append(((j, alpha), (losses * weights).sum() / n))
            for k, (key, loss) in enumerate(jobs):
                grad = torch.autograd.grad(loss, params, retain_graph=k < len(jobs) - 1)
                flat_grad = torch.cat([g.detach().reshape(-1) for g in grad]).double()
                if isinstance(key, tuple):
                    direct[key] += flat_grad
                else:
                    result[key] += flat_grad
            del losses, jobs, grad, flat_grad, xb, yb
        theta = torch.cat([p.detach().reshape(-1) for p in params]).double()
        result[0] += wd * theta
        penalty = wd * float(torch.dot(theta, theta)) / 2
        checks = []
        for (j, alpha), value in direct.items():
            value += wd * theta
            expected = result[0] - alpha * result[j]
            error = float(torch.linalg.vector_norm(expected - value))
            scale = max(float(torch.linalg.vector_norm(value)),
                        float(torch.linalg.vector_norm(result[0])),
                        alpha * float(torch.linalg.vector_norm(result[j])))
            passed = error <= 2e-5 * scale + 1e-8
            checks.append(dict(patient_id=patients[j-1], alpha=alpha,
                               error_norm=error, scale=scale, passed=passed))
            if not passed:
                raise RuntimeError(f'Fixed-mask gradient identity failed: {checks[-1]}')
    if not bool(torch.isfinite(result).all()) or not np.isfinite(ce + penalty):
        raise RuntimeError('Non-finite gradient or objective')
    return dict(ce=ce, l2_penalty=penalty, objective=ce + penalty,
                patient_ce_contribution=patient_ce, identity_checks=checks), result.cpu().numpy()
