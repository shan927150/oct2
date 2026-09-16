"""Numerical helpers for the independent Stage 1 audit.

No attack training, source-artifact writes, or damping selection occurs here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()))


def cosine(x, y):
    nx, ny = norm(x), norm(y)
    if nx == 0 or ny == 0:
        return None
    return float(torch.clamp(torch.dot(x.double(), y.double()) / (nx * ny), -1, 1))


def geometry(pred, actual):
    npred, nactual = norm(pred), norm(actual)
    return {
        "pred_norm": npred, "true_norm": nactual,
        "cosine": cosine(pred, actual),
        "norm_ratio": npred / nactual if nactual else None,
        "relative_error": norm(pred - actual) / nactual if nactual else None,
    }


def snapshot_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


@contextmanager
def preserve_rng_and_modes(model):
    state = snapshot_rng()
    modes = [(m, m.training) for m in model.modules()]
    try:
        yield
    finally:
        restore_rng(state)
        for module, mode in modes:
            module.training = mode


def exact_tree(a, b):
    """Compare optimizer/RNG/model payloads without device-dependent equality."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def objective_snapshot(model, X, y, indices, batch, wd, with_grad, dropout=False):
    """Full fixed-dataset CE and CE+L2; autograd.grad never changes p.grad.

    Caller owns RNG preservation for dropout snapshots. Report norm of the
    dataset gradient, not the mean of minibatch gradient norms.
    """
    model.train(dropout)
    params = list(model.parameters())
    device, dtype = params[0].device, params[0].dtype
    grads = torch.zeros(sum(p.numel() for p in params), device=device, dtype=dtype)
    ce, n = 0.0, len(indices)
    with torch.set_grad_enabled(with_grad):
        for start in range(0, n, batch):
            idx = indices[start:start + batch]
            xb = torch.as_tensor(X[idx], device=device, dtype=dtype)
            yb = torch.as_tensor(y[idx], device=device, dtype=torch.long)
            loss = F.cross_entropy(model(xb), yb, reduction="sum") / n
            ce += float(loss.detach())
            if with_grad:
                g = torch.autograd.grad(loss, params)
                grads += torch.cat([v.reshape(-1) for v in g])
    theta = flat(model)
    penalty = wd * float(torch.dot(theta.double(), theta.double())) / 2
    if with_grad:
        grads += wd * theta
    return {"ce": ce, "l2_penalty": penalty, "objective": ce + penalty,
            "grad_norm": norm(grads) if with_grad else None}, grads


def observe_checkpoint(model, X, y, indices, batch, wd, with_grad,
                       mc_reps=0, mc_seed=910000):
    """Evaluation preserves training RNG, module modes and accumulated grads."""
    with preserve_rng_and_modes(model):
        result, _ = objective_snapshot(model, X, y, indices, batch, wd, with_grad)
        out = {"eval_" + k: v for k, v in result.items()}
        if mc_reps:
            grad_sum, square_norm_sum, losses = None, 0.0, []
            for rep in range(mc_reps):
                torch.manual_seed(mc_seed + rep)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(mc_seed + rep)
                row, grad = objective_snapshot(model, X, y, indices, batch, wd, True, dropout=True)
                losses.append(row["ce"])
                grad_sum = grad.double() if grad_sum is None else grad_sum + grad.double()
                square_norm_sum += norm(grad) ** 2
            mean_grad = grad_sum / mc_reps
            out.update(dropout_mc_reps=mc_reps,
                       dropout_mc_ce_mean=float(np.mean(losses)),
                       dropout_mc_ce_se=float(np.std(losses, ddof=1) / math.sqrt(mc_reps)) if mc_reps > 1 else None,
                       dropout_mc_objective=float(np.mean(losses)) + result["l2_penalty"],
                       dropout_mc_mean_gradient_norm=norm(mean_grad),
                       dropout_mc_gradient_rms_spread=math.sqrt(max(0., square_norm_sum / mc_reps - norm(mean_grad) ** 2)))
        return out


def lanczos_probe(matvec, size, steps, probe_seed, device, dtype):
    """Full reorthogonalization, Ritz values and explicit endpoint residuals.

    A positive minimum Ritz value is NOT a proof of positive definiteness.
    The stored tridiagonal describes this Krylov projection, not a full ESD.
    """
    generator = torch.Generator(device="cpu").manual_seed(probe_seed)
    q = torch.randn(size, generator=generator, dtype=dtype).to(device)
    q /= q.norm()
    basis, diagonal, off_diagonal = [], [], []
    beta_prev, qprev = 0., torch.zeros_like(q)
    for _ in range(min(steps, size)):
        basis.append(q)
        z = matvec(q)
        alpha = float(torch.dot(q, z))
        z = z - alpha * q - beta_prev * qprev
        for _pass in range(2):
            for v in basis:
                z = z - torch.dot(v, z) * v
        diagonal.append(alpha)
        beta = norm(z)
        if beta < 1e-10 or len(diagonal) == min(steps, size):
            break
        off_diagonal.append(beta)
        qprev, q = q, z / beta
        beta_prev = beta
    T = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    if off_diagonal:
        off = torch.tensor(off_diagonal, dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    Q = torch.stack(basis, dim=1)
    endpoints = {}
    for label, index in [("min", 0), ("max", -1)]:
        vector = Q @ eigenvectors[:, index].to(device=device, dtype=dtype)
        vector /= vector.norm()
        value = float(eigenvalues[index])
        residual = norm(matvec(vector) - value * vector)
        endpoints[label] = {"value": value, "residual_abs": residual,
                            "residual_scaled": residual / max(1., abs(value))}
    half = max(1, len(diagonal) // 2)
    ev_half = torch.linalg.eigvalsh(T[:half, :half])
    return {"probe_seed": probe_seed, "steps": len(diagonal), "endpoints": endpoints,
            "half_steps": half, "half_min": float(ev_half.min()), "half_max": float(ev_half.max()),
            "ritz_values": eigenvalues.tolist(), "tridiagonal_diagonal": diagonal,
            "tridiagonal_off_diagonal": off_diagonal,
            "note": "Finite Krylov diagnostic, not an SPD certificate or full spectral distribution."}


def spectrum_screen(probes, gamma, ritz_tol):
    min_shift = min(p["endpoints"]["min"]["value"] for p in probes) + gamma
    positive = all(p["endpoints"]["min"]["value"] + gamma >
                   p["endpoints"]["min"]["residual_abs"] for p in probes)
    residual_ok = all(p["endpoints"][label]["residual_scaled"] <= ritz_tol
                      for p in probes for label in ("min", "max"))
    return {"min_damped_ritz_estimate": min_shift, "positive_ritz_screen": positive,
            "ritz_residual_screen": residual_ok, "passed": positive and residual_ok,
            "not_spd_certificate": True}


def solver_qualified(cg, screen, fail_tol):
    residual = cg.get("final_rel_residual")
    return bool(screen["passed"] and not cg.get("nonpositive_curvature")
                and not cg.get("nonfinite") and residual is not None
                and math.isfinite(residual) and residual <= fail_tol)


def replay_baseline(pilot, X, y, orders, seed, lr, batch, wd, observe,
                    check_epoch=None, n_hidden=128):
    """Exact v4.1 baseline arithmetic plus a RNG-preserving epoch observer.

    Kept separate from 05 so an in-flight truth/score job never imports changed
    training code. Caller validates against original checkpoints and RNG.
    """
    pilot.seed_everything(seed, deterministic=True)
    model = pilot.build_model("cnn", X.shape[1], n_hidden, int(np.max(y)) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    initial = {"epoch": 0, "online_train_ce": None, "epoch_update_norm": None,
               "relative_epoch_update": None}
    with preserve_rng_and_modes(model):
        observe(model, initial)
    for epoch, order in enumerate(orders, start=1):
        before = flat(model).clone()
        model.train()
        total_loss, seen = 0., 0
        for batch_idx in pilot._iter_batches(order, batch):
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=pilot.DEVICE)
            yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=pilot.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            per_example = F.cross_entropy(model(xb), yb, reduction="none")
            weights = torch.ones_like(per_example)
            loss = (weights * per_example).sum() / per_example.numel()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_idx)
            seen += len(batch_idx)
        if check_epoch is not None:
            check_epoch(epoch, model, optimizer)
        update_norm = norm(flat(model) - before)
        row = {"epoch": epoch, "online_train_ce": total_loss / seen,
               "epoch_update_norm": update_norm,
               "relative_epoch_update": update_norm / norm(before) if norm(before) else None}
        with preserve_rng_and_modes(model):
            observe(model, row)
    return model, optimizer, rng_fingerprint()
