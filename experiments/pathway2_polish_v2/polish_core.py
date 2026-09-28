"""Deterministic, dropout-off, full-batch polish of a fixed eval objective.

The objective is EXACTLY the one Eq. (52)-(56) use for g and H:

    F(theta) = (1/N) sum_i CE_i(theta) + (wd/2) ||theta||^2 ,   N = 2045 fixed.

The optional deletion objective is the fixed-mask downweight of one patient:

    F_{alpha,p}(theta) = F(theta) - (alpha/N) sum_{i in p} CE_i(theta) .

Nothing here writes a checkpoint, mutates the baseline, trains with dropout, or
shuffles. Dropout is OFF (model.eval()) but gradients are on. The damping gamma
used at Hessian-solve time is NOT part of this loss. L2 lives only in F.

This module is optimizer-side only. The controller recomputes both g and b
on FP64 copies of the endpoints. Intermediate snapshots use the model dtype
with FP64 accumulation; accumulation alone is not FP64 differentiation.
"""
from contextlib import contextmanager
import copy

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def deterministic_matmul():
    """No TF32 / no mixed precision during polish; restored on exit."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


@contextmanager
def preserved_rng(model):
    """Guarantee the polish leaves global RNG and module modes as it found them."""
    import random
    modes = [(m, m.training) for m in model.modules()]
    py, nrng, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(nrng)
        torch.set_rng_state(cpu)
        if gpu:
            torch.cuda.set_rng_state_all(gpu)
        for module, mode in modes:
            module.training = mode


def _patient_mask(idx, rows, device):
    return torch.as_tensor(np.isin(idx, rows), device=device)


class FullBatchObjective:
    """Chunked full-batch F (or F_{alpha,p}); denominator is always N.

    A single closure() call accumulates the gradient of the WHOLE training set
    into p.grad, then adds the L2 gradient once. Parameters never move between
    chunks, so this is a genuine full-batch step, not minibatch training.
    """

    def __init__(self, model, X, y, indices, wd, batch, *,
                 patient_rows=None, alpha=0.0, deleted_patient=None):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params or any(not p.requires_grad for p in model.parameters()):
            raise ValueError("All model parameters must require grad")
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
               for m in model.modules()):
            raise ValueError("BatchNorm is outside the frozen OCT CNN contract")
        self.X, self.y = X, y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.N = len(self.indices)
        if self.N < 1 or batch < 1 or len(np.unique(self.indices)) != self.N or wd < 0:
            raise ValueError("Invalid fixed training set, batch or L2")
        self.wd, self.batch = wd, batch
        self.alpha = float(alpha)
        self.deleted_rows = None
        if not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if self.alpha != 0.0:
            if deleted_patient is None or patient_rows is None:
                raise ValueError("Deletion needs a patient and its rows")
            rows = np.asarray(patient_rows[deleted_patient], dtype=np.int64)
            if len(rows) == 0 or not np.isin(rows, self.indices).all():
                raise ValueError("Invalid deleted-patient rows")
            self.deleted_rows = rows
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.closure_calls = 0

    def objective_value(self):
        """Read-only F value; forward precision follows the model dtype."""
        was_training = self.model.training
        with torch.no_grad():
            self.model.eval()
            ce = 0.0
            for s in range(0, self.N, self.batch):
                idx = self.indices[s:s + self.batch]
                xb = torch.as_tensor(self.X[idx], device=self.device, dtype=self.dtype)
                yb = torch.as_tensor(self.y[idx], device=self.device, dtype=torch.long)
                losses = F.cross_entropy(self.model(xb), yb, reduction="none")
                chunk = losses.sum() / self.N
                if self.alpha != 0.0:
                    m = _patient_mask(idx, self.deleted_rows, self.device)
                    chunk = chunk - self.alpha * losses[m].sum() / self.N
                ce += float(chunk.double())
            theta = torch.cat([p.detach().reshape(-1) for p in self.params]).double()
            l2 = float(self.wd * torch.dot(theta, theta) / 2)
        self.model.train(was_training)
        return ce + l2

    def make_closure(self, optimizer):
        def closure():
            self.closure_calls += 1
            optimizer.zero_grad(set_to_none=False)
            self.model.eval()  # dropout OFF, grad ON
            total = 0.0
            with torch.enable_grad():
                for s in range(0, self.N, self.batch):
                    idx = self.indices[s:s + self.batch]
                    xb = torch.as_tensor(self.X[idx], device=self.device, dtype=self.dtype)
                    yb = torch.as_tensor(self.y[idx], device=self.device, dtype=torch.long)
                    losses = F.cross_entropy(self.model(xb), yb, reduction="none")
                    chunk = losses.sum() / self.N
                    if self.alpha != 0.0:
                        m = _patient_mask(idx, self.deleted_rows, self.device)
                        chunk = chunk - self.alpha * losses[m].sum() / self.N
                    chunk.backward()          # accumulates data grad into p.grad
                    total += float(chunk.detach())
            # add the L2 term's value and gradient exactly once
            l2 = 0.0
            with torch.no_grad():
                for p in self.params:
                    p.grad.add_(self.wd * p.detach())
                    l2 += float(self.wd * (p.detach() ** 2).sum() / 2)
            return torch.as_tensor(total + l2, dtype=torch.float64)
        return closure


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def grad_norm_fp64_model(model, X, y, indices, wd, batch):
    """Authoritative ||g|| at a FP64 copy of the model (dropout off)."""
    m64 = copy.deepcopy(model).double()
    for p in m64.parameters():
        p.requires_grad_(True)
    m64.eval()
    params = [p for p in m64.parameters()]
    idxs = np.asarray(indices, dtype=np.int64)
    N = len(idxs)
    for p in params:
        p.grad = None
    total = 0.0
    for s in range(0, N, batch):
        idx = idxs[s:s + batch]
        xb = torch.as_tensor(X[idx], device=params[0].device, dtype=torch.float64)
        yb = torch.as_tensor(y[idx], device=params[0].device, dtype=torch.long)
        loss = F.cross_entropy(m64(xb), yb, reduction="sum") / N
        loss.backward()
        total += float(loss.detach())
    g = []
    for p in params:
        gp = (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        gp = gp + wd * p.detach().reshape(-1)
        g.append(gp)
    g = torch.cat(g)
    return float(torch.linalg.vector_norm(g)), total


def polish(model, X, y, indices, wd, batch, *,
           patient_rows=None, alpha=0.0, deleted_patient=None,
           max_outer=100, eval_budget=500, history_size=20,
           grad_target=0.0, diagnostic_every=10, snapshot_fn=None,
           patients_for_snapshot=None, stall_tol=0.0):
    """Bound optimizer closures, roll back unfinished line searches, audit accepted points.

    The closure budget excludes read-only diagnostic snapshots. Those are
    counted separately. No convergence claim is made by this routine.
    """
    if min(max_outer, eval_budget, history_size, diagnostic_every) < 1:
        raise ValueError('Budgets must be positive')
    obj = FullBatchObjective(model, X, y, indices, wd, batch,
                             patient_rows=patient_rows, alpha=alpha,
                             deleted_patient=deleted_patient)
    theta0 = flat_params(obj.params).clone()
    opt = torch.optim.LBFGS(obj.params, lr=1.0, max_iter=1,
                            history_size=history_size,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)
    raw_closure = obj.make_closure(opt)

    class BudgetEnd(Exception):
        pass

    def closure():
        if obj.closure_calls >= eval_budget:
            raise BudgetEnd()
        val = raw_closure()
        if not torch.isfinite(val) or any(not torch.isfinite(p.grad).all() for p in obj.params):
            raise FloatingPointError('Non-finite objective or gradient')
        return val

    def fp32_grad_norm():
        with torch.no_grad():
            return float(torch.linalg.vector_norm(
                torch.cat([p.grad.reshape(-1) for p in obj.params])))

    def diagnostic(outer, value):
        row = dict(outer=outer, closure_calls=obj.closure_calls,
                   objective=float(value),
                   fp32_grad_norm=fp32_grad_norm(),
                   param_shift=float(torch.linalg.vector_norm(
                       flat_params(obj.params) - theta0)),
                   param_norm=float(torch.linalg.vector_norm(flat_params(obj.params))))
        if snapshot_fn is not None:
            info, vectors = snapshot_fn(model, dropout=False)
            g = vectors[0]
            row["snapshot_g_norm"] = float(np.linalg.norm(g))
            if patients_for_snapshot:
                for j, pid in enumerate(patients_for_snapshot, 1):
                    row[f"b_norm_{pid}"] = float(np.linalg.norm(vectors[j]))
        return row

    trajectory = []
    stop = "iteration_budget_exhausted"
    accepted_outer = 0
    snapshot_calls = 0
    with deterministic_matmul(), preserved_rng(model):
        value = closure()
        trajectory.append(diagnostic(0, value))
        snapshot_calls += int(snapshot_fn is not None)
        for outer in range(1, max_outer + 1):
            if obj.closure_calls >= eval_budget:
                stop = 'eval_budget_exhausted'
                break
            old_params = [p.detach().clone() for p in obj.params]
            old_grads = [p.grad.detach().clone() for p in obj.params]
            old_value = value
            try:
                opt.step(closure)
                # LBFGS can leave .grad at a line-search trial; measure the
                # actual accepted endpoint before using a gradient in a report.
                value = closure()
                if float(value) > float(old_value) + 1e-7 * max(1., abs(float(old_value))):
                    raise FloatingPointError('Accepted objective increased')
            except (BudgetEnd, FloatingPointError) as exc:
                with torch.no_grad():
                    for p, oldp, oldg in zip(obj.params, old_params, old_grads):
                        p.copy_(oldp)
                        p.grad = oldg
                value = old_value
                stop = ('eval_budget_exhausted_rolled_back' if isinstance(exc, BudgetEnd)
                        else 'nonfinite_or_increase_rolled_back')
                break
            accepted_outer = outer
            moved = float(torch.linalg.vector_norm(torch.cat([
                (p.detach() - oldp).reshape(-1) for p, oldp in zip(obj.params, old_params)])))
            if outer % diagnostic_every == 0 or outer == max_outer:
                trajectory.append(diagnostic(outer, value))
                snapshot_calls += int(snapshot_fn is not None)
                print(f'outer={outer} closures={obj.closure_calls} objective={float(value):.8g}', flush=True)
            gnorm = fp32_grad_norm()
            if not np.isfinite(gnorm):
                stop = "non_finite_gradient"
                break
            if grad_target > 0 and gnorm <= grad_target:
                stop = "fp32_gradient_threshold_reached"
                break
            if moved <= stall_tol:
                stop = "line_search_stalled"
                break
            if obj.closure_calls >= eval_budget:
                stop = "eval_budget_exhausted"
                break
        final = diagnostic(accepted_outer, value)
        snapshot_calls += int(snapshot_fn is not None)
        if trajectory[-1]['outer'] != accepted_outer:
            trajectory.append(final)
    return dict(stop_reason=stop, closure_calls=obj.closure_calls,
                accepted_iterations=accepted_outer, diagnostic_snapshot_calls=snapshot_calls,
                theta_shift=float(torch.linalg.vector_norm(flat_params(obj.params) - theta0)),
                theta0_norm=float(torch.linalg.vector_norm(theta0)),
                trajectory=trajectory, final=final)
