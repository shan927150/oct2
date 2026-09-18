#!/usr/bin/env python3
"""B1-smoke: where along the ORIGINAL training prefix does a tiny deletion-weight
perturbation stop being linearly predictable?

Eq. 74 propagates u_t = dw_t/dalpha through the real Adam map.  Before that derivative
can be used for anything it has to be shown to BE the derivative, on prefixes short
enough that a paired finite difference is still a valid reference.  This script does
only that, on the original algorithm with nothing changed: same initialisation, split,
recorded batch order, dropout stream, coupled weight decay, lr and Adam state.

Four gates, in order.  Each one must pass before the next is attempted.

  G1  faithful loop == original trainer.
      A loop written here reproduces 05's epoch checkpoints bitwise (parameters, Adam
      moments, step counter, RNG states).  Without this every later number is about a
      different algorithm.

  G2  functional Adam == faithful loop.
      The hand-written functional step, which is what jvp differentiates, reproduces
      the faithful loop's state at every prefix.  torch.optim.Adam may use a foreach
      kernel whose reduction order differs from a single-tensor implementation, so this
      gate is a tight tolerance with the measured deviation reported, not bitwise.

  G3  zero direction.
      A "patient" whose rows are not in this shadow's training split must produce an
      exactly zero tangent.  This catches plumbing, not the Jacobian.

  G4  non-zero direction: jvp vs paired finite difference, per prefix, TWO precisions.
      For each prefix k and several exactly-representable alpha, compare the propagated
      tangent u_k against D_{alpha,k} = (theta_k(alpha) - theta_k(0)) / alpha, in
      parameter space AND in prediction space.  The truth for prefix k is the perturbed
      run truncated at k -- never the 50-epoch endpoint.

      The finite difference is taken twice: in float32, which is the precision the real
      pipeline trains in, and in float64 on the same trajectory.  Only the float64
      comparison gates, because only it can distinguish "the tangent is wrong" from
      "float32 cannot resolve the tangent".  The float32 result is reported as a finding
      about what a paired truth can measure, not as a pass or a failure.

  G5  why the tangent is where it is (reported, never gated).
      At t=1 Adam's bias correction makes the update exactly  -lr * g/(|g|+eps).  The
      derivative of that map, eps/(|g|+eps)^2, is ~1/(4 eps) where |g| ~ eps and ~eps/g^2
      where |g| >> eps, so d theta_1/d alpha concentrates on the coordinates whose
      gradient sits at Adam's epsilon floor -- numerically dead coordinates.  G5 measures
      that concentration instead of assuming it: the share of coordinates with |g| < 10
      eps, the quantiles of eps/(|g|+eps)^2, and how much of ||u||^2 the largest 0.1% of
      coordinates carry.  A tangent carried by a handful of epsilon-floor coordinates is
      a correct derivative of a map that is not doing what the derivation assumes.

Prefixes default to one step, one epoch and five epochs.  alpha values are powers of two
because 1-alpha is then exactly representable in float32: the decimal ladder already
loses 1.3% of the requested dose at 1e-6 and 19% at 1e-7, and 2^-25 rounds the weight to
1.0 so the perturbation disappears.  2^-24 is the exact floor.

Reading G4: at a given prefix, FD should approach the jvp as alpha falls.  The largest
alpha that still agrees is the linear range of that prefix.  If the agreement degrades
as the prefix grows, that is the amplification; where it degrades to uselessness is the
longest window a first-order trajectory score can serve.

Stage-1 only.  No attack training, no J10/J11, no claim about alpha=1.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402

spec = importlib.util.spec_from_file_location("b1_e0", HERE / "13_e0_residual_decomposition.py")
e0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e0
spec.loader.exec_module(e0)

LOG = logging.getLogger("b1_smoke")
SCHEMA = "pathway2_b1_smoke_v1"
FLOAT32_EXACT_FLOOR_EXPONENT = 24        # 1 - 2^-25 rounds to 1.0 in float32
ADAM_BETAS, ADAM_EPS = (0.9, 0.999), 1e-8


# ---------------------------------------------------------------------------
# dropout masks
# ---------------------------------------------------------------------------

class RecordedDropout(nn.Module):
    """Applies a pre-recorded, already-scaled dropout mask; constant w.r.t. alpha.

    The mask is a plain attribute, not a parameter or buffer, so functional_call
    leaves it alone and forward-mode AD treats it as a constant -- which is what it
    is: fixed_mask keeps the baseline batches and the baseline dropout draws, so the
    same mask applies to the perturbed run.
    """

    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        # nn.Dropout is the identity in eval mode, and this must be too: predictions are
        # read with model.eval() on the interface rows, whose batch size need not match
        # the training batch the mask was recorded for.
        if not self.training:
            return x
        if self.mask is None:
            raise RuntimeError("No dropout mask has been installed for this step")
        if self.mask.shape != x.shape:
            raise RuntimeError(f"Recorded mask {tuple(self.mask.shape)} does not match "
                               f"activation {tuple(x.shape)}")
        return x * self.mask


def dropout_module(model, kind=(nn.Dropout,)):
    """The single dropout site of SmallCNN, before or after it is swapped for a recorder."""
    found = [m for m in model.modules() if isinstance(m, kind)]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one dropout site in SmallCNN, found {len(found)}")
    return found[0]


def swap_in_recorder(model):
    """Replace the real Dropout with a RecordedDropout in place; returns the recorder."""
    module = dropout_module(model)
    recorded = RecordedDropout()
    parent = model.classifier
    parent[list(parent).index(module)] = recorded
    return recorded


def install_recorder(model):
    """Hook the real Dropout so a faithful run records the mask it actually drew.

    mask = (out != 0) / (1-p).  Where the input is 0 the mask is not recoverable, but
    dropout here follows a ReLU and PyTorch defines ReLU'(0) = 0, so the tangent at
    those positions is 0 for either mask value: the ambiguity cannot reach a derivative.
    The recorded mask is checked to reproduce the recorded output exactly.
    """
    module, store = dropout_module(model), []

    def hook(_module, inputs, output):
        x = inputs[0]
        mask = (output != 0).to(output.dtype) / (1.0 - _module.p)
        if not torch.equal(output, x * mask):
            raise RuntimeError("Recorded dropout mask does not reproduce the dropout output")
        store.append(mask.detach().clone())

    return module.register_forward_hook(hook), store


# ---------------------------------------------------------------------------
# faithful loop (G1) -- same operations and order as 05
# ---------------------------------------------------------------------------

def batch_plan(orders, batch_size, n_steps):
    plan = []
    for epoch, order in enumerate(orders):
        for start in range(0, len(order), batch_size):
            plan.append((epoch, np.asarray(order[start:start + batch_size], dtype=np.int64)))
            if n_steps is not None and len(plan) >= n_steps:
                return plan
    return plan


def faithful_run(pilot, X, y, orders, seed, cfg, alpha, excluded, n_steps, masks=None,
                 prefixes=(), record_masks=False):
    """05's loop, optionally truncated after n_steps, optionally replaying masks."""
    pilot.seed_everything(seed, cfg["deterministic"])
    model = pilot.build_model("cnn", X.shape[1], 128, int(np.max(y)) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    device = pilot.DEVICE
    excluded_tensor = torch.as_tensor(sorted(int(v) for v in excluded), dtype=torch.int64, device=device)
    handle, store = (install_recorder(model) if record_masks else (None, None))
    if masks is not None:
        recorded = swap_in_recorder(model)
    snapshots, wanted = {}, set(prefixes)
    model.train()
    plan = batch_plan(orders, cfg["batch_size"], n_steps)
    for step, (_epoch, batch_idx) in enumerate(plan):
        if masks is not None:
            recorded.mask = masks[step]
        xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        per_example = nn.functional.cross_entropy(model(xb), yb, reduction="none")
        weights = torch.ones_like(per_example)
        if excluded_tensor.numel():
            idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=device)
            removed = torch.isin(idx_t, excluded_tensor)
            weights = torch.where(removed, weights * (1.0 - alpha), weights)
        loss = (weights * per_example).sum() / per_example.numel()
        loss.backward()
        optimizer.step()
        if (step + 1) in wanted:
            snapshots[step + 1] = adam_state(model, optimizer)
    if handle is not None:
        handle.remove()
    return model, optimizer, snapshots, (store if record_masks else masks), len(plan)


def adam_state(model, optimizer):
    """Flattened (theta, m, v, step) in parameter order, detached on CPU float64."""
    params = [p for p in model.parameters() if p.requires_grad]
    state = optimizer.state
    m = torch.cat([state[p]["exp_avg"].detach().reshape(-1) for p in params]) if state else None
    v = torch.cat([state[p]["exp_avg_sq"].detach().reshape(-1) for p in params]) if state else None
    steps = {int(state[p]["step"]) if not torch.is_tensor(state[p]["step"])
             else int(state[p]["step"].item()) for p in params} if state else {0}
    if len(steps) != 1:
        raise RuntimeError("Adam step counters diverged across parameter groups")
    return {"theta": torch.cat([p.detach().reshape(-1) for p in params]).clone(),
            "m": None if m is None else m.clone(), "v": None if v is None else v.clone(),
            "step": steps.pop()}


# ---------------------------------------------------------------------------
# functional Adam (G2) -- the map jvp differentiates
# ---------------------------------------------------------------------------

def make_functional_step(model, X, y, cfg, device):
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [int(np.prod(s)) for s in shapes]

    def unflatten(flat):
        out, k = {}, 0
        for name, shape, n in zip(names, shapes, sizes):
            out[name] = flat[k:k + n].view(shape)
            k += n
        return out

    def loss_of(theta, xb, yb, removed, alpha):
        per = nn.functional.cross_entropy(
            functional_call(model, unflatten(theta), (xb,)), yb, reduction="none")
        weights = torch.where(removed, torch.ones_like(per) * (1.0 - alpha), torch.ones_like(per))
        return (weights * per).sum() / per.numel()

    def step(theta, m, v, t, alpha, xb, yb, removed):
        g = grad(loss_of)(theta, xb, yb, removed, alpha) + cfg["wd"] * theta
        b1, b2 = ADAM_BETAS
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        denom = v.sqrt() / math.sqrt(bc2) + ADAM_EPS
        return theta - (cfg["lr"] / bc1) * m / denom, m, v

    return step, unflatten, names, sizes


def functional_trajectory(model, X, y, cfg, device, theta0, plan, masks, excluded,
                          alpha, prefixes, tangent=False):
    """Primal, and optionally the forward tangent du/dalpha, over the given prefix plan."""
    step_fn, unflatten, _, _ = make_functional_step(model, X, y, cfg, device)
    recorded = dropout_module(model, (RecordedDropout,))
    theta = theta0.clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    ut = torch.zeros_like(theta)
    um = torch.zeros_like(theta)
    uv = torch.zeros_like(theta)
    excluded_tensor = torch.as_tensor(sorted(int(q) for q in excluded), dtype=torch.int64, device=device)
    out, wanted = {}, set(prefixes)
    for k, (_epoch, batch_idx) in enumerate(plan):
        # dtype follows theta so the same trajectory can be replayed in float64
        recorded.mask = masks[k].to(theta.dtype)
        xb = torch.as_tensor(X[batch_idx], dtype=theta.dtype, device=device)
        yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)
        idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=device)
        removed = (torch.isin(idx_t, excluded_tensor) if excluded_tensor.numel()
                   else torch.zeros_like(idx_t, dtype=torch.bool))
        t = k + 1
        if tangent:
            def f(th, mm, vv, al):
                return step_fn(th, mm, vv, t, al, xb, yb, removed)
            (theta, m, v), (ut, um, uv) = jvp(
                f, (theta, m, v, torch.as_tensor(alpha, dtype=theta.dtype, device=device)),
                (ut, um, uv, torch.ones((), dtype=theta.dtype, device=device)))
        else:
            theta, m, v = step_fn(theta, m, v, t, alpha, xb, yb, removed)
        if t in wanted:
            out[t] = {"theta": theta.detach().clone(), "m": m.detach().clone(),
                      "v": v.detach().clone(), "step": t,
                      "u_theta": ut.detach().clone() if tangent else None}
    return out


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------

def prediction_jvp(model, X, rows, theta, u_theta, device):
    """p(theta) on the interface rows, and its directional derivative along u_theta."""
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [int(np.prod(s)) for s in shapes]
    xb = torch.as_tensor(X[rows], dtype=torch.float32, device=device)

    def predict(flat):
        pieces, k = {}, 0
        for name, shape, n in zip(names, shapes, sizes):
            pieces[name] = flat[k:k + n].view(shape)
            k += n
        was = model.training
        model.eval()
        try:
            return torch.softmax(functional_call(model, pieces, (xb,)), dim=1)
        finally:
            model.train(was)

    if u_theta is None:
        return predict(theta), None
    return jvp(predict, (theta,), (u_theta,))


# ---------------------------------------------------------------------------

def batch_gradient(model, X, y, cfg, device, theta, plan, masks, step, excluded, alpha):
    """The raw gradient entering Adam at `step`, including coupled weight decay."""
    step_fn, unflatten, _, _ = make_functional_step(model, X, y, cfg, device)
    del step_fn
    recorded = dropout_module(model, (RecordedDropout,))
    recorded.mask = masks[step].to(theta.dtype)
    _epoch, batch_idx = plan[step]
    xb = torch.as_tensor(X[batch_idx], dtype=theta.dtype, device=device)
    yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)
    idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=device)
    ex = torch.as_tensor(sorted(int(q) for q in excluded), dtype=torch.int64, device=device)
    removed = (torch.isin(idx_t, ex) if ex.numel()
               else torch.zeros_like(idx_t, dtype=torch.bool))

    def loss_of(th, al):
        per = nn.functional.cross_entropy(
            functional_call(model, unflatten(th), (xb,)), yb, reduction="none")
        w = torch.where(removed, torch.ones_like(per) * (1.0 - al), torch.ones_like(per))
        return (w * per).sum() / per.numel()

    return grad(loss_of)(theta, torch.as_tensor(alpha, dtype=theta.dtype, device=device)) \
        + cfg["wd"] * theta


def epsilon_structure(g, u):
    """G5. Where Adam's epsilon puts the alpha-sensitivity, and where the tangent lives.

    At t=1 the Adam update is exactly -lr*g/(|g|+eps); its derivative w.r.t. g is
    eps/(|g|+eps)^2.  That is ~1/(4 eps) at |g|=eps and ~eps/g^2 for |g| >> eps, so the
    sensitivity is carried entirely by coordinates sitting at the epsilon floor.
    """
    g = g.detach().cpu().double().abs()
    sens = ADAM_EPS / (g + ADAM_EPS) ** 2
    out = {"n_coordinates": int(g.numel()),
           "share_grad_below_10eps": float((g < 10 * ADAM_EPS).double().mean()),
           "share_grad_below_eps": float((g < ADAM_EPS).double().mean()),
           "grad_abs_quantiles": {f"q{int(100 * q):02d}": float(torch.quantile(g, q))
                                  for q in (0.01, 0.10, 0.50, 0.90, 0.99)},
           "d_update_d_grad_median": float(sens.median()),
           "d_update_d_grad_max": float(sens.max())}
    if u is not None:
        sq = torch.sort(u.detach().cpu().double() ** 2, descending=True).values
        total = float(sq.sum())
        n = sq.numel()
        out.update({
            "tangent_norm": float(sq.sum().sqrt()),
            "share_of_tangent_sq_in_top_0p1pct": (float(sq[:max(1, n // 1000)].sum() / total)
                                                  if total > 0 else None),
            "share_of_tangent_sq_in_top_1pct": (float(sq[:max(1, n // 100)].sum() / total)
                                                if total > 0 else None),
            "largest_single_coordinate_share": (float(sq[0] / total) if total > 0 else None)})
    return out


def exact_alpha(exponent: int) -> float:
    alpha = 2.0 ** -exponent
    effective = float(1.0 - np.float32(1.0 - alpha))
    if effective != alpha:
        raise RuntimeError(f"alpha=2^-{exponent} is not exact in float32 (got {effective!r})")
    return alpha


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full_dir", required=True)
    p.add_argument("--dose_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--patients", type=int, nargs="+", default=[807, 2085, 1369],
                   help="1369 is the frozen low-endpoint-gradient sentinel: its eval gradient "
                        "at epoch 50 is ~1e-8, which says nothing about its influence during "
                        "early training, and these prefixes are early training")
    p.add_argument("--prefix_epochs", type=float, nargs="+", default=[0, 1, 5],
                   help="0 means a single optimizer step; other values are whole epochs")
    p.add_argument("--exponents", type=int, nargs="+", default=[10, 14, 18, 22],
                   help="finite-difference alphas as 2^-k (exactly representable in float32)")
    p.add_argument("--primal_tol", type=float, default=1e-6,
                   help="G2 gate: relative state deviation between functional and faithful Adam")
    p.add_argument("--derivative_tol", type=float, default=1e-2,
                   help="G4 reporting threshold for 'the finite difference has converged'")
    p.add_argument("--require_cuda", action="store_true")
    args = p.parse_args()
    bad = [k for k in args.exponents if k < 1 or k > FLOAT32_EXACT_FLOOR_EXPONENT]
    if bad:
        raise ValueError(f"exponents {bad} outside 1..{FLOAT32_EXACT_FLOOR_EXPONENT}")
    return args


def run(args, context, legacy, pilot, output):
    p, X, y = context["pargs"], context["X"], context["y"]
    seed, shadow = args.seed, p.affected_shadow
    device = pilot.DEVICE
    orders = legacy.verify_orders(context, seed)
    splits = legacy.read_json(context["full"] / "splits/fresh_patient_split.json")
    split = splits["shadow_models"][shadow]
    panel = {int(v["patient_id"]): v for v in context["patients"]}
    missing = set(args.patients) - panel.keys()
    if missing:
        raise RuntimeError(f"Patients {sorted(missing)} are not in the frozen panel")
    cfg = {"lr": p.shadow_lr, "batch_size": p.shadow_batch_size,
           "wd": context["config"]["oct_config"]["target_l2"], "deterministic": p.deterministic}
    per_epoch = math.ceil(len(orders[0]) / cfg["batch_size"])
    prefixes = sorted({1 if e == 0 else int(round(e)) * per_epoch for e in args.prefix_epochs})
    n_steps = max(prefixes)
    LOG.info("%d batches/epoch; prefixes (steps) = %s", per_epoch, prefixes)

    # ---- G1: faithful loop reproduces the original trainer
    t0 = time.time()
    model, optimizer, base_snap, masks, n_plan = faithful_run(
        pilot, X, y, orders, seed, cfg, 1.0, (), n_steps, prefixes=prefixes, record_masks=True)
    if len(masks) != n_plan:
        raise RuntimeError(f"Recorded {len(masks)} dropout masks for {n_plan} steps")
    g1 = {"prefix_steps": prefixes, "batches_per_epoch": per_epoch, "elapsed_seconds": time.time() - t0,
          "checked_against_original_epoch_checkpoints": []}
    frozen = sorted(set(p.save_epoch_checkpoints))
    for epoch in frozen:
        if epoch * per_epoch > n_steps:
            continue
        reference = legacy.load_payload(context["full"] / "checkpoints" /
                                        f"baseline_seed{seed}_epochs/epoch{epoch:03d}.pt", pilot)
        _, _, snap, _, _ = faithful_run(pilot, X, y, orders, seed, cfg, 1.0, (),
                                        epoch * per_epoch, prefixes=[epoch * per_epoch])
        got = snap[epoch * per_epoch]
        ref_theta = torch.cat([v.detach().cpu().reshape(-1) for v in reference["state_dict"].values()])
        exact = torch.equal(got["theta"].cpu(), ref_theta)
        g1["checked_against_original_epoch_checkpoints"].append(
            {"epoch": epoch, "parameters_bitwise_equal": bool(exact),
             "relative_difference": core.norm(got["theta"].cpu().double() - ref_theta.double())
                                    / core.norm(ref_theta.double())})
        if not exact:
            raise RuntimeError(f"G1 failed: faithful loop differs from 05 at epoch {epoch}")
    LOG.info("G1 passed: %s", g1["checked_against_original_epoch_checkpoints"])

    # ---- G2: functional Adam reproduces the faithful loop
    pilot.seed_everything(seed, cfg["deterministic"])
    theta0_model = pilot.build_model("cnn", X.shape[1], 128, int(np.max(y)) + 1)
    theta0 = torch.cat([q.detach().reshape(-1) for q in theta0_model.parameters()]).clone()
    swap_in_recorder(theta0_model)
    plan = batch_plan(orders, cfg["batch_size"], n_steps)
    functional_base = functional_trajectory(theta0_model, X, y, cfg, device, theta0, plan,
                                            masks, (), 1.0, prefixes, tangent=False)
    g2 = []
    for k in prefixes:
        a, b = functional_base[k], base_snap[k]
        row = {"prefix_steps": k, "step_counter_equal": a["step"] == b["step"]}
        for field in ("theta", "m", "v"):
            ref = b[field].cpu().double()
            row[f"{field}_relative_difference"] = core.norm(a[field].cpu().double() - ref) / core.norm(ref)
        worst = max(row[f"{field}_relative_difference"] for field in ("theta", "m", "v"))
        row["passed"] = bool(row["step_counter_equal"] and worst <= args.primal_tol)
        g2.append(row)
        if not row["passed"]:
            raise RuntimeError(f"G2 failed at prefix {k}: worst relative deviation {worst:.3e} "
                               f"> {args.primal_tol:.1e}")
    LOG.info("G2 passed: worst relative deviation %.3e",
             max(r[f"{f}_relative_difference"] for r in g2 for f in ("theta", "m", "v")))

    # ---- G3: a direction that touches no training row must give an exactly zero tangent
    outside = sorted(set(range(len(y))) - set(int(v) for v in split["train_idx"]))
    if not outside:
        raise RuntimeError("No row outside this shadow's training split for the zero-direction test")
    zero = functional_trajectory(theta0_model, X, y, cfg, device, theta0, plan, masks,
                                 outside[:8], 0.0, prefixes, tangent=True)
    g3 = [{"prefix_steps": k, "tangent_norm": core.norm(zero[k]["u_theta"]),
           "exactly_zero": bool(torch.count_nonzero(zero[k]["u_theta"]).item() == 0)}
          for k in prefixes]
    if not all(r["exactly_zero"] for r in g3):
        raise RuntimeError(f"G3 failed: absent-patient direction produced a non-zero tangent: {g3}")
    LOG.info("G3 passed: absent-patient tangent is exactly zero at every prefix")

    # ---- G4: jvp vs paired finite difference, per prefix, two precisions
    # The float64 trajectory is the same functional map replayed in double.  It is the
    # reference the jvp is gated against; the float32 faithful run is what the real
    # pipeline can actually produce, and is reported, not gated.
    theta0_64 = theta0.double()
    base_64 = functional_trajectory(theta0_model, X, y, cfg, device, theta0_64, plan,
                                    masks, (), 1.0, prefixes, tangent=False)
    rows_interface = np.concatenate([split["train_idx"], split["test_idx"]]).astype(np.int64)
    rows, g5 = [], []
    for pid in args.patients:
        idx = list(map(int, panel[pid]["raw_indices"]))
        tan = functional_trajectory(theta0_model, X, y, cfg, device, theta0, plan, masks,
                                    idx, 0.0, prefixes, tangent=True)
        # The gate compares a float64 difference against a float64 tangent. Comparing it
        # against the float32 tangent would measure the tangent's own precision instead,
        # which is reported separately as tangent_float32_vs_float64.
        tan64 = functional_trajectory(theta0_model, X, y, cfg, device, theta0_64, plan, masks,
                                      idx, 0.0, prefixes, tangent=True)
        # The first Adam step is the one whose denominator is exactly |g|+eps, so its
        # gradient is where the epsilon structure is directly readable.  Measured once.
        g_first = batch_gradient(theta0_model, X, y, cfg, device, theta0,
                                 plan, masks, 0, idx, 0.0)
        for k in prefixes:
            u = tan[k]["u_theta"]
            u64 = tan64[k]["u_theta"].cpu().double()
            tangent_precision = core.norm(u.cpu().double() - u64) / max(core.norm(u64), 1e-300)
            base_theta = functional_base[k]["theta"]
            p_base, p_tan = prediction_jvp(theta0_model, X, rows_interface, base_theta, u, device)
            g5.append({"seed": seed, "patient_id": pid, "prefix_steps": k,
                       "gradient_measured_at_step": 1,
                       "tangent_float32_vs_float64": tangent_precision,
                       **epsilon_structure(g_first, u64)})
            for exponent in sorted(args.exponents):
                alpha = exact_alpha(exponent)
                _, _, snap, _, _ = faithful_run(pilot, X, y, orders, seed, cfg, alpha, idx,
                                                k, masks=masks, prefixes=[k])
                fd = (snap[k]["theta"] - base_snap[k]["theta"]) / alpha
                pert64 = functional_trajectory(theta0_model, X, y, cfg, device, theta0_64, plan,
                                               masks, idx, alpha, [k], tangent=False)
                fd64 = (pert64[k]["theta"] - base_64[k]["theta"]) / alpha
                p_pert, _ = prediction_jvp(theta0_model, X, rows_interface, snap[k]["theta"], None, device)
                # predictions are (rows, classes); compare them as one flat vector
                fd_pred = ((p_pert - p_base) / alpha).reshape(-1).cpu().double()
                p_tan_flat = p_tan.reshape(-1).cpu().double()
                fd64 = fd64.cpu().double()
                rows.append({
                    "seed": seed, "patient_id": pid, "n_images": len(idx),
                    "prefix_steps": k, "prefix_epochs": k / per_epoch,
                    "exponent": exponent, "alpha": alpha,
                    "tangent_norm": core.norm(u), "fd_norm": core.norm(fd),
                    "param_relative_error": core.norm(fd.cpu().double() - u.cpu().double())
                                            / max(core.norm(u), 1e-300),
                    "param_cosine": core.cosine(fd.cpu().double(), u.cpu().double()),
                    # the gating comparison: same map, same alpha, float64 throughout
                    "fd64_norm": core.norm(fd64),
                    "param_relative_error_float64": core.norm(fd64 - u64) / max(core.norm(u64), 1e-300),
                    "param_cosine_float64": core.cosine(fd64, u64),
                    "pred_tangent_norm": core.norm(p_tan_flat), "pred_fd_norm": core.norm(fd_pred),
                    "pred_relative_error": core.norm(fd_pred - p_tan_flat)
                                           / max(core.norm(p_tan_flat), 1e-300),
                    "pred_cosine": core.cosine(fd_pred, p_tan_flat),
                    "displacement_norm": core.norm(snap[k]["theta"] - base_snap[k]["theta"]),
                })
                legacy.write_csv(output / "b1_rows.csv", rows)
                LOG.info("p%d prefix=%d(%.2f ep) alpha=2^-%d  f32_rel=%.3e  f64_rel=%.3e  pred_rel=%.3e",
                         pid, k, k / per_epoch, exponent, rows[-1]["param_relative_error"],
                         rows[-1]["param_relative_error_float64"], rows[-1]["pred_relative_error"])
    # G4 gates on float64 only: the jvp must be reproducible by a finite difference of
    # the same map at the same alpha when precision is not the limiting factor.
    g4 = []
    for pid in sorted({r["patient_id"] for r in rows}):
        for k in prefixes:
            sel = [r for r in rows if r["patient_id"] == pid and r["prefix_steps"] == k]
            best = min(r["param_relative_error_float64"] for r in sel)
            g4.append({"seed": seed, "patient_id": pid, "prefix_steps": k,
                       "best_float64_relative_error": best,
                       "best_float64_cosine": max(r["param_cosine_float64"] for r in sel),
                       "best_float32_relative_error": min(r["param_relative_error"] for r in sel),
                       "passed": bool(best <= args.derivative_tol)})
            if not g4[-1]["passed"]:
                raise RuntimeError(
                    f"G4 failed for patient {pid} at prefix {k}: best float64 finite-difference "
                    f"error {best:.3e} > {args.derivative_tol:.1e}. The tangent itself is wrong; "
                    f"this is not a precision effect.")
    LOG.info("G4 passed in float64: worst best-case error %.3e",
             max(r["best_float64_relative_error"] for r in g4))
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}, rows, prefixes, per_epoch


def summarise(gates, rows, tol):
    """Two separate questions, kept separate.

    (a) Is the propagated tangent the derivative?  Answered in float64, where precision
        is not the limiting factor.  This is what G4 gates on.
    (b) Can the real float32 pipeline produce a paired finite difference that sees it?
        Answered by the same comparison in float32.  Reported, never gated: a float32
        floor is a fact about what a paired truth can measure, not a wrong derivative.
    """
    out = {"gates": gates, "linear_range": {}, "float32_resolvability": {},
           "note": ("The linear range is where a paired finite difference reproduces the propagated "
                    "tangent at this prefix. It is a statement about that prefix only; it does not "
                    "license extrapolation to alpha=1 or to the 50-epoch endpoint. "
                    "float64 answers whether the tangent is correct; float32 answers whether the "
                    "pipeline's own precision can resolve it. A float32 floor with a clean float64 "
                    "result is a measurement limit, not a derivative error.")}
    for pid in sorted({r["patient_id"] for r in rows}):
        for k in sorted({r["prefix_steps"] for r in rows}):
            sel = sorted([r for r in rows if r["patient_id"] == pid and r["prefix_steps"] == k],
                         key=lambda r: -r["alpha"])
            if not sel:
                continue
            ok64 = [r for r in sel if r["param_relative_error_float64"] <= tol]
            ok32 = [r for r in sel if r["param_relative_error"] <= tol]
            key = f"p{pid}_prefix{k}"
            out["linear_range"][key] = {
                "prefix_epochs": sel[0]["prefix_epochs"],
                "largest_alpha_within_tol_float64": max((r["alpha"] for r in ok64), default=None),
                "best_param_relative_error_float64": min(r["param_relative_error_float64"] for r in sel),
                "best_param_cosine_float64": max(r["param_cosine_float64"] for r in sel),
                "tangent_norm": sel[0]["tangent_norm"],
                "converging_float64": bool(len(sel) >= 2 and
                                           sel[-1]["param_relative_error_float64"]
                                           < sel[0]["param_relative_error_float64"]),
            }
            best32 = min(sel, key=lambda r: r["param_relative_error"])
            out["float32_resolvability"][key] = {
                "largest_alpha_within_tol_float32": max((r["alpha"] for r in ok32), default=None),
                "best_param_relative_error_float32": best32["param_relative_error"],
                "best_param_cosine_float32": best32["param_cosine"],
                "alpha_at_best": best32["alpha"],
                "best_pred_relative_error": min(r["pred_relative_error"] for r in sel),
                "resolvable_at_tolerance": bool(ok32),
                "floor_is_numerical": bool(not ok32 and out["linear_range"][key]
                                           ["best_param_relative_error_float64"] <= tol),
            }
    sentinel = [r for r in rows if r["patient_id"] == 1369]
    if sentinel:
        by_prefix = {}
        for k in sorted({r["prefix_steps"] for r in sentinel}):
            sel = [r for r in sentinel if r["prefix_steps"] == k]
            by_prefix[k] = {"tangent_norm": sel[0]["tangent_norm"],
                            "best_param_relative_error_float64":
                                min(r["param_relative_error_float64"] for r in sel)}
        others = [r["tangent_norm"] for r in rows if r["patient_id"] != 1369]
        out["low_endpoint_gradient_sentinel"] = {
            "patient_id": 1369, "per_prefix": by_prefix,
            "tangent_norm_ratio_to_panel_median": (
                float(np.median([v["tangent_norm"] for v in by_prefix.values()])
                      / np.median(others)) if others else None),
            "note": ("1369's eval gradient at the frozen endpoint is ~1e-8. That is an endpoint "
                     "statement. These prefixes are early training, where a non-negligible "
                     "tangent would show the endpoint gradient does not bound early influence. "
                     "Reported, not used as evidence for or against any hypothesis here.")}
    g5 = gates.get("G5", [])
    if g5:
        out["epsilon_concentration"] = {
            "why": ("theta_1 = theta_0 - lr*g/(|g|+eps) exactly, so d/dalpha is weighted by "
                    "eps/(|g|+eps)^2, which is ~1/(4 eps) at |g|=eps and ~eps/g^2 for |g|>>eps. "
                    "If the tangent is concentrated on a handful of coordinates whose gradient "
                    "sits at the epsilon floor, it is a correct derivative of a map whose "
                    "alpha-sensitivity lives on numerically dead coordinates."),
            "worst_share_grad_below_10eps": max(r["share_grad_below_10eps"] for r in g5),
            "worst_top_0p1pct_share_of_tangent_sq": max(
                (r["share_of_tangent_sq_in_top_0p1pct"] for r in g5
                 if r.get("share_of_tangent_sq_in_top_0p1pct") is not None), default=None),
            "worst_largest_single_coordinate_share": max(
                (r["largest_single_coordinate_share"] for r in g5
                 if r.get("largest_single_coordinate_share") is not None), default=None),
            "worst_tangent_float32_vs_float64": max(r["tangent_float32_vs_float64"] for r in g5),
            "tangent_precision_note": (
                "The same amplification degrades the jvp itself at production precision: "
                "eps/(|g|+eps)^2 is ~1e8 where |g| ~ eps, so float32 noise in g reaches the "
                "tangent. This is the float32 tangent's distance from the float64 one."),
        }
    return out


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Delta invocation")
    output = Path(args.out_dir).resolve()
    for source in (Path(args.full_dir).resolve(), Path(args.dose_dir).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("B1 output overlaps a truth directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    legacy, score, pilot = e0.load_modules()
    tf32 = e0.set_tf32(False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {"schema": SCHEMA, "status": "running", "args": vars(args), "git_commit": commit,
                "python": platform.python_version(), "torch": torch.__version__,
                "device": str(pilot.DEVICE), "tf32": tf32,
                "scope": ("Stage-1 derivative validation on training prefixes of the original "
                          "algorithm. Not a fix, not an alpha=1 claim, no Stage-2."),
                "source_sha256": {str(q): legacy.sha(q) for q in (
                    HERE / "05_end_to_end_patient_loo_pilot.py", HERE / "11_stage1_diagnostics.py",
                    HERE / "13_e0_residual_decomposition.py", Path(__file__),
                    HERE.parents[1] / "src" / "models.py")}}
    legacy.write_json(output / "manifest.json", manifest)
    fingerprints = {}
    try:
        ctx_args = argparse.Namespace(full_dir=args.full_dir, dose_dir=args.dose_dir,
                                      mode="damping", data_dir=args.data_dir, seeds=[args.seed])
        context = legacy.load_context(ctx_args, pilot)
        fingerprints = {str(q): legacy.sha(q) for q in legacy.input_paths(context, ctx_args)}
        legacy.write_json(output / "input_sha256.json", fingerprints)
        gates, rows, prefixes, per_epoch = run(args, context, legacy, pilot, output)
        summary = summarise(gates, rows, args.derivative_tol)
        legacy.write_json(output / "b1_summary.json", summary)
        legacy.write_json(output / "b1_rows.json", {"rows": rows, "prefixes": prefixes,
                                                    "batches_per_epoch": per_epoch})
        after = {str(q): legacy.sha(q) for q in legacy.input_paths(context, ctx_args)}
        manifest.update(status="complete" if after == fingerprints else "complete_but_inputs_changed",
                        elapsed_seconds=time.time() - started, n_rows=len(rows),
                        input_files_unchanged=(after == fingerprints), summary=summary)
        legacy.write_json(output / "manifest.json", manifest)
        print(json.dumps({k: summary[k] for k in
                          ("linear_range", "float32_resolvability", "epsilon_concentration")
                          if k in summary}, indent=2))
    except Exception as exc:  # noqa: BLE001
        manifest.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
