"""Paired Adam prefix diagnostics. No changes to the frozen 05 training code.

Forward tangents implement the chain rule dual to pathway-2 (73)-(74), with
alpha a DOWNweight: loss = sum((1-alpha*patient_mask)*CE)/original_batch_size.
Step count, data order and dropout draws have zero tangent. Coupled weight decay
is included in both the gradient and its tangent.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp

BETAS, EPS = (0.9, 0.999), 1e-8


def tf32_settings():
    return {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def rng_states():
    out = {"torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        out["torch_cuda"] = [v.clone() for v in torch.cuda.get_rng_state_all()]
    return out


class RecordedDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        if self.mask is None or self.mask.shape != x.shape:
            raise RuntimeError("Missing or incorrectly shaped dropout mask")
        return x * self.mask


def dropout_module(model, kind=(nn.Dropout,)):
    found = [m for m in model.modules() if isinstance(m, kind)]
    if len(found) != 1:
        raise RuntimeError("Expected exactly one SmallCNN dropout site")
    return found[0]


def swap_in_recorder(model):
    original = dropout_module(model)
    recorded = RecordedDropout()
    model.classifier[list(model.classifier).index(original)] = recorded
    return recorded


class MaskCapture:
    """Recover ALL Bernoulli draws by replaying dropout on ones from its pre-RNG.

    Inferring a mask from nonzero outputs loses draws at zero activations. Such
    positions may become nonzero in a paired perturbed/high-precision trajectory.
    fork_rng restores the post-forward RNG, so the native training stream stays
    untouched. Full checkpoint replay independently checks that invariant.
    """
    def __init__(self, model, limit):
        self.store, self.limit, self.pre = [], limit, None
        module = dropout_module(model)
        self.handles = [module.register_forward_pre_hook(self.before),
                        module.register_forward_hook(self.after)]

    def before(self, module, inputs):
        if len(self.store) < self.limit:
            self.pre = rng_states()

    def after(self, module, inputs, output):
        if len(self.store) >= self.limit:
            return
        x = inputs[0]
        devices = list(range(torch.cuda.device_count())) if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.pre["torch_cpu"])
            if x.is_cuda:
                torch.cuda.set_rng_state_all(self.pre["torch_cuda"])
            mask = nn.functional.dropout(torch.ones_like(x), p=module.p, training=True)
        if not torch.equal(x * mask, output):
            raise RuntimeError("RNG-replayed dropout mask does not reproduce native output")
        self.store.append(mask.detach().cpu().clone())

    def remove(self):
        for h in self.handles:
            h.remove()


def batch_plan(orders, batch_size, n_steps=None):
    plan = [(epoch, np.asarray(order[k:k + batch_size], dtype=np.int64))
            for epoch, order in enumerate(orders)
            for k in range(0, len(order), batch_size)]
    if n_steps is not None:
        if n_steps < 1 or n_steps > len(plan):
            raise ValueError("Requested prefix is outside the frozen training trajectory")
        plan = plan[:n_steps]
    return plan


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def adam_state(model, optimizer, audit=False):
    ps = list(model.parameters())
    states = [optimizer.state[p] for p in ps]
    steps = {int(s["step"].item()) for s in states}
    if len(steps) != 1:
        raise RuntimeError("Inconsistent Adam step counters")
    out = {"theta": flat(model).cpu(),
           "m": torch.cat([s["exp_avg"].detach().reshape(-1) for s in states]).cpu().clone(),
           "v": torch.cat([s["exp_avg_sq"].detach().reshape(-1) for s in states]).cpu().clone(),
           "step": steps.pop()}
    if audit:
        out.update(state_dict=cpu_tree(model.state_dict()),
                   optimizer_state=cpu_tree(optimizer.state_dict()), rng_states=rng_states())
    return out


def faithful_run(pilot, X, y, orders, seed, cfg, alpha, excluded, n_steps,
                 prefixes=(), record_masks=0, masks=None, dtype=torch.float32,
                 audit_steps=()):
    """Native 05 operations, no TF32 overrides. Float64 is a labelled control."""
    pilot.seed_everything(seed, cfg["deterministic"])
    model = pilot.build_model("cnn", X.shape[1], 128, int(np.max(y)) + 1)
    model.to(dtype=dtype)  # initialization is always the original float32 draw
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    device = pilot.DEVICE
    excluded_t = torch.as_tensor(sorted(excluded), dtype=torch.int64, device=device)
    capture = MaskCapture(model, int(record_masks)) if record_masks else None
    recorded = swap_in_recorder(model) if masks is not None else None
    snapshots, wanted = {}, set(prefixes) | set(audit_steps)
    plan = batch_plan(orders, cfg["batch_size"], n_steps)
    model.train()
    try:
        for k, (_, batch_idx) in enumerate(plan, 1):
            if recorded is not None:
                recorded.mask = masks[k-1].to(device=device, dtype=dtype)
            xb = torch.as_tensor(X[batch_idx], dtype=dtype, device=device)
            yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            per = nn.functional.cross_entropy(model(xb), yb, reduction="none")
            weights = torch.ones_like(per)
            if excluded_t.numel():
                idx_t = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
                weights = torch.where(torch.isin(idx_t, excluded_t),
                                      weights * (1.0-alpha), weights)
            loss = (weights * per).sum() / per.numel()
            loss.backward()
            optimizer.step()
            if k in wanted:
                snapshots[k] = adam_state(model, optimizer, k in audit_steps)
    finally:
        if capture is not None:
            capture.remove()
    return model, optimizer, snapshots, (capture.store if capture else masks), len(plan)


def make_gradient(model, cfg):
    named = list(model.named_parameters())
    if list(model.buffers()):
        raise RuntimeError("This B1 runner requires the buffer-free frozen SmallCNN")
    sizes = [p.numel() for _, p in named]

    def unflatten(theta):
        return {name: part.view(p.shape)
                for (name, p), part in zip(named, theta.split(sizes))}

    def loss(theta, alpha, xb, yb, removed):
        per = nn.functional.cross_entropy(functional_call(model, unflatten(theta), (xb,)),
                                          yb, reduction="none")
        weights = torch.where(removed, torch.ones_like(per) * (1.0-alpha), torch.ones_like(per))
        return (weights * per).sum() / per.numel()

    loss_grad = grad(loss)

    def gradient(theta, alpha, xb, yb, removed):
        return loss_grad(theta, alpha, xb, yb, removed).add(theta, alpha=cfg["wd"])
    return gradient, unflatten


def adam_primal(theta, m, v, g, step, lr):
    """PyTorch's single-tensor operation ordering, not a reassociated formula.

    CUDA foreach equivalence is measured, never presumed or accepted by tolerance.
    """
    b1, b2 = BETAS
    m1 = torch.lerp(m, g, 1-b1)
    v1 = v.mul(b2).addcmul(g, g, value=1-b2)
    denom = v1.sqrt().div(math.sqrt(1-b2**step)).add(EPS)
    th1 = theta.addcdiv(m1, denom, value=-lr/(1-b1**step))
    return th1, m1, v1


def adam_tangent(m1, v1, g, q, ut, um, uv, step, lr):
    """Reachable-state directional derivative, with no clipping or epsilon change.

    At v1=0 a reachable exact-arithmetic state also has m1=uv1=0.
    The root alone is not differentiable there; the composite parameter update
    has first derivative -lr/bc1 * um1/EPS. Avoid the spurious 0*inf from AD of
    sqrt(g*g). A non-reachable/underflow-degenerate zero state is rejected.
    """
    b1, b2 = BETAS
    um1 = b1*um + (1-b1)*q
    uv1 = b2*uv + 2*(1-b2)*g*q
    root = v1.sqrt()
    zero = root == 0
    if bool(torch.any(zero & ((m1 != 0) | (uv1 != 0)))):
        raise RuntimeError("Degenerate zero second moment with nonzero moment/tangent")
    root_safe = torch.where(zero, torch.ones_like(root), root)
    dden = torch.where(zero, torch.zeros_like(root), uv1/(2*root_safe)) / math.sqrt(1-b2**step)
    den = root / math.sqrt(1-b2**step) + EPS
    ut1 = ut - (lr/(1-b1**step)) * (um1/den - m1*dden/(den*den))
    return ut1, um1, uv1


def norm(x):
    return float(torch.linalg.vector_norm(x.detach().cpu().double()))


def compare(value, reference):
    a, b = value.detach().cpu().double().reshape(-1), reference.detach().cpu().double().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError("Non-finite comparison vector")
    na, nb, error = norm(a), norm(b), norm(a-b)
    return {"norm": na, "reference_norm": nb, "absolute_error": error,
            "relative_error": error/nb if nb else None,
            "cosine": float(torch.dot(a/na, b/nb)) if na and nb else None,
            "bitwise_equal": bool(torch.equal(value.cpu(), reference.cpu()))}


def fd(perturbed, baseline, alpha):
    # Preserve all stored bits before differencing; do not round subtraction in f32.
    return (perturbed.detach().cpu().double()-baseline.detach().cpu().double())/alpha


def functional_trajectory(model, X, y, cfg, device, theta0, plan, masks, excluded,
                          alpha, prefixes, tangent=False):
    gradient, _ = make_gradient(model, cfg)
    recorder = dropout_module(model, (RecordedDropout,))
    theta = theta0.detach().clone()
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    ut, um, uv = torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)
    excluded_t = torch.as_tensor(sorted(excluded), dtype=torch.int64, device=device)
    out, worst_ad_primal_gap = {}, 0.0
    model.train()
    for k, (_, idx) in enumerate(plan, 1):
        recorder.mask = masks[k-1].to(device=device, dtype=theta.dtype)
        xb = torch.as_tensor(X[idx], dtype=theta.dtype, device=device)
        yb = torch.as_tensor(y[idx], dtype=torch.long, device=device)
        removed = torch.isin(torch.as_tensor(idx, dtype=torch.long, device=device), excluded_t)
        al = torch.as_tensor(alpha, dtype=theta.dtype, device=device)
        g = gradient(theta, al, xb, yb, removed)
        if tangent:
            g_ad, q = jvp(lambda th, a: gradient(th, a, xb, yb, removed),
                          (theta, al), (ut, torch.ones_like(al)))
            gap = norm(g_ad-g)/max(norm(g), 1e-300)
            worst_ad_primal_gap = max(worst_ad_primal_gap, gap)
        theta, m, v = adam_primal(theta, m, v, g, k, cfg["lr"])
        if tangent:
            ut, um, uv = adam_tangent(m, v, g, q, ut, um, uv, k, cfg["lr"])
        theta, m, v, ut, um, uv = [x.detach() for x in (theta, m, v, ut, um, uv)]
        if not all(bool(torch.isfinite(x).all()) for x in (theta, m, v, ut, um, uv)):
            raise RuntimeError(f"Non-finite state/tangent at step {k}")
        if k in prefixes:
            out[k] = {"theta": theta.cpu().clone(), "m": m.cpu().clone(), "v": v.cpu().clone(),
                      "step": k, "max_gradient_ad_primal_relative_gap": worst_ad_primal_gap}
            if tangent:
                out[k].update(u_theta=ut.cpu().clone(), u_m=um.cpu().clone(), u_v=uv.cpu().clone())
    return out


def prediction_jvp(model, X, rows, theta, u, device, batch=64):
    _, unflatten = make_gradient(model, {"wd": 0.0})
    theta = theta.to(device)
    u = u.to(device) if u is not None else None
    predictions, tangents = [], []
    was = model.training
    model.eval()
    try:
        for start in range(0, len(rows), batch):
            xb = torch.as_tensor(X[rows[start:start+batch]], dtype=theta.dtype, device=device)
            def predict(th):
                return torch.softmax(functional_call(model, unflatten(th), (xb,)), dim=1)
            if u is None:
                value, tangent_value = predict(theta), None
            else:
                value, tangent_value = jvp(predict, (theta,), (u,))
            predictions.append(value.detach().cpu())
            if tangent_value is not None:
                tangents.append(tangent_value.detach().cpu())
    finally:
        model.train(was)
    return torch.cat(predictions), torch.cat(tangents) if u is not None else None


def exact_alpha(exponent, dtype=torch.float32):
    limit = 24 if dtype == torch.float32 else 53
    if not 1 <= exponent <= limit:
        raise ValueError(f"Exponent must lie in 1..{limit} for {dtype}")
    a = 2.0**-exponent
    if 1-float(torch.tensor(1-a, dtype=dtype)) != a:
        raise RuntimeError("Deletion weight is not exactly representable")
    return a


def concentration(u):
    sq = u.detach().cpu().double().square()
    ranked = sq.sort(descending=True).values
    total, n = float(sq.sum()), sq.numel()
    return {"tangent_norm": math.sqrt(total),
            "top_0p1pct_energy_share": float(ranked[:max(1, math.ceil(n*.001))].sum())/total if total else None,
            "top_1pct_energy_share": float(ranked[:max(1, math.ceil(n*.01))].sum())/total if total else None,
            "largest_coordinate_energy_share": float(ranked[0])/total if total else None}


def first_step_epsilon(g, q, u, lr):
    g, q, u = [x.detach().cpu().double() for x in (g, q, u)]
    slope = EPS/(g.abs()+EPS).square()
    predicted = -lr*slope*q
    total = float(u.square().sum())
    return {"scope": "first update only; g includes coupled weight decay; q=dg/dalpha",
            "g_abs_quantiles": {str(p): float(torch.quantile(g.abs(), p)) for p in (.01,.1,.5,.9,.99)},
            "fraction_coordinates_abs_g_lt_10eps": float((g.abs()<10*EPS).double().mean()),
            "normalized_slope_max": float(slope.max()), "update_gain_max": float(lr*slope.max()),
            "q_norm": norm(q), "analytic_u1_check": compare(u, predicted),
            "energy_in_abs_g_lt_eps": float(u[g.abs()<EPS].square().sum())/total if total else None,
            "energy_in_abs_g_lt_10eps": float(u[g.abs()<10*EPS].square().sum())/total if total else None,
            **concentration(u),
            "interpretation": "Small total gradients can reflect cancellation. Concentration alone does not imply dead coordinates, a numerical error, or an invalid Eq.74."}
