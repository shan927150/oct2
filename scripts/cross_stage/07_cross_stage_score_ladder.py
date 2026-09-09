#!/usr/bin/env python3
"""Cross-stage influence score (continue_pathway2 Eq. 75-79) + oracle ladder.

Runs OFFLINE on the artifacts of a finished 05 pilot directory: it never
retrains Stage 1.  Everything it needs already exists there:

    checkpoints/shadow_<a>_baseline_seed<r>.pt         theta_base
    checkpoints/shadow_<a>_seed<r>_patient<pid>.pt     theta_loo   (true Delta theta)
    checkpoints/shadow_<k>_fixed.pt, target_fixed.pt   fixed models
    runs/seed<r>_patient<pid>.json                     J00/J10/J01/J11
    runs/seed<r>_patient<pid>_interface.npz            p_baseline / p_loo
    splits/fresh_patient_split.json, selected_patients.json

Per (seed, class) it computes the implicit-differentiation quantities of the
derivation, all for the matched-class endpoint J_{Q,c} (omega_c = 1):

    q_c   = grad_phi J_{Q,c}(phi*)                                (Eq. 75)
    u_c   = (H_A + gamma_A I)^{-1} q_c                            (Eq. 21)
    v_sj  = - B_sj^T u_c = - d/dp_sj [ u_c . grad_phi L_A ]       (Eq. 76)
    h_s   = grad_theta sum_j v_sj^T p_sj(theta)                   (Eq. 77)
    w_s   = (H_s + gamma_s I)^{-1} h_s     (CG with HVPs)
    IF_i  = (1/n_s) w_s^T g_si                                    (Eq. 61/78)
    IF_patient = sum_{i in patient} IF_i

and evaluates an "oracle ladder" that tells you WHERE the chain breaks:

    actual       Delta_value = J10-J00, Delta_full = J11-J00  (from runs/*.json)
    L1_lin       sum_j v_j^T (p1_j - p0_j)         true P1, linear attack response
    L2_lin       h_s^T (theta_loo - theta_base)    true Delta theta, linear p and attack
    L2_retrain   retrain attack on P0 + J_p Delta theta (true Delta theta)
    L3_lin       IF_patient                        full continuous score
    L3_retrain   retrain attack on P0 + J_p Delta theta_hat (implicit Delta theta)
    L3_hybrid    same as L3_retrain but with relabeled M1 -> predicts Delta_full
    frozen_h     sum_i h_s^T g_si                  one-checkpoint TracIn-style
    frozen_self  sum_i <grad CE(A(p_i),m_i), g_si> old pilot proxy (Eq. 80)

Optional ``--attack_seed_reps K`` re-trains the Stage-2 attack under K extra
attack seeds for the four (P, M) conditions using the TRUE P1, giving the
Stage-2-only noise SD of Delta_value / Delta_full for every patient-seed run.

Usage (Delta, GPU recommended, ~minutes):
    python scripts/cross_stage/07_cross_stage_score_ladder.py \
        --pilot_dir results/cross_stage_patient_loo_05_pilot_v3 --data_dir ./data
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

HERE = Path(__file__).resolve().parent
LOGGER = logging.getLogger("score_ladder")


def import_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "pilot05", HERE / "05_end_to_end_patient_loo_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pilot05"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def flat_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(vec: torch.Tensor, like: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out, k = [], 0
    for t in like:
        n = t.numel()
        out.append(vec[k:k + n].view_as(t))
        k += n
    return out


def param_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in model.named_parameters()}


# ---------------------------------------------------------------------------
# Stage 2 (attack) implicit differentiation  -- Eq. 11-21, 75-76
# ---------------------------------------------------------------------------

def attack_loss_fn(model, P: torch.Tensor, M: torch.Tensor, params=None) -> torch.Tensor:
    if params is None:
        return F.cross_entropy(model(P), M)
    return F.cross_entropy(functional_call(model, params, (P,)), M)


def attack_implicit_v(model, P_train: np.ndarray, M_train: np.ndarray,
                      P_query: np.ndarray, M_query: np.ndarray,
                      damping: float, device) -> Dict[str, object]:
    """Return v (N_train x 4) = dJ_Q/dp_j via implicit differentiation, plus diagnostics."""
    model.eval()
    names = [n for n, p in model.named_parameters()]
    params = [p.detach().clone().requires_grad_(True) for _, p in model.named_parameters()]
    pdict = dict(zip(names, params))
    dt = model_dtype(model)
    P = torch.as_tensor(P_train, dtype=dt, device=device).requires_grad_(True)
    M = torch.as_tensor(M_train, dtype=torch.long, device=device)
    Pq = torch.as_tensor(P_query, dtype=dt, device=device)
    Mq = torch.as_tensor(M_query, dtype=torch.long, device=device)

    # q = grad_phi J_Q
    Jq = F.cross_entropy(functional_call(model, pdict, (Pq,)), Mq)
    q = flatten(torch.autograd.grad(Jq, params))

    # training-loss gradient (with graph) and its norm (stationarity check)
    L = F.cross_entropy(functional_call(model, pdict, (P,)), M)
    gL = torch.autograd.grad(L, params, create_graph=True)
    gL_flat = flatten(gL)
    grad_norm = float(gL_flat.detach().norm())

    # dense Hessian of L wrt phi (d ~ 450)
    d = gL_flat.numel()
    H = torch.zeros(d, d, device=device, dtype=dt)
    for k in range(d):
        row = torch.autograd.grad(gL_flat[k], params, retain_graph=True, allow_unused=True)
        H[k] = flatten([r if r is not None else torch.zeros_like(p) for r, p in zip(row, params)])
    H = 0.5 * (H + H.T)
    eig = torch.linalg.eigvalsh(H)
    u = torch.linalg.solve(H + damping * torch.eye(d, device=device, dtype=dt), q)

    # v_j = - d/dp_j [u . grad_phi L]
    s = (gL_flat * u).sum()
    V = torch.autograd.grad(s, P)[0]
    v = (-V).detach().cpu().numpy()
    return {
        "v": v,
        "u": u.detach(),
        "q_norm": float(q.norm()),
        "train_grad_norm": grad_norm,
        "hessian_eig_min": float(eig.min()),
        "hessian_eig_max": float(eig.max()),
        "damping": damping,
    }


# ---------------------------------------------------------------------------
# Stage 1 (shadow) quantities  -- Eq. 48-49, 53-61
# ---------------------------------------------------------------------------

def shadow_h(model, X_rows: np.ndarray, v_rows: np.ndarray, device, batch: int = 128) -> torch.Tensor:
    """h_s = grad_theta sum_j v_j^T softmax(f_theta(x_j)), eval mode."""
    model.eval()
    params = flat_params(model)
    dt = model_dtype(model)
    h = torch.zeros(sum(p.numel() for p in params), device=device, dtype=dt)
    for s in range(0, len(X_rows), batch):
        xb = torch.as_tensor(X_rows[s:s + batch], dtype=dt, device=device)
        vb = torch.as_tensor(v_rows[s:s + batch], dtype=dt, device=device)
        obj = (torch.softmax(model(xb), dim=1) * vb).sum()
        g = torch.autograd.grad(obj, params, allow_unused=True)
        h += flatten([gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)])
    return h


def shadow_loss_grad(model, X: np.ndarray, y: np.ndarray, idx: np.ndarray, device,
                     weight_decay: float, batch: int = 128) -> torch.Tensor:
    """grad_theta [ mean CE over idx + wd/2 ||theta||^2 ]  (Adam-L2 objective)."""
    model.eval()
    params = flat_params(model)
    dt = model_dtype(model)
    g = torch.zeros(sum(p.numel() for p in params), device=device, dtype=dt)
    n = len(idx)
    for s in range(0, n, batch):
        b = idx[s:s + batch]
        xb = torch.as_tensor(X[b], dtype=dt, device=device)
        yb = torch.as_tensor(y[b], dtype=torch.long, device=device)
        loss = F.cross_entropy(model(xb), yb, reduction="sum") / n
        gb = torch.autograd.grad(loss, params)
        g += flatten(gb)
    g += weight_decay * flatten([p.detach() for p in params])
    return g


def per_image_grads(model, X: np.ndarray, y: np.ndarray, idx: Sequence[int], device) -> torch.Tensor:
    model.eval()
    params = flat_params(model)
    dt = model_dtype(model)
    out = []
    for i in idx:
        xb = torch.as_tensor(X[[i]], dtype=dt, device=device)
        yb = torch.as_tensor(y[[i]], dtype=torch.long, device=device)
        loss = F.cross_entropy(model(xb), yb)
        out.append(flatten(torch.autograd.grad(loss, params)))
    return torch.stack(out)


def make_hvp(model, X: np.ndarray, y: np.ndarray, idx: np.ndarray, device,
             weight_decay: float, damping: float, batch: int = 128):
    """Return a function vec -> (H_s + wd I + damping I) vec using double backward."""
    model.eval()
    params = flat_params(model)
    n = len(idx)
    dt = model_dtype(model)

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(vec)
        vs = unflatten(vec, params)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb = torch.as_tensor(X[b], dtype=dt, device=device)
            yb = torch.as_tensor(y[b], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(xb), yb, reduction="sum") / n
            g = torch.autograd.grad(loss, params, create_graph=True)
            gv = sum((gi * vi).sum() for gi, vi in zip(g, vs))
            hv = torch.autograd.grad(gv, params, allow_unused=True)
            out += flatten([h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)])
        return out + (weight_decay + damping) * vec
    return hvp


def conjugate_gradient(hvp, b: torch.Tensor, iters: int, tol: float) -> Dict[str, object]:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = r @ r
    b_norm = float(b.norm()) + 1e-30
    history = []
    for k in range(iters):
        Ap = hvp(p)
        alpha = rs / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        rel = float(rs_new.sqrt()) / b_norm
        history.append(rel)
        if rel < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return {"x": x, "iters": len(history), "final_rel_residual": history[-1] if history else None}


def linearized_predictions(model, X_rows: np.ndarray, dtheta: torch.Tensor, device,
                           batch: int = 128) -> np.ndarray:
    """p0 + (dp/dtheta) dtheta for every row, via forward-mode JVP; renormalised to the simplex."""
    model.eval()
    names = [n for n, _ in model.named_parameters()]
    base = {n: p.detach() for n, p in model.named_parameters()}
    tangent = dict(zip(names, unflatten(dtheta, [base[n] for n in names])))
    out = []
    dt = model_dtype(model)
    for s in range(0, len(X_rows), batch):
        xb = torch.as_tensor(X_rows[s:s + batch], dtype=dt, device=device)

        def f(params):
            return torch.softmax(functional_call(model, params, (xb,)), dim=1)

        p0, dp = jvp(f, (base,), (tangent,))
        p1 = torch.clamp(p0 + dp, min=1e-6)
        p1 = p1 / p1.sum(dim=1, keepdim=True)
        out.append(p1.detach().cpu().numpy())
    return np.concatenate(out)


def frozen_self_scores(shadow, attack, X: np.ndarray, y: np.ndarray, idx: Sequence[int],
                       device) -> np.ndarray:
    """Old proxy (Eq. 80): <grad_theta CE(A(p_i(theta)), m_i=1), grad_theta CE(f(x_i), y_i)>."""
    shadow.eval(); attack.eval()
    params = flat_params(shadow)
    dt = model_dtype(shadow)
    out = []
    for i in idx:
        xb = torch.as_tensor(X[[i]], dtype=dt, device=device)
        yb = torch.as_tensor(y[[i]], dtype=torch.long, device=device)
        logits = shadow(xb)
        j = F.cross_entropy(attack(torch.softmax(logits, dim=1)), torch.tensor([1], device=device))
        gd = torch.autograd.grad(j, params, retain_graph=True, allow_unused=True)
        gc = torch.autograd.grad(F.cross_entropy(logits, yb), params)
        out.append(float(sum(((a if a is not None else torch.zeros_like(p)) * b).sum()
                             for a, b, p in zip(gd, gc, params))))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", required=True)
    ap.add_argument("--data_dir", default=None, help="override data_dir stored in experiment_config")
    ap.add_argument("--out_dir", default=None, help="default: <pilot_dir>/score_ladder")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="subset of qualified OCT classes to process (default: all qualified)")
    ap.add_argument("--damping_attack", type=float, default=1e-3)
    ap.add_argument("--damping_shadow", type=float, default=1e-2)
    ap.add_argument("--cg_iters", type=int, default=100)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--hvp_batch", type=int, default=64,
                    help="images per double-backward chunk; lower it if memory is tight")
    ap.add_argument("--attack_seed_reps", type=int, default=0,
                    help="extra attack seeds per condition for Stage-2-only noise (0 = skip)")
    ap.add_argument("--skip_frozen_self", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    t0 = time.time()

    pilot = import_pilot_module()
    device = pilot.DEVICE
    pilot_dir = Path(args.pilot_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else pilot_dir / "score_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload = json.loads((pilot_dir / "experiment_config.json").read_text(encoding="utf-8"))
    pargs = argparse.Namespace(**cfg_payload["args"])
    if args.data_dir:
        pargs.data_dir = args.data_dir
    pargs.overwrite = False
    seeds = args.seeds or list(pargs.seeds)
    a = int(pargs.affected_shadow)
    summary = json.loads((pilot_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    loo_classes = [int(c) for c in summary.get("qualified_classes", pargs.classes)]
    if args.classes:
        loo_classes = [c for c in loo_classes if c in set(args.classes)]
    LOGGER.info("pilot=%s affected_shadow=%d seeds=%s classes=%s", pilot_dir, a, seeds, loo_classes)

    cfg = pilot.build_config(pargs)
    X, y, groups = pilot.load_dataset(cfg)
    _, splits = pilot.prepare_split(cfg, X, y, groups, pilot_dir, overwrite=False)
    n_classes = int(np.max(y)) + 1
    ckpt = pilot_dir / "checkpoints"
    n_hidden = 128

    target_model = pilot.load_model(ckpt / "target_fixed.pt", X.shape[1], n_hidden, n_classes)
    target_queries = pilot.make_target_queries(target_model, splits, X, y)
    fixed = {}
    for sid in range(pargs.n_shadow):
        if sid != a:
            fixed[sid] = pilot.load_model(ckpt / f"shadow_{sid}_fixed.pt", X.shape[1], n_hidden, n_classes)
    patients = json.loads((pilot_dir / "selected_patients.json").read_text(encoding="utf-8"))["patients"]
    patients = [p for p in patients if int(p["oct_class"]) in loo_classes]
    affected_train = np.asarray(splits["shadow_models"][a]["train_idx"], dtype=np.int64)
    n_s = len(affected_train)
    wd = 1e-5  # Stage-1 Adam weight decay used by 05

    rows: List[dict] = []
    diagnostics: Dict[str, object] = {}
    for seed in seeds:
        LOGGER.info("=== seed %d ===", seed)
        base_model = pilot.load_model(ckpt / f"shadow_{a}_baseline_seed{seed}.pt", X.shape[1], n_hidden, n_classes)
        models = [base_model if sid == a else fixed[sid] for sid in range(pargs.n_shadow)]
        interface0 = pilot.make_interface(models, splits, X, y)
        aff = interface0["shadow_id"] == a

        # Stage-1 gradient at theta_base (stationarity diagnostic) and HVP operator
        gL = shadow_loss_grad(base_model, X, y, affected_train, device, wd, args.hvp_batch)
        hvp = make_hvp(base_model, X, y, affected_train, device, wd, args.damping_shadow, args.hvp_batch)
        diagnostics[f"seed{seed}"] = {"stage1_train_grad_norm": float(gL.norm()), "classes": {}}

        for cls in loo_classes:
            tr = interface0["classes"] == cls
            te = target_queries["classes"] == cls
            attack_seed = int(seed * 100 + cls)
            n_reps = max(1, int(getattr(pargs, "attack_seed_reps", 1)))

            def train_attack(x_tr, m_tr, rep_seed):
                return pilot.train_attack_model_deterministic(
                    x_tr, m_tr, rep_seed, pargs.attack_epochs, pargs.attack_lr,
                    pargs.attack_batch_size, n_hidden=64, deterministic=pargs.deterministic)

            def ce_avg(x_tr, m_tr) -> float:
                """Matched-class CE averaged over the same attack-seed reps 05 used."""
                vals = []
                for rep in range(n_reps):
                    rep_seed = attack_seed if rep == 0 else attack_seed + 10007 * rep
                    m = train_attack(x_tr, m_tr, rep_seed)
                    pr = pilot.predict_attack_prob(m, target_queries["x"][te])
                    vals.append(pilot.binary_metrics(target_queries["membership"][te], pr)["cross_entropy"])
                return float(np.mean(vals))

            attack = train_attack(interface0["x"][tr], interface0["membership"][tr], attack_seed)
            J00 = ce_avg(interface0["x"][tr], interface0["membership"][tr])

            imp = attack_implicit_v(
                attack, interface0["x"][tr], interface0["membership"][tr],
                target_queries["x"][te], target_queries["membership"][te],
                args.damping_attack, device)
            v_all = imp["v"]                      # rows of class cls, all shadows
            tr_idx = np.where(tr)[0]
            aff_cls = aff[tr]                     # affected-shadow rows within class rows
            raw_aff = interface0["raw_index"][tr_idx[aff_cls]]
            v_aff = v_all[aff_cls]
            h = shadow_h(base_model, X[raw_aff], v_aff, device)
            cg = conjugate_gradient(hvp, h, args.cg_iters, args.cg_tol)
            w = cg["x"]
            diagnostics[f"seed{seed}"]["classes"][str(cls)] = {
                "J00_recomputed": float(J00),
                "attack": {k: val for k, val in imp.items() if k not in ("v", "u")},
                "h_norm": float(h.norm()),
                "cg_iters": cg["iters"], "cg_final_rel_residual": cg["final_rel_residual"],
                "n_affected_rows": int(len(raw_aff)),
            }
            LOGGER.info("seed=%d class=%d J00=%.6f |gradL_A|=%.2e H_A eig[%.2e, %.2e] |h|=%.3e CG iters=%d res=%.2e",
                        seed, cls, J00, imp["train_grad_norm"], imp["hessian_eig_min"],
                        imp["hessian_eig_max"], float(h.norm()), cg["iters"], cg["final_rel_residual"])

            for pat in [p for p in patients if int(p["oct_class"]) == cls]:
                pid = int(pat["patient_id"])
                run_path = pilot_dir / "runs" / f"seed{seed}_patient{pid}.json"
                npz_path = pilot_dir / "runs" / f"seed{seed}_patient{pid}_interface.npz"
                loo_path = ckpt / f"shadow_{a}_seed{seed}_patient{pid}.pt"
                if not (run_path.exists() and npz_path.exists() and loo_path.exists()):
                    LOGGER.warning("missing artifacts for seed=%d patient=%d; skipping", seed, pid)
                    continue
                run = json.loads(run_path.read_text(encoding="utf-8"))
                cond = {c: run["conditions"][c]["per_class"][str(cls)]["cross_entropy"]
                        for c in ("J00", "J10", "J01", "J11")}
                npz = np.load(npz_path)
                # align NPZ (affected rows, all classes) to class-cls affected rows
                npz_cls = npz["oct_class"] == cls
                assert np.array_equal(npz["raw_index"][npz_cls], raw_aff), "row alignment mismatch"
                p0 = npz["p_baseline"][npz_cls]
                p1 = npz["p_loo"][npz_cls]
                removed = np.asarray(pat["raw_indices"], dtype=np.int64)

                # --- Stage-1 quantities for this patient
                G = per_image_grads(base_model, X, y, removed, device)           # (n_i, d)
                if_per_image = (G @ w / n_s).cpu().numpy()                       # Eq. 61
                frozen_h = (G @ h).cpu().numpy()
                loo_model = pilot.load_model(loo_path, X.shape[1], n_hidden, n_classes)
                dtheta_true = flatten([q.detach() for q in flat_params(loo_model)]) - \
                    flatten([q.detach() for q in flat_params(base_model)])
                gsum = G.sum(dim=0)
                cg_p = conjugate_gradient(hvp, gsum / n_s, args.cg_iters, args.cg_tol)
                dtheta_hat = cg_p["x"]                                           # Eq. 56 with eps=-1/n

                # --- ladder predictions of Delta_value
                L1_lin = float((v_aff * (p1 - p0)).sum())
                L2_lin = float(h @ dtheta_true)
                L3_lin = float(if_per_image.sum())
                p1_hat_true = linearized_predictions(base_model, X[raw_aff], dtheta_true, device)
                p1_hat_if = linearized_predictions(base_model, X[raw_aff], dtheta_hat, device)

                def retrain_ce(p_aff_rows: np.ndarray, membership: np.ndarray) -> float:
                    x_tr = interface0["x"][tr].copy()
                    x_tr[aff_cls] = p_aff_rows
                    return ce_avg(x_tr, membership)

                M0 = interface0["membership"][tr]
                M1 = pilot.membership_after_removal(interface0, removed, a)[tr]
                L2_retrain = retrain_ce(p1_hat_true, M0) - J00
                L3_retrain = retrain_ce(p1_hat_if, M0) - J00
                L3_hybrid = retrain_ce(p1_hat_if, M1) - J00
                L2_hybrid = retrain_ce(p1_hat_true, M1) - J00
                # relabel-only prediction at baseline vectors (exact retrain, cheap) = J01 - J00
                row = {
                    "seed": seed, "patient_id": pid, "oct_class": cls, "class_name": pat["class_name"],
                    "n_images": len(removed),
                    "J00": cond["J00"], "J10": cond["J10"], "J01": cond["J01"], "J11": cond["J11"],
                    "J00_recomputed": J00,
                    "actual_value": cond["J10"] - cond["J00"],
                    "actual_relabel": cond["J01"] - cond["J00"],
                    "actual_full": cond["J11"] - cond["J00"],
                    "actual_relabel_given_P1": cond["J11"] - cond["J10"],
                    "L1_lin_value": L1_lin,
                    "L2_lin_value": L2_lin,
                    "L2_retrain_value": L2_retrain,
                    "L2_hybrid_full": L2_hybrid,
                    "L3_lin_value": L3_lin,
                    "L3_retrain_value": L3_retrain,
                    "L3_hybrid_full": L3_hybrid,
                    "frozen_h": float(frozen_h.sum()),
                    "dtheta_true_norm": float(dtheta_true.norm()),
                    "dtheta_hat_norm": float(dtheta_hat.norm()),
                    "dtheta_cosine": float((dtheta_true @ dtheta_hat) /
                                           (dtheta_true.norm() * dtheta_hat.norm() + 1e-30)),
                    "p1hat_true_mean_js_deleted": float(pilot.js_divergence(
                        p0[npz["is_deleted_patient"][npz_cls]], p1_hat_true[npz["is_deleted_patient"][npz_cls]]).mean()),
                    "p1_true_mean_js_deleted": float(pilot.js_divergence(
                        p0[npz["is_deleted_patient"][npz_cls]], p1[npz["is_deleted_patient"][npz_cls]]).mean()),
                    "cg_patient_iters": cg_p["iters"],
                }
                if not args.skip_frozen_self:
                    row["frozen_self"] = float(frozen_self_scores(base_model, attack, X, y, removed, device).sum())

                if args.attack_seed_reps > 0:
                    dv, df = [], []
                    x_full = interface0["x"][tr].copy(); x_full[aff_cls] = p1
                    for k in range(args.attack_seed_reps):
                        s_k = attack_seed + 7919 * (k + 1)   # distinct from the 05 rep seeds

                        def ce(xx, mm):
                            m = train_attack(xx, mm, s_k)
                            pr = pilot.predict_attack_prob(m, target_queries["x"][te])
                            return pilot.binary_metrics(target_queries["membership"][te], pr)["cross_entropy"]
                        j00 = ce(interface0["x"][tr], M0)
                        dv.append(ce(x_full, M0) - j00)
                        df.append(ce(x_full, M1) - j00)
                    row["attackseed_value_mean"] = float(np.mean(dv)); row["attackseed_value_sd"] = float(np.std(dv, ddof=1)) if len(dv) > 1 else None
                    row["attackseed_full_mean"] = float(np.mean(df)); row["attackseed_full_sd"] = float(np.std(df, ddof=1)) if len(df) > 1 else None
                rows.append(row)
                LOGGER.info("seed=%d pid=%d actual value=%+.5f full=%+.5f | L1=%+.5f L2=%+.5f/%+.5f L3=%+.5f/%+.5f hybrid=%+.5f cos=%.3f",
                            seed, pid, row["actual_value"], row["actual_full"], L1_lin, L2_lin, L2_retrain,
                            L3_lin, L3_retrain, L3_hybrid, row["dtheta_cosine"])
                write_rows(out_dir / "ladder_rows.csv", rows)

    write_rows(out_dir / "ladder_rows.csv", rows)
    analysis = analyze(rows)
    (out_dir / "ladder_summary.json").write_text(json.dumps({
        "pilot_dir": str(pilot_dir), "args": vars(args), "elapsed_seconds": time.time() - t0,
        "diagnostics": diagnostics, "analysis": analysis,
    }, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))
    LOGGER.info("done in %.1fs -> %s", time.time() - t0, out_dir)


def write_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + [k for k in fields if k not in rows[0]])
        w.writeheader(); w.writerows(rows)


def analyze(rows: List[dict]) -> Dict[str, object]:
    from scipy.stats import pearsonr, spearmanr
    if len(rows) < 3:
        return {"n": len(rows)}
    pairs = {
        "value": [("L1_lin_value", "actual_value"), ("L2_lin_value", "actual_value"),
                  ("L2_retrain_value", "actual_value"), ("L3_lin_value", "actual_value"),
                  ("L3_retrain_value", "actual_value"), ("frozen_h", "actual_value"),
                  ("frozen_self", "actual_value")],
        "full": [("L2_hybrid_full", "actual_full"), ("L3_hybrid_full", "actual_full"),
                 ("L3_lin_value", "actual_full"), ("frozen_h", "actual_full"),
                 ("frozen_self", "actual_full")],
    }

    def corr(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
            return {"n": int(len(xs))}
        return {"n": int(len(xs)), "spearman": float(spearmanr(xs, ys)[0]),
                "pearson": float(pearsonr(xs, ys)[0]),
                "sign_agreement": float(np.mean(np.sign(xs) == np.sign(ys))),
                "pred_mean": float(xs.mean()), "actual_mean": float(ys.mean()),
                "pred_sd": float(xs.std(ddof=1)), "actual_sd": float(ys.std(ddof=1))}

    out = {"per_patient_seed": {}, "per_patient_mean": {}}
    by_pid: Dict[int, List[dict]] = {}
    for r in rows:
        by_pid.setdefault(int(r["patient_id"]), []).append(r)
    for level, plist in pairs.items():
        for pk, ak in plist:
            if pk not in rows[0]:
                continue
            out["per_patient_seed"][f"{pk}~{ak}"] = corr([r[pk] for r in rows], [r[ak] for r in rows])
            out["per_patient_mean"][f"{pk}~{ak}"] = corr(
                [np.mean([r[pk] for r in rs]) for rs in by_pid.values()],
                [np.mean([r[ak] for r in rs]) for rs in by_pid.values()])
    if "attackseed_value_sd" in rows[0]:
        out["stage2_only_noise"] = {
            "mean_sd_value": float(np.mean([r["attackseed_value_sd"] for r in rows if r["attackseed_value_sd"] is not None])),
            "mean_sd_full": float(np.mean([r["attackseed_full_sd"] for r in rows if r["attackseed_full_sd"] is not None])),
        }
    out["dtheta_cosine_mean"] = float(np.mean([r["dtheta_cosine"] for r in rows]))
    return out


if __name__ == "__main__":
    main()
