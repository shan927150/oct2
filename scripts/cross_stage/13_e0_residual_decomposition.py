#!/usr/bin/env python3
"""E0: exact residual decomposition of the static Stage 1 equation on existing checkpoints.

No training. For every (seed, core patient, alpha in {1, 0.1}) it evaluates, in the
eval-mode objective  L_0(theta) = mean CE over the affected shadow training set +
wd/2 ||theta||^2  and  L_alpha = L_0 - alpha * L_patient  (fixed_mask semantics,
N unchanged):

    g_0   = grad L_0(theta_0)                 (baseline endpoint gradient)
    g_a   = grad L_alpha(theta_alpha)         (perturbed endpoint gradient, own objective)
    d     = theta_alpha - theta_0
    b(t)  = (1/N) sum_{i in patient} grad ell_i(t)      (no weight decay)
    Hd    = Hbar d,  Hbar = Hessian of L_0 at theta_0 (already includes wd)
    R_a   = grad L_0(theta_alpha) - g_0 - Hd - alpha [b(theta_alpha) - b(theta_0)]

and reports the exact identity

    (Hbar + gamma I) d - alpha b(theta_0) = (g_a - g_0) - R_a + gamma d .

The left side is the reverse residual of the damped static system evaluated at the
true displacement; the right side splits it into an endpoint-stationarity term
(g_a - g_0) and a finite-displacement Taylor remainder (R_a).  Norms of the two
terms are not additive causal shares; the signed projections onto the residual
direction are reported as well and do sum to one.

Also writes a per-seed signal monitor for the whole patient panel at theta_0
(||b_p||, per-image gradient norms, cancellation ratio, CE, correct-class
probability, dropout-MC b_p).  Inputs are read only; outputs go to a new
directory that must not overlap either truth directory.
"""
from __future__ import annotations
import argparse
import hashlib
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402

LOG = logging.getLogger("e0_residual")
SCHEMA = "pathway2_e0_residual_decomposition_v1"


def load_modules():
    """07 (score), 05 (pilot) and 11 (legacy loaders) exactly as v1.1 does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("e0_legacy11", HERE / "11_stage1_diagnostics.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    score = legacy.module_from_path("e0_score07", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    return legacy, score, pilot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full_dir", required=True, help="original alpha=1 truth directory (read only)")
    p.add_argument("--dose_dir", required=True, help="original alpha=0.1 truth directory (read only)")
    p.add_argument("--data_dir", default="/u/yli103/oct2/data")
    p.add_argument("--out_dir", required=True, help="new directory under the B output root")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--core_patients", type=int, nargs="+", default=[807, 2085],
                   help="patients receiving the full decomposition (pre-registered core cells)")
    p.add_argument("--monitor_patients", type=int, nargs="*", default=None,
                   help="patients in the signal monitor; default = whole frozen panel")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.03, 0.3, 1.0],
                   help="report ||(Hbar+gamma I)d - alpha b|| for these damping values")
    p.add_argument("--hvp_batch", type=int, default=64)
    p.add_argument("--dropout_mc_reps", type=int, default=16,
                   help="dropout-mode MC replicates for g_0 and b_p (0 = eval only)")
    p.add_argument("--mc_seed", type=int, default=910000)
    p.add_argument("--require_cuda", action="store_true")
    args = p.parse_args()
    if args.hvp_batch < 1 or args.dropout_mc_reps < 0 or any(g < 0 for g in args.gammas):
        raise ValueError("hvp_batch >= 1, dropout_mc_reps >= 0, gammas >= 0")
    return args


# ---------------------------------------------------------------------------
# vector helpers (float64 on CPU for all reported algebra)
# ---------------------------------------------------------------------------

def d64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.float64)


def cos(a, b):
    return core.cosine(a, b)


def safe_div(a, b):
    return a / b if b else None


def signed_share(e: torch.Tensor, part: torch.Tensor):
    """<e, part> / ||e||^2 : share of the residual explained by `part` along e."""
    ee = float(torch.dot(e, e))
    return float(torch.dot(e, part)) / ee if ee else None


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def patient_gradients(score, model, X, y, raw_indices, device):
    """Per-image eval-mode CE gradients (n_p, d) on device, no weight decay (07's per_image_grads)."""
    return score.per_image_grads(model, X, y, list(map(int, raw_indices)), device)


def patient_eval_stats(model, X, y, raw_indices, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        xb = torch.as_tensor(X[raw_indices], dtype=dtype, device=device)
        yb = torch.as_tensor(y[raw_indices], dtype=torch.long, device=device)
        logp = torch.log_softmax(model(xb), dim=1)
        ce = -logp.gather(1, yb[:, None]).squeeze(1)
        prob = ce.neg().exp()
    return {"patient_mean_ce": float(ce.mean()), "patient_max_ce": float(ce.max()),
            "patient_mean_correct_prob": float(prob.mean()), "patient_min_correct_prob": float(prob.min())}


def mc_mean_gradient(model, X, y, indices, batch, wd, reps, seed):
    """Dropout-mode MC estimate of the mean gradient of (mean CE over indices + wd/2||theta||^2).

    Returns (mean vector float64 cpu, per-rep spread, se-of-mean estimate).
    Caller must not rely on RNG state afterwards; modes and RNG are restored.
    """
    if reps <= 0:
        return None, None, None
    total, sq = None, 0.0
    with core.preserve_rng_and_modes(model):
        for rep in range(reps):
            torch.manual_seed(seed + rep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + rep)
            _, g = core.objective_snapshot(model, X, y, indices, batch, wd, True, dropout=True)
            g = d64(g)
            total = g.clone() if total is None else total + g
            sq += float(torch.dot(g, g))
    mean = total / reps
    spread = math.sqrt(max(0.0, sq - reps * float(torch.dot(mean, mean))) / (reps-1)) if reps > 1 else 0.0
    return mean, spread, spread / math.sqrt(reps) if reps > 1 else None


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------

def load_checkpoint_model(legacy, pilot, path, X, y, expect_seed, expect_indices=None, expect_alpha=None,
                          expect_rng=None):
    payload = legacy.load_payload(path, pilot)
    md = payload["metadata"]
    legacy.assert_equal(md["seed"], expect_seed, f"seed mismatch in {path.name}")
    if expect_indices is not None:
        legacy.assert_equal(sorted(md["excluded_indices"]), sorted(expect_indices), f"patient mismatch in {path.name}")
        legacy.assert_equal(float(md["deletion_weight"]), expect_alpha, f"alpha mismatch in {path.name}")
    if expect_rng is not None:
        legacy.assert_equal(md["metrics"]["post_train_rng_sha256"], expect_rng, f"RNG fingerprint mismatch in {path.name}")
    model = legacy.model_from_payload(payload, pilot, X, y)
    return payload, model


def run_seed(args, context, legacy, score, pilot, seed, core_patients, monitor_patients):
    X, y, train = context["X"], context["y"], context["train"]
    N = len(train)
    a = context["pargs"].affected_shadow
    device = pilot.DEVICE
    WD = legacy.WD
    legacy.verify_orders(context, seed)

    base_name = f"shadow_{a}_baseline_seed{seed}.pt"
    base_payload, model0 = load_checkpoint_model(legacy, pilot, context["full"] / "checkpoints" / base_name, X, y, seed)
    other = legacy.load_payload(context["dose"] / "checkpoints" / base_name, pilot)
    if not core.exact_tree(base_payload["state_dict"], other["state_dict"]):
        raise RuntimeError("full and dose baseline parameters differ")
    rng = base_payload["metadata"]["metrics"]["post_train_rng_sha256"]
    legacy.assert_equal(other["metadata"]["metrics"]["post_train_rng_sha256"], rng, "Baseline RNG mismatch")
    theta0 = core.flat(model0).clone()

    t0 = time.time()
    g0_dev = score.shadow_loss_grad(model0, X, y, train, device, WD, args.hvp_batch)
    g0 = d64(g0_dev)
    hvp = score.make_hvp(model0, X, y, train, device, WD, 0.0, args.hvp_batch)
    LOG.info("seed=%d ||g_0||=%.4g (eval, %.1fs)", seed, core.norm(g0), time.time() - t0)
    g0_mc, g0_mc_spread, g0_mc_se = mc_mean_gradient(model0, X, y, train, args.hvp_batch, WD,
                                                     args.dropout_mc_reps, args.mc_seed)
    seed_record = {
        "seed": seed, "n_train": int(N), "theta0_norm": core.norm(theta0),
        "g0_eval_norm": core.norm(g0),
        "g0_dropout_mc_norm": core.norm(g0_mc) if g0_mc is not None else None,
        "g0_dropout_mc_rms_spread": g0_mc_spread, "g0_dropout_mc_mean_se_estimate": g0_mc_se,
        "cos_g0_eval_mc": cos(g0, g0_mc) if g0_mc is not None else None,
        "dropout_mc_reps": args.dropout_mc_reps,
    }

    # ---- signal monitor at theta_0 for the whole requested panel
    monitor_rows = []
    for patient in context["patients"]:
        pid = int(patient["patient_id"])
        if pid not in monitor_patients:
            continue
        idx = list(map(int, patient["raw_indices"]))
        G = patient_gradients(score, model0, X, y, idx, device)
        gsum = d64(G).sum(0)
        per_norm = torch.linalg.vector_norm(d64(G), dim=1)
        b_eval = gsum / N
        row = {"seed": seed, "patient_id": pid, "oct_class": int(patient["oct_class"]), "n_images": len(idx),
               "b_eval_norm": core.norm(b_eval), "sum_per_image_grad_norm_over_N": float(per_norm.sum()) / N,
               "per_image_grad_norm_min": float(per_norm.min()), "per_image_grad_norm_max": float(per_norm.max()),
               "cancellation_ratio": safe_div(core.norm(gsum), float(per_norm.sum())),
               "cos_b_g0": cos(b_eval, g0), **patient_eval_stats(model0, X, y, idx, device)}
        if args.dropout_mc_reps:
            # objective_snapshot averages over the patient images; rescale to (1/N) sum.
            m, spread, se = mc_mean_gradient(model0, X, y, np.asarray(idx), args.hvp_batch, 0.0,
                                             args.dropout_mc_reps, args.mc_seed + 7000 + pid)
            scale = len(idx) / N
            row.update(b_dropout_mc_norm=core.norm(m) * scale, b_dropout_mc_rms_spread=spread * scale,
                       b_dropout_mc_mean_se_estimate=(se * scale) if se is not None else None,
                       cos_b_eval_mc=cos(b_eval, m))
        monitor_rows.append(row)
        LOG.info("seed=%d patient=%d ||b||=%.3e cancel=%.3f meanCE=%.3e", seed, pid, row["b_eval_norm"],
                 row["cancellation_ratio"] or float("nan"), row["patient_mean_ce"])

    # ---- exact decomposition for the core cells
    rows = []
    for patient in context["patients"]:
        pid = int(patient["patient_id"])
        if pid not in core_patients:
            continue
        idx = list(map(int, patient["raw_indices"]))
        b0 = d64(patient_gradients(score, model0, X, y, idx, device)).sum(0) / N
        for condition, root, alpha in (("full", context["full"], 1.0), ("dose01", context["dose"], 0.1)):
            legacy.validate_run(root, patient, seed, context, alpha)
            path = root / "checkpoints" / f"shadow_{a}_seed{seed}_patient{pid}.pt"
            _, model_a = load_checkpoint_model(legacy, pilot, path, X, y, seed, idx, alpha, rng)
            theta_a = core.flat(model_a).clone()
            d_dev = (theta_a - theta0).to(device)
            d = d64(d_dev)
            t1 = time.time()
            gL0_at_a = d64(score.shadow_loss_grad(model_a, X, y, train, device, WD, args.hvp_batch))
            b_a = d64(patient_gradients(score, model_a, X, y, idx, device)).sum(0) / N
            g_a = gL0_at_a - alpha * b_a                      # gradient of L_alpha at theta_alpha
            Hd = d64(hvp(d_dev))                              # (H_CE + wd I) d
            R = gL0_at_a - g0 - Hd - alpha * (b_a - b0)       # Taylor remainder (exact definition)
            stat = g_a - g0                                   # endpoint-stationarity term
            e = Hd - alpha * b0                               # gamma = 0 static residual at the true d
            closure = core.norm(e - (stat - R))               # float rounding only; identity is exact
            if not all(torch.isfinite(v).all() for v in (d, g0, g_a, Hd, R, b0, b_a)):
                raise RuntimeError("Non-finite E0 vector")
            ab = alpha * b0
            row = {
                "seed": seed, "patient_id": pid, "oct_class": int(patient["oct_class"]), "condition": condition,
                "alpha": alpha, "n_images": len(idx),
                "d_norm": core.norm(d), "d_over_theta0": safe_div(core.norm(d), core.norm(theta0)),
                "g0_norm": core.norm(g0), "g_alpha_norm": core.norm(g_a), "gradL0_at_theta_alpha_norm": core.norm(gL0_at_a),
                "stationarity_term_norm": core.norm(stat), "taylor_remainder_norm": core.norm(R),
                "Hd_norm": core.norm(Hd), "alpha_b0_norm": core.norm(ab), "alpha_b_alpha_norm": core.norm(alpha * b_a),
                "rms_curvature_along_d": safe_div(core.norm(Hd), core.norm(d)),
                "rayleigh_quotient_d": safe_div(float(torch.dot(d, Hd)), float(torch.dot(d, d))),
                "residual_gamma0_norm": core.norm(e),
                "residual_gamma0_over_alpha_b0": safe_div(core.norm(e), core.norm(ab)),
                "residual_gamma0_over_d": safe_div(core.norm(e), core.norm(d)),
                "algebra_closure_abs": closure, "algebra_closure_rel": safe_div(closure, core.norm(e)),
                "share_stationarity": signed_share(e, stat), "share_remainder": signed_share(e, -R),
                "stationarity_over_Hd": safe_div(core.norm(stat), core.norm(Hd)),
                "remainder_over_Hd": safe_div(core.norm(R), core.norm(Hd)),
                "cos_stationarity_remainder": cos(stat, R), "cos_stationarity_residual": cos(stat, e),
                "cos_remainder_residual": cos(R, e), "cos_Hd_alpha_b0": cos(Hd, ab), "cos_d_b0": cos(d, b0),
                "cos_d_Hd": cos(d, Hd), "cos_b0_b_alpha": cos(b0, b_a),
                "b_change_norm_over_b0": safe_div(core.norm(b_a - b0), core.norm(b0)),
                "cos_g0_g_alpha": cos(g0, g_a), "elapsed_seconds": time.time() - t1,
            }
            for gamma in args.gammas:
                r = e + gamma * d
                row[f"residual_gamma{gamma:g}_over_alpha_b0"] = safe_div(core.norm(r), core.norm(ab))
                row[f"residual_gamma{gamma:g}_scaled"] = safe_div(
                    core.norm(r), core.norm(Hd) + gamma * core.norm(d) + core.norm(ab))
            rows.append(row)
            LOG.info("seed=%d patient=%d %s: ||g_a-g_0||=%.3g ||R||=%.3g ||Hd||=%.3g ||alpha b||=%.3g "
                     "share_stat=%.3f share_rem=%.3f", seed, pid, condition, row["stationarity_term_norm"],
                     row["taylor_remainder_norm"], row["Hd_norm"], row["alpha_b0_norm"],
                     row["share_stationarity"] or float("nan"), row["share_remainder"] or float("nan"))
    return seed_record, monitor_rows, rows


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Delta invocation")
    output = Path(args.out_dir).resolve()
    for source in (Path(args.full_dir).resolve(), Path(args.dose_dir).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("E0 output overlaps a truth directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    legacy, score, pilot = load_modules()
    source_hashes = {str(p): legacy.sha(p) for p in (
        HERE / "05_end_to_end_patient_loo_pilot.py", HERE / "07_cross_stage_score_ladder.py",
        HERE / "11_stage1_diagnostics.py", HERE / "stage1_diagnostic_core.py", Path(__file__),
        HERE.parents[1] / "src" / "models.py")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {"schema": SCHEMA, "status": "running", "args": vars(args), "python": platform.python_version(),
                "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(pilot.DEVICE),
                "git_commit": commit, "source_sha256": source_hashes,
                "objective": "eval-mode mean CE + wd/2||theta||^2; L_alpha = L_0 - alpha L_patient (fixed_mask, N fixed)",
                "identity": "(Hbar+gamma I)d - alpha b0 = (g_alpha - g_0) - R_alpha + gamma d",
                "interpretation": ("Norms of the two terms are not additive causal shares; signed shares along the "
                                   "residual direction sum to one. Dropout-MC quantities are labelled separately "
                                   "and are not mixed with the eval Hessian.")}
    legacy.write_json(output / "manifest.json", manifest)
    fingerprints = {}
    try:
        ctx_args = argparse.Namespace(full_dir=args.full_dir, dose_dir=args.dose_dir, mode="damping",
                                      data_dir=args.data_dir, seeds=args.seeds)
        v11 = legacy.module_from_path("e0_metadata_v11", HERE / "12_stage1_diagnostics_v11.py")
        context = v11.metadata_context(ctx_args, pilot)
        fingerprints = {str(p): legacy.sha(p) for p in legacy.input_paths(context, ctx_args)}
        legacy.write_json(output / "input_sha256.json", fingerprints)
        context = legacy.load_context(ctx_args, pilot)
        manifest["loaded_data_sha256"] = {
            key: hashlib.sha256(memoryview(context[key]).cast("B")).hexdigest() for key in ("X", "y")}
        panel_ids = [int(p["patient_id"]) for p in context["patients"]]
        core_patients = set(args.core_patients)
        monitor_patients = set(args.monitor_patients) if args.monitor_patients is not None else set(panel_ids)
        missing = (core_patients | monitor_patients) - set(panel_ids)
        if missing:
            raise RuntimeError(f"Requested patients not in the frozen panel: {sorted(missing)}")
        seed_records, monitor_rows, rows = [], [], []
        for seed in args.seeds:
            LOG.info("START E0 seed=%d", seed)
            rec, mon, dec = run_seed(args, context, legacy, score, pilot, seed, core_patients, monitor_patients)
            seed_records.append(rec)
            monitor_rows.extend(mon)
            rows.extend(dec)
            legacy.write_csv(output / "e0_rows.csv", rows)
            legacy.write_csv(output / "signal_monitor.csv", monitor_rows)
            legacy.write_json(output / "e0_rows.json", {"seeds": seed_records, "rows": rows, "monitor": monitor_rows})
        manifest.update(status="complete", elapsed_seconds=time.time() - started, n_rows=len(rows),
                        n_monitor_rows=len(monitor_rows),
                        core_patients=sorted(core_patients), monitor_patients=sorted(monitor_patients))
    except Exception as exc:  # noqa: BLE001
        manifest.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise
    finally:
        changed = [p for p, h in fingerprints.items() if not Path(p).is_file() or legacy.sha(Path(p)) != h]
        manifest.update(changed_inputs=changed, input_files_unchanged=not changed)
        if changed:
            manifest["status"] = "failed_inputs_changed"
        legacy.write_json(output / "manifest.json", manifest)
        if changed:
            raise RuntimeError("Original E0 input files changed")
    LOG.info("E0 complete: %d rows, %d monitor rows, %.1fs", len(rows), len(monitor_rows), time.time() - started)


if __name__ == "__main__":
    main()
