#!/usr/bin/env python3
"""Independent Stage 1 damping and convergence diagnostics for OCT v4.1.

Reads completed full/dose truth without changing their files or fitting any
attack model. Every invocation creates a fresh, separate output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from formal_analysis import clean_json
import stage1_diagnostic_core as core

LOG = logging.getLogger("stage1_diagnostics")
WD = 1e-5


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row if not isinstance(row[k], (dict, list))))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_json(rows))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_equal(a, b, message):
    if a != b:
        raise RuntimeError(message)


def panel_key(panel):
    return sorted((int(p["patient_id"]), int(p["oct_class"]),
                   tuple(sorted(map(int, p["raw_indices"])))) for p in panel["patients"])


def validate_sources(full, dose, pilot, seeds):
    """Metadata checks before GPU work. Loading checkpoints adds bitwise checks."""
    config = read_json(full / "experiment_config.json")
    summary = read_json(full / "experiment_summary.json")
    panel = read_json(full / "selected_patients.json")
    pilot.require_training_numerics(config, str(full))
    args = config["args"]
    assert_equal(summary["status"], "complete", "full truth is incomplete")
    assert_equal(args["deletion_mode"], "fixed_mask", "Expected fixed_mask")
    assert_equal(float(args["deletion_weight"]), 1., "Expected full alpha=1")
    assert_equal(args.get("removal_epochs"), None, "Windowed truth is not supported")
    assert_equal(args["deterministic"], True, "Strict deterministic training is required")
    assert_equal(args["window_membership"], "value_only", "Expected value-only dose design")
    assert_equal(config["oct_config"]["target_l2"], WD, "Stage 1 weight decay differs")
    assert_equal(config["oct_config"]["optimizer_type"], "adam", "Expected original Adam training")
    if not set(seeds).issubset(set(args["seeds"])):
        raise RuntimeError("Diagnostic seeds are not in the frozen training panel")
    split = full / "splits" / "fresh_patient_split.json"
    assert_equal(sha(split)[:16], summary["split_sha256"], "full split hash mismatch")
    assert_equal(panel["split_sha256"], summary["split_sha256"], "full panel split mismatch")
    if len(panel_key(panel)) != len(set(p[0] for p in panel_key(panel))):
        raise RuntimeError("Duplicate patients")
    if dose is not None:
        dc = read_json(dose / "experiment_config.json")
        ds = read_json(dose / "experiment_summary.json")
        dp = read_json(dose / "selected_patients.json")
        pilot.require_training_numerics(dc, str(dose))
        assert_equal(ds["status"], "complete", "dose truth is incomplete")
        assert_equal(ds["split_sha256"], summary["split_sha256"], "dose split mismatch")
        assert_equal(sha(dose / "splits" / "fresh_patient_split.json"), sha(split), "split bytes differ")
        assert_equal(panel_key(dp), panel_key(panel), "dose patient panel differs")
        for key in ("seeds", "attack_seeds", "affected_shadow", "split_seed", "target_seed",
                    "fixed_shadow_seed", "shadow_epochs", "shadow_lr", "shadow_batch_size",
                    "attack_epochs", "deletion_mode", "window_membership", "deterministic",
                    "n_total_samples", "target_data_size", "shadow_data_size", "n_shadow"):
            assert_equal(dc["args"].get(key), args.get(key), f"dose config differs: {key}")
        assert_equal(dc["args"].get("removal_epochs"), None, "Windowed dose is not supported")
        assert_equal(float(dc["args"]["deletion_weight"]), .1, "Expected dose01 alpha=0.1")
    return config, summary, panel


def load_payload(path, pilot):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pilot.require_training_numerics(payload.get("metadata", {}), str(path))
    return payload


def model_from_payload(payload, pilot, X, y):
    model = pilot.build_model("cnn", X.shape[1], 128, int(np.max(y)) + 1)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_context(args, pilot):
    full = Path(args.full_dir).resolve()
    dose = Path(args.dose_dir).resolve() if args.mode == "damping" else None
    config, summary, panel = validate_sources(full, dose, pilot, args.seeds)
    pargs = argparse.Namespace(**config["args"])
    pargs.data_dir = args.data_dir
    cfg = pilot.build_config(pargs)
    pilot.seed_everything(pargs.target_seed, True)
    X, y, groups = pilot.load_dataset(cfg)
    splits = read_json(full / "splits" / "fresh_patient_split.json")
    pilot.validate_patient_split(splits, groups)
    train = np.asarray(splits["shadow_models"][pargs.affected_shadow]["train_idx"], dtype=np.int64)
    for patient in panel["patients"]:
        actual = train[groups[train] == int(patient["patient_id"])]
        assert_equal(sorted(actual.tolist()), sorted(patient["raw_indices"]), "Patient raw indices changed")
        classes, counts = np.unique(y[actual], return_counts=True)
        assert_equal(int(classes[counts.argmax()]), patient["oct_class"], "Patient OCT class changed")
    return {"full": full, "dose": dose, "config": config, "summary": summary,
            "pargs": pargs, "patients": panel["patients"], "X": X, "y": y, "train": train}


def input_paths(context, args):
    paths = []
    a = context["pargs"].affected_shadow
    for root in (context["full"], context["dose"]):
        if root is None:
            continue
        paths.extend(root / name for name in ("experiment_config.json", "experiment_summary.json",
                     "selected_patients.json", "splits/fresh_patient_split.json"))
        for seed in args.seeds:
            paths.append(root / "checkpoints" / f"shadow_{a}_baseline_seed{seed}.pt")
            paths.append(root / f"stage1_order_seed{seed}.npz")
            paths.append(root / f"baseline_seed{seed}.json")
            if args.mode == "damping":
                for patient in context["patients"]:
                    pid = patient["patient_id"]
                    paths.extend([root / "checkpoints" / f"shadow_{a}_seed{seed}_patient{pid}.pt",
                                  root / "runs" / f"seed{seed}_patient{pid}.json"])
            else:
                for epoch in sorted(set(context["pargs"].save_epoch_checkpoints)):
                    paths.append(root / "checkpoints" / f"baseline_seed{seed}_epochs" / f"epoch{epoch:03d}.pt")
    return paths


def verify_orders(context, seed):
    p = context["pargs"]
    orders = np.load(context["full"] / f"stage1_order_seed{seed}.npz")["raw_index_order"]
    expected = (p.shadow_epochs, len(context["train"]))
    assert_equal(orders.shape, expected, "Recorded epoch order has wrong shape")
    target = np.sort(context["train"])
    if not all(np.array_equal(np.sort(row), target) for row in orders):
        raise RuntimeError("Recorded epoch order is not a permutation of the full training set")
    if context["dose"] is not None:
        other = np.load(context["dose"] / f"stage1_order_seed{seed}.npz")["raw_index_order"]
        if not np.array_equal(orders, other):
            raise RuntimeError("full and dose epoch orders differ")
    return orders


def validate_run(root, patient, seed, context, alpha):
    run = read_json(root / "runs" / f"seed{seed}_patient{patient['patient_id']}.json")
    assert_equal(run["seed"], seed, "Run seed mismatch")
    assert_equal(run["patient"], patient, "Run patient mismatch")
    assert_equal(run["split_sha256"], context["summary"]["split_sha256"], "Run split mismatch")
    assert_equal(run["affected_shadow"], context["pargs"].affected_shadow, "Run shadow mismatch")
    assert_equal(float(run["exposure"]["deletion_weight"]), alpha, "Run deletion weight mismatch")
    cls = str(patient["oct_class"])
    return run["conditions"]["J00"]["per_class"][cls]


def damping_seed(args, context, pilot, score, seed, output):
    X, y, train = context["X"], context["y"], context["train"]
    a = context["pargs"].affected_shadow
    verify_orders(context, seed)
    filename = f"shadow_{a}_baseline_seed{seed}.pt"
    base = load_payload(context["full"] / "checkpoints" / filename, pilot)
    other = load_payload(context["dose"] / "checkpoints" / filename, pilot)
    if not core.exact_tree(base["state_dict"], other["state_dict"]):
        raise RuntimeError("full and dose baseline parameters differ")
    assert_equal(base["metadata"]["metrics"]["post_train_rng_sha256"],
                 other["metadata"]["metrics"]["post_train_rng_sha256"], "Baseline RNG mismatch")
    model = model_from_payload(base, pilot, X, y)
    theta = core.flat(model).clone()
    grad = score.shadow_loss_grad(model, X, y, train, pilot.DEVICE, WD, args.hvp_batch)
    H = score.make_hvp(model, X, y, train, pilot.DEVICE, WD, 0., args.hvp_batch)
    probes = []
    for probe_seed in args.probe_seeds:
        LOG.info("seed=%s Lanczos probe=%s", seed, probe_seed)
        probes.append(core.lanczos_probe(H, theta.numel(), args.lanczos_iters,
                                       probe_seed, theta.device, theta.dtype))
    spectra = {"seed": seed, "operator": "Hbar = eval-mode mean CE Hessian + weight_decay I",
               "weight_decay": WD, "baseline_gradient_norm": core.norm(grad), "probes": probes}
    minimum = min(p["endpoints"]["min"]["value"] for p in probes)
    maximum = max(p["endpoints"]["max"]["value"] for p in probes)
    spectra.update(min_ritz_estimate=minimum, max_ritz_estimate=maximum,
                   ce_min_ritz_estimate=minimum - WD, ce_max_ritz_estimate=maximum - WD)
    write_json(output / f"spectrum_seed{seed}.json", spectra)
    rows, raw_rows, dose_rows = [], [], []
    for patient in context["patients"]:
        pid = patient["patient_id"]
        b = score.per_image_grads(model, X, y, patient["raw_indices"], pilot.DEVICE).sum(0) / len(train)
        truth = {}
        j00 = []
        for condition, root, alpha in (("full", context["full"], 1.), ("dose01", context["dose"], .1)):
            j00.append(validate_run(root, patient, seed, context, alpha))
            payload = load_payload(root / "checkpoints" / f"shadow_{a}_seed{seed}_patient{pid}.pt", pilot)
            md = payload["metadata"]
            assert_equal(md["seed"], seed, "Checkpoint seed mismatch")
            assert_equal(sorted(md["excluded_indices"]), sorted(patient["raw_indices"]), "Checkpoint patient mismatch")
            assert_equal(float(md["deletion_weight"]), alpha, "Checkpoint alpha mismatch")
            assert_equal(md["metrics"]["post_train_rng_sha256"],
                         base["metadata"]["metrics"]["post_train_rng_sha256"], "Truth RNG mismatch")
            removed = model_from_payload(payload, pilot, X, y)
            truth[condition] = core.flat(removed) - theta
            del removed, payload
        for key in ("attack_seeds", "per_rep_cross_entropy", "cross_entropy"):
            assert_equal(j00[0][key], j00[1][key], f"full/dose J00 mismatch: {key}")
        pairing = {"seed": seed, "patient_id": pid, "oct_class": patient["oct_class"],
                   **core.geometry(.1 * truth["full"], truth["dose01"])}
        dose_rows.append(pairing)
        for gamma in args.damping_grid:
            screen = core.spectrum_screen(probes, gamma, args.ritz_tol)
            A = lambda z: H(z) + gamma * z
            solved = score.conjugate_gradient(A, b, args.cg_iters, args.cg_tol)
            x = solved.pop("x")
            qualified = core.solver_qualified(solved, screen, args.cg_fail_tol)
            bnorm = core.norm(b)
            Hx = H(x)
            common = {"seed": seed, "patient_id": pid, "oct_class": patient["oct_class"],
                      "n_images": patient["n_images"], "gamma": gamma, "qualified": qualified,
                      "rhs_norm": bnorm, "baseline_gradient_norm": core.norm(grad),
                      "undamped_min_ritz_estimate": minimum, "undamped_max_ritz_estimate": maximum,
                      "gamma_over_lambda_max_estimate": gamma / maximum if maximum > 0 else None,
                      "gamma_over_negative_edge_estimate": gamma / abs(minimum) if minimum < 0 else None,
                      "cg_iters": solved["iters"], "cg_true_rel_residual": solved["final_rel_residual"],
                      "nonpositive_curvature": solved["nonpositive_curvature"],
                      "positive_ritz_screen": screen["positive_ritz_screen"],
                      "ritz_residual_screen": screen["ritz_residual_screen"],
                      "min_damped_ritz_estimate": screen["min_damped_ritz_estimate"]}
            extras = {"cosine_with_rhs": core.cosine(x, b),
                      "isotropic_relative_difference": core.norm(x - b / gamma) / core.norm(x) if core.norm(x) else None,
                      "damping_term_over_rhs": gamma * core.norm(x) / bnorm if bnorm else None,
                      "curvature_term_over_rhs": core.norm(Hx) / bnorm if bnorm else None,
                      "undamped_equation_rel_residual": core.norm(Hx - b) / bnorm if bnorm else None}
            for condition, alpha in (("full", 1.), ("dose01", .1)):
                measured = {**core.geometry(alpha * x, truth[condition]), **extras}
                raw_rows.append({**common, "condition": condition, "alpha": alpha,
                                 "raw_metrics_do_not_use_if_unqualified": measured,
                                 "cg_diagnostics": solved, "spectral_screen": screen})
                rows.append({**common, "condition": condition, "alpha": alpha,
                             **{k: v if qualified else None for k, v in measured.items()}})
            LOG.info("seed=%s patient=%s gamma=%.3g qualified=%s residual=%s", seed, pid, gamma,
                     qualified, solved["final_rel_residual"])
            write_json(output / f"damping_seed{seed}.json", {"rows": rows, "raw_rows": raw_rows})
            write_csv(output / f"damping_seed{seed}.csv", rows)
    write_csv(output / f"true_dose_scaling_seed{seed}.csv", dose_rows)
    return rows, dose_rows


def convergence_seed(args, context, pilot, seed, output):
    X, y, train = context["X"], context["y"], context["train"]
    p = context["pargs"]
    orders = verify_orders(context, seed)
    orders_expected = pilot.make_epoch_orders(train, p.shadow_epochs, seed + 700000)
    if not np.array_equal(orders, orders_expected):
        raise RuntimeError("Recorded orders disagree with original seed derivation")
    checks, rows = [], []
    epochs = set(map(int, p.save_epoch_checkpoints))
    if not epochs or p.shadow_epochs not in epochs:
        raise RuntimeError("Need original epoch checkpoints including the final epoch")

    def check_epoch(epoch, model, optimizer):
        if epoch not in epochs:
            return
        path = context["full"] / "checkpoints" / f"baseline_seed{seed}_epochs" / f"epoch{epoch:03d}.pt"
        payload = load_payload(path, pilot)
        md = payload["metadata"]
        current_rng = core.snapshot_rng()
        assert_equal(md["epoch_order_sha256"], hashlib.sha256(np.ascontiguousarray(orders).tobytes()).hexdigest()[:16],
                     "Checkpoint order fingerprint mismatch")
        row = {"epoch": epoch, "parameters_equal": core.exact_tree(model.state_dict(), payload["state_dict"]),
               "optimizer_equal": core.exact_tree(optimizer.state_dict(), md["optimizer_state"]),
               "cpu_rng_equal": core.exact_tree(current_rng["torch_cpu"], md["rng_states"]["torch_cpu"]),
               "cuda_rng_equal": core.exact_tree(current_rng["torch_cuda"], md["rng_states"].get("torch_cuda", []))}
        row["passed"] = all(row[k] for k in ("parameters_equal", "optimizer_equal", "cpu_rng_equal", "cuda_rng_equal"))
        checks.append(row)
        write_json(output / f"replay_checks_seed{seed}.json", checks)
        if not row["passed"]:
            raise RuntimeError(f"Seed {seed} epoch {epoch}: replay mismatch; do not interpret this as the original curve")

    def observe(model, row):
        epoch = row["epoch"]
        with_grad = epoch % args.gradient_every == 0 or epoch in epochs
        mc = args.dropout_mc_reps if epoch in ({0} | epochs) else 0
        info = core.observe_checkpoint(model, X, y, train, args.hvp_batch, WD, with_grad,
                                       mc_reps=mc, mc_seed=910000 + seed * 100)
        rows.append({"seed": seed, **row, **info})
        write_csv(output / f"training_curve_seed{seed}.csv", rows)
        LOG.info("seed=%s epoch=%s online_CE=%s eval_objective=%.6g grad=%s", seed, epoch,
                 row["online_train_ce"], info["eval_objective"], info["eval_grad_norm"])

    model, optimizer, rng_fp = core.replay_baseline(pilot, X, y, orders, seed, p.shadow_lr,
                                                   p.shadow_batch_size, WD, observe, check_epoch)
    payload = load_payload(context["full"] / "checkpoints" / f"shadow_{p.affected_shadow}_baseline_seed{seed}.pt", pilot)
    matched = core.exact_tree(model.state_dict(), payload["state_dict"])
    matched_rng = rng_fp == payload["metadata"]["metrics"]["post_train_rng_sha256"]
    final = {"seed": seed, "parameters_equal_final": matched, "rng_equal_final": matched_rng,
             "post_train_rng_sha256": rng_fp, "passed": matched and matched_rng and all(c["passed"] for c in checks)}
    write_json(output / f"replay_final_seed{seed}.json", final)
    if not final["passed"]:
        raise RuntimeError("Final replay differs from the original baseline")
    return rows, final


def make_plots(mode, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if mode == "convergence":
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        for seed in sorted({r["seed"] for r in rows}):
            selected = [r for r in rows if r["seed"] == seed]
            for ax, field in zip(axes, ("online_train_ce", "eval_objective", "eval_grad_norm")):
                use = [r for r in selected if r.get(field) is not None]
                ax.plot([r["epoch"] for r in use], [r[field] for r in use], label=str(seed))
                ax.set_xlabel("Epoch")
                ax.set_title(field.replace("_", " "))
                ax.grid(alpha=.2)
        axes[0].legend(title="Stage 1 seed", fontsize=8)
        fig.suptitle("Baseline replay. Gradient is for the eval CE + L2 objective.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
        grid = {r["gamma"] for r in rows}
        cells = {}
        for row in rows:
            cells.setdefault((row["seed"], row["patient_id"], row["condition"]), []).append(row)
        common = {key for key, rs in cells.items()
                  if {r["gamma"] for r in rs} == grid and all(r["qualified"] for r in rs)}
        for condition in ("full", "dose01"):
            use = [r for r in rows if r["condition"] == condition
                   and (r["seed"], r["patient_id"], condition) in common]
            for ax, field in zip(axes, ("cosine", "norm_ratio")):
                gammas, medians = [], []
                for gamma in sorted({r["gamma"] for r in use}):
                    values = [r[field] for r in use if r["gamma"] == gamma and r[field] is not None]
                    if values:
                        gammas.append(gamma)
                        medians.append(float(np.median(values)))
                ax.plot(gammas, medians, marker="o", label=condition)
                ax.set_xlabel("Stage 1 damping")
                ax.set_title("Common-cell median " + field)
                ax.grid(alpha=.2)
        axes[0].legend()
        if any(r.get("norm_ratio", 0) and r["norm_ratio"] > 0 for r in rows):
            axes[1].set_yscale("log")
        fig.suptitle("Descriptive medians on cells qualified at every tested damping value.")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{mode}_overview.{suffix}", dpi=180)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["damping", "convergence"], required=True)
    ap.add_argument("--full_dir", default="results/cross_stage_calibration_v4_1/shadow3_full")
    ap.add_argument("--dose_dir", default="results/cross_stage_calibration_v4_1/shadow3_dose01")
    ap.add_argument("--data_dir", default="/u/yli103/oct2/data")
    ap.add_argument("--out_dir", required=True, help="Must be new and outside both source directories")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--damping_grid", type=float, nargs="+", default=[.7, 1., 2.])
    ap.add_argument("--hvp_batch", type=int, default=64)
    ap.add_argument("--cg_iters", type=int, default=150)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--cg_fail_tol", type=float, default=1e-3)
    ap.add_argument("--lanczos_iters", type=int, default=50)
    ap.add_argument("--probe_seeds", type=int, nargs="+", default=[1701, 1702])
    ap.add_argument("--ritz_tol", type=float, default=.005)
    ap.add_argument("--gradient_every", type=int, default=5)
    ap.add_argument("--dropout_mc_reps", type=int, default=4)
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args()
    if any(x < 1 for x in (args.hvp_batch, args.cg_iters, args.lanczos_iters, args.gradient_every)):
        ap.error("Batch size and iteration counts must be positive")
    if any(not math.isfinite(g) or g <= 0 for g in args.damping_grid):
        ap.error("Damping candidates must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.cg_tol, args.cg_fail_tol, args.ritz_tol)):
        ap.error("Tolerances must be finite and positive")
    if args.cg_tol > args.cg_fail_tol or args.dropout_mc_reps < 0:
        ap.error("Require cg_tol <= cg_fail_tol and nonnegative MC count")
    for values in (args.seeds, args.damping_grid, args.probe_seeds):
        if len(set(values)) != len(values):
            ap.error("Duplicate seed or damping value")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Delta invocation")
    output = Path(args.out_dir).resolve()
    for source in (Path(args.full_dir).resolve(), Path(args.dose_dir).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("Diagnostic output overlaps a truth directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    score = module_from_path("stage1_audit_score07", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    source_hashes = {str(path): sha(path) for path in (
        HERE / "05_end_to_end_patient_loo_pilot.py", HERE / "07_cross_stage_score_ladder.py",
        HERE / "stage1_diagnostic_core.py", Path(__file__), HERE.parents[1] / "src" / "models.py")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {"schema": "stage1_diagnostics_v1", "status": "running", "args": vars(args),
                "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(pilot.DEVICE), "git_commit": commit, "source_sha256": source_hashes,
                "objective": "Stage 1 eval CE + L2. This is not the dropout training expectation.",
                "selection": "Fixed damping grid. No best-gamma selection or attack training.",
                "spectrum_warning": "Positive finite Ritz estimates are not an SPD certificate."}
    write_json(output / "manifest.json", manifest)
    try:
        context = load_context(args, pilot)
        fingerprints = {str(path): sha(path) for path in input_paths(context, args)}
        write_json(output / "input_sha256.json", fingerprints)
        rows, extra = [], []
        for seed in args.seeds:
            LOG.info("START %s seed=%s", args.mode, seed)
            if args.mode == "damping":
                result, detail = damping_seed(args, context, pilot, score, seed, output)
                rows.extend(result); extra.extend(detail)
            else:
                result, detail = convergence_seed(args, context, pilot, seed, output)
                rows.extend(result); extra.append(detail)
            write_csv(output / "all_rows.csv", rows)
        unchanged = all(sha(Path(path)) == value for path, value in fingerprints.items())
        if not unchanged:
            raise RuntimeError("An input artifact changed during diagnostics")
        make_plots(args.mode, rows, output)
        manifest.update(status="complete", elapsed_seconds=time.time() - started,
                        input_files_unchanged=True, n_rows=len(rows), per_seed_extra=extra)
        if args.mode == "damping":
            # Fair common-row summaries, never select a gamma from these metrics.
            cells = {}
            for r in rows:
                cells.setdefault((r["seed"], r["patient_id"], r["condition"]), []).append(r)
            common = {k for k, rs in cells.items() if len(rs) == len(args.damping_grid) and all(r["qualified"] for r in rs)}
            summaries = []
            for condition in ("full", "dose01"):
                for gamma in args.damping_grid:
                    subset = [r for r in rows if r["condition"] == condition and r["gamma"] == gamma]
                    shared = [r for r in subset if (r["seed"], r["patient_id"], condition) in common]
                    item = {"condition": condition, "gamma": gamma, "n_total": len(subset),
                            "n_qualified": sum(r["qualified"] for r in subset), "n_common": len(shared)}
                    for key in ("cosine", "norm_ratio", "relative_error", "cosine_with_rhs", "damping_term_over_rhs"):
                        values = [r[key] for r in shared if r.get(key) is not None]
                        item["common_median_" + key] = float(np.median(values)) if values else None
                    summaries.append(item)
            manifest["damping_summary"] = summaries
            write_csv(output / "damping_summary.csv", summaries)
        write_json(output / "manifest.json", manifest)
        LOG.info("COMPLETE -> %s", output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time() - started)
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
