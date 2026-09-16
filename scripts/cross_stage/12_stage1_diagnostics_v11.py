#!/usr/bin/env python3
"""Stage 1 diagnostics v1.1: checkpoint preflight, convergence, shared Krylov audit."""
from __future__ import annotations
import argparse
import csv
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
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core
import stage1_krylov_audit as audit

spec = importlib.util.spec_from_file_location("stage1_v11_legacy", HERE / "11_stage1_diagnostics.py")
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
LOG = logging.getLogger("stage1_v11")
WD = legacy.WD


def parameter_vector(payload):
    state = payload["state_dict"]
    if not all(torch.is_floating_point(t) for t in state.values()):
        raise RuntimeError("Expected the buffer-free v4.1 SmallCNN state dictionary")
    return torch.cat([value.detach().cpu().reshape(-1) for value in state.values()])


def checkpoint_pair(context, pilot, seed, patient=None):
    shadow = context["pargs"].affected_shadow
    baseline = []
    for root in (context["full"], context["dose"]):
        payload = legacy.load_payload(root / "checkpoints" / f"shadow_{shadow}_baseline_seed{seed}.pt", pilot)
        legacy.assert_equal(payload["metadata"]["seed"], seed, "Baseline seed mismatch")
        baseline.append(payload)
    if not core.exact_tree(baseline[0]["state_dict"], baseline[1]["state_dict"]):
        raise RuntimeError("full/dose baseline parameters differ")
    rng = baseline[0]["metadata"]["metrics"]["post_train_rng_sha256"]
    legacy.assert_equal(rng, baseline[1]["metadata"]["metrics"]["post_train_rng_sha256"], "Baseline RNG mismatch")
    theta = parameter_vector(baseline[0])
    if patient is None:
        return baseline[0], theta
    truth, entries = {}, []
    for condition, root, alpha in (("full", context["full"], 1.), ("dose01", context["dose"], .1)):
        entries.append(legacy.validate_run(root, patient, seed, context, alpha))
        pid = patient["patient_id"]
        payload = legacy.load_payload(root / "checkpoints" / f"shadow_{shadow}_seed{seed}_patient{pid}.pt", pilot)
        md = payload["metadata"]
        legacy.assert_equal(md["seed"], seed, "Truth checkpoint seed mismatch")
        legacy.assert_equal(sorted(md["excluded_indices"]), sorted(patient["raw_indices"]), "Truth patient mismatch")
        legacy.assert_equal(float(md["deletion_weight"]), alpha, "Truth alpha mismatch")
        legacy.assert_equal(md["metrics"]["post_train_rng_sha256"], rng, "Truth RNG mismatch")
        legacy.assert_equal(list(payload["state_dict"]), list(baseline[0]["state_dict"]), "State layout mismatch")
        truth[condition] = parameter_vector(payload) - theta
    for key in ("attack_seeds", "per_rep_cross_entropy", "cross_entropy"):
        legacy.assert_equal(entries[0][key], entries[1][key], "full/dose J00 mismatch: " + key)
    return theta, truth


def metadata_context(args, pilot):
    full, dose = Path(args.full_dir).resolve(), Path(args.dose_dir).resolve()
    config, summary, panel = legacy.validate_sources(full, dose, pilot, args.seeds)
    pargs = argparse.Namespace(**config["args"])
    splits = legacy.read_json(full / "splits/fresh_patient_split.json")
    train = np.asarray(splits["shadow_models"][pargs.affected_shadow]["train_idx"], dtype=np.int64)
    return {"full": full, "dose": dose, "config": config, "summary": summary,
            "patients": panel["patients"], "pargs": pargs, "train": train, "splits": splits}


def checkpoint_preflight(context, args, pilot, output):
    bases = {}
    for seed in args.seeds:
        legacy.verify_orders(context, seed)
        _, bases[seed] = checkpoint_pair(context, pilot, seed)
    rows, gaps = [], []
    for seed in args.seeds:
        distances = [core.norm(bases[seed] - bases[s]) for s in args.seeds if s != seed]
        median_gap = float(np.median(distances)) if distances else None
        for other in args.seeds:
            if seed < other:
                gaps.append({"seed1": seed, "seed2": other, "parameter_distance": core.norm(bases[seed] - bases[other])})
        for patient in context["patients"]:
            theta, truth = checkpoint_pair(context, pilot, seed, patient)
            full, dose = truth["full"], truth["dose01"]
            f, d, t = core.norm(full), core.norm(dose), core.norm(theta)
            rows.append({"seed": seed, "patient_id": patient["patient_id"], "oct_class": patient["oct_class"],
                         "full_norm": f, "dose01_norm": d, "baseline_norm": t,
                         "dose01_over_point1_full": d / (.1 * f) if f else None,
                         "cosine_dose01_full": core.cosine(dose, full),
                         "dose01_relative_linear_error": core.norm(dose - .1 * full) / (.1 * f) if f else None,
                         "full_over_theta": f / t if t else None,
                         "median_baseline_seed_distance": median_gap,
                         "full_over_median_seed_distance": f / median_gap if median_gap else None})
    legacy.write_csv(output / "checkpoint_ratios.csv", rows)
    legacy.write_csv(output / "baseline_seed_distances.csv", gaps)
    return rows, {"interpretation": "Scale references only. Ratios do not decompose response and chaos.",
                  "n_patients": len(context["patients"]), "no_hvp_or_training": True}


def source_files():
    return [HERE / name for name in ("05_end_to_end_patient_loo_pilot.py", "07_cross_stage_score_ladder.py",
            "11_stage1_diagnostics.py", "stage1_diagnostic_core.py", "stage1_krylov_audit.py",
            "12_stage1_diagnostics_v11.py")] + [HERE.parents[1] / "src" / "models.py"]


def input_files(context, args):
    mode = "damping" if args.mode == "preflight" else args.mode
    files = legacy.input_paths(context, argparse.Namespace(mode=mode, seeds=args.seeds))
    if args.mode == "damping":
        checkpoint = context["full"] / "checkpoints"
        files.append(checkpoint / "target_fixed.pt")
        files.extend(checkpoint / f"shadow_{s}_fixed.pt" for s in range(context["pargs"].n_shadow)
                     if s != context["pargs"].affected_shadow)
        for root in (context["full"], context["dose"]):
            reference = root / "score_ladder_A0.2_S1/ladder_rows.csv"
            if reference.exists():
                files.append(reference)
    return list(dict.fromkeys(files))


def h_for_seed(context, args, pilot, score, baseline, seed, output):
    """Reproduce Stage 2 once, check paired J00, and save all attack-seed h vectors."""
    X, y, p = context["X"], context["y"], context["pargs"]
    checkpoints = context["full"] / "checkpoints"
    classes = sorted({int(pat["oct_class"]) for pat in context["patients"]})
    models = [baseline if s == p.affected_shadow else
              pilot.load_model(checkpoints / f"shadow_{s}_fixed.pt", X.shape[1], 128, int(y.max()) + 1)
              for s in range(p.n_shadow)]
    target = pilot.load_model(checkpoints / "target_fixed.pt", X.shape[1], 128, int(y.max()) + 1)
    interface = pilot.make_interface(models, context["splits"], X, y)
    queries = pilot.make_target_queries(target, context["splits"], X, y)
    affected = interface["shadow_id"] == p.affected_shadow
    result, metadata = {}, {}
    for cls in classes:
        train_mask, query_mask = interface["classes"] == cls, queries["classes"] == cls
        affected_in_class = affected[train_mask]
        raw = interface["raw_index"][train_mask][affected_in_class]
        seeds = pilot.attack_rep_seeds(seed, cls, p)
        expected = []
        for patient in context["patients"]:
            if patient["oct_class"] == cls:
                for root, alpha in ((context["full"], 1.), (context["dose"], .1)):
                    row = legacy.validate_run(root, patient, seed, context, alpha)
                    legacy.assert_equal(row["attack_seeds"], seeds, "h reconstruction attack seeds differ")
                    expected.append(np.asarray(row["per_rep_cross_entropy"], dtype=float))
        hs, checks = [], []
        for index, attack_seed in enumerate(seeds):
            attack = pilot.train_attack_model_deterministic(
                interface["x"][train_mask], interface["membership"][train_mask], attack_seed,
                p.attack_epochs, p.attack_lr, p.attack_batch_size, n_hidden=64, deterministic=p.deterministic)
            metric, _, _ = pilot.evaluate_attack_queries(attack, queries["x"][query_mask], queries["membership"][query_mask])
            error = max(abs(metric["cross_entropy"] - e[index]) for e in expected)
            if not math.isfinite(error) or error > args.j00_tol:
                raise RuntimeError(f"J00 reproduction failed seed={seed} class={cls} attack_seed={attack_seed}: {error}")
            implicit = score.attack_implicit_v(attack, interface["x"][train_mask], interface["membership"][train_mask],
                       queries["x"][query_mask], queries["membership"][query_mask], args.damping_attack, pilot.DEVICE)
            if not implicit["solve_success"]:
                raise RuntimeError("Attack solve failed while reconstructing h")
            h = score.shadow_h(baseline, X[raw], implicit["v"][affected_in_class], pilot.DEVICE)
            hs.append(h)
            checks.append({"attack_seed": attack_seed, "j00_max_abs_diff": error,
                           **{k: v for k, v in implicit.items() if k not in ("v", "u")}})
        matrix = torch.stack(hs)
        result[cls] = {"matrix": matrix, "attack_seeds": seeds,
                       "attack_qualified": all(c["attack_solve_reliable"] for c in checks)}
        metadata[str(cls)] = {"checks": checks, "h_norms": [core.norm(h) for h in hs]}
    identity = {"schema": "stage1_h_cache_v11", "seed": seed, "damping_attack": args.damping_attack,
                "split_sha256": context["summary"]["split_sha256"],
                "baseline_sha256": legacy.sha(checkpoints / f"shadow_{p.affected_shadow}_baseline_seed{seed}.pt"),
                "source_sha256": {str(path): legacy.sha(path) for path in source_files()}, "classes": metadata}
    torch.save({"metadata": identity, "h_by_class": {cls: value["matrix"].detach().cpu() for cls, value in result.items()}},
               output / f"h_seed{seed}.pt")
    legacy.write_json(output / f"h_checks_seed{seed}.json", identity)
    return result


def compare_existing_l2(context, args, seed, patient, condition, projection):
    if args.damping_attack != .2:
        return None
    path = context["full" if condition == "full" else "dose"] / "score_ladder_A0.2_S1/ladder_rows.csv"
    if not path.exists():
        return None
    with path.open() as f:
        matches = [r for r in csv.DictReader(f) if int(r["seed"]) == seed and int(r["patient_id"]) == patient["patient_id"]]
    if len(matches) != 1 or not matches[0].get("L2_lin_value"):
        raise RuntimeError("Existing ladder row is incomplete or duplicated")
    reference = float(matches[0]["L2_lin_value"])
    if not np.isclose(projection, reference, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Reconstructed h dot true-dtheta differs from existing L2: {projection} vs {reference}")
    return abs(projection - reference)


def damping_seed(args, context, pilot, score, seed, output):
    legacy.verify_orders(context, seed)
    payload, theta_cpu = checkpoint_pair(context, pilot, seed)
    X, y, train = context["X"], context["y"], context["train"]
    model = legacy.model_from_payload(payload, pilot, X, y)
    h = h_for_seed(context, args, pilot, score, model, seed, output)
    H = audit.CountedOperator(score.make_hvp(model, X, y, train, pilot.DEVICE, WD, 0., args.hvp_batch))
    theta = core.flat(model)
    probes = [core.lanczos_probe(H, theta.numel(), args.lanczos_iters, s, theta.device, theta.dtype)
              for s in args.probe_seeds]
    grad = score.shadow_loss_grad(model, X, y, train, pilot.DEVICE, WD, args.hvp_batch)
    minimum = min(p["endpoints"]["min"]["value"] for p in probes)
    maximum = max(p["endpoints"]["max"]["value"] for p in probes)
    legacy.write_json(output / f"spectrum_seed{seed}.json", {"probes": probes,
          "undamped_min_ritz_estimate": minimum, "undamped_max_ritz_estimate": maximum,
          "baseline_eval_gradient_norm": core.norm(grad), "weight_decay": WD,
          "warning": "Finite Ritz estimates, not an SPD certificate"})
    rows, raw, attack_rows = [], [], []
    for patient in context["patients"]:
        pid, cls = patient["patient_id"], patient["oct_class"]
        _, truths_cpu = checkpoint_pair(context, pilot, seed, patient)
        truths = {key: value.to(theta.device) for key, value in truths_cpu.items()}
        b = score.per_image_grads(model, X, y, patient["raw_indices"], pilot.DEVICE).sum(0) / len(train)
        h_matrix = h[cls]["matrix"].double()
        actual = {key: h_matrix @ value.double() for key, value in truths.items()}
        l2_checks = {condition: compare_existing_l2(context, args, seed, patient, condition, float(values.mean()))
                     for condition, values in actual.items()}
        reverse = {condition: audit.reverse_residual(H, truths[condition], b, alpha, args.damping_grid)
                   for condition, alpha in (("full", 1.), ("dose01", .1))}
        legacy.write_json(output / f"reverse_seed{seed}_patient{pid}.json", reverse)
        krylov = audit.lanczos_from_rhs(H, b, max(args.krylov_steps))
        spectra = [audit.weighted_spectrum(krylov, m, args.damping_grid) for m in args.krylov_steps]
        legacy.write_json(output / f"rhs_spectrum_seed{seed}_patient{pid}.json", spectra)
        for gamma in args.damping_grid:
            short = audit.solve_shift(H, b, krylov, gamma, args.krylov_steps[0])
            solved = audit.solve_shift(H, b, krylov, gamma, args.krylov_steps[1])
            gate = audit.qualify_pair(short, solved, krylov["orthogonality_error"], args.residual_tol, args.stability_tol)
            screen = core.spectrum_screen(probes, gamma, args.ritz_tol)
            kind = "spd_screen" if screen["passed"] else ("negative_curvature_detected" if minimum + gamma < 0 else "undetermined")
            x, Hx = solved["x"], solved["Hx"]
            bnorm, xnorm = core.norm(b), core.norm(x)
            predicted = h_matrix @ x.double()
            predicted_short = h_matrix @ short["x"].double()
            projection_norm = core.norm(predicted)
            projection_change = core.norm(predicted - predicted_short)
            projection_relative_change = (projection_change / projection_norm if projection_norm else
                                          (0. if projection_change == 0 else None))
            projection_stable = (projection_relative_change is not None and
                                 math.isfinite(projection_relative_change) and
                                 projection_relative_change <= args.stability_tol)
            qualified = gate["linear_solve_qualified"] and h[cls]["attack_qualified"] and projection_stable
            shared = {"seed": seed, "patient_id": pid, "oct_class": cls, "gamma": gamma,
                      "qualified": qualified, "attack_solve_qualified": h[cls]["attack_qualified"], **gate,
                      "projection_depth_relative_change": projection_relative_change,
                      "projection_depth_stability_passed": projection_stable,
                      "operator_category": kind, "gamma_over_lambda_max_estimate": gamma / maximum if maximum > 0 else None,
                      "true_relative_residual": solved["true_relative_residual"],
                      "short_true_relative_residual": short["true_relative_residual"],
                      "projected_relative_residual": solved["projected_relative_residual"],
                      "rhs_norm": bnorm, "steps": solved["steps"]}
            extras = {"cosine_with_rhs": core.cosine(x, b),
                      "isotropic_relative_difference": core.norm(x - b / gamma) / xnorm if xnorm else None,
                      "damping_term_over_rhs": gamma * xnorm / bnorm if bnorm else None,
                      "curvature_term_over_rhs": core.norm(Hx) / bnorm if bnorm else None,
                      "signed_damping_projection": gamma * float(b.double() @ x.double()) / bnorm**2 if bnorm else None}
            for condition, alpha in (("full", 1.), ("dose01", .1)):
                values = {**core.geometry(alpha * x, truths[condition]), **extras,
                          "pred_h_projection": alpha * float(predicted.mean()),
                          "true_h_projection": float(actual[condition].mean())}
                base = {**shared, "condition": condition, "alpha": alpha,
                        "l2_reference_max_abs_diff": l2_checks[condition]}
                rows.append({**base, **{k: v if qualified else None for k, v in values.items()}})
                raw.append({**base, "unqualified_metrics_are_diagnostic_only": values,
                            "short_solve": {k: v for k, v in short.items() if k not in ("x", "Hx")},
                            "long_solve": {k: v for k, v in solved.items() if k not in ("x", "Hx")}})
                for index, attack_seed in enumerate(h[cls]["attack_seeds"]):
                    attack_rows.append({**base, "attack_seed": attack_seed,
                                        "pred_h_projection": alpha * float(predicted[index]) if qualified else None,
                                        "true_h_projection": float(actual[condition][index]) if qualified else None})
            legacy.write_csv(output / f"damping_seed{seed}.csv", rows)
            legacy.write_json(output / f"damping_seed{seed}.json", {"rows": rows, "raw_rows": raw})
            LOG.info("seed=%s patient=%s gamma=%s category=%s qualified=%s true_residual=%s",
                     seed, pid, gamma, kind, qualified, solved["true_relative_residual"])
    legacy.write_csv(output / f"projection_attack_seeds_seed{seed}.csv", attack_rows)
    return rows, {"seed": seed, "hvp_calls_measured": H.calls}


def projection_summary(rows, gammas):
    """Descriptive common-cell comparisons. No gamma selection and no p-values."""
    output = []
    for condition in ("full", "dose01"):
        candidates = [r for r in rows if r["condition"] == condition]
        expected_seeds = {r["seed"] for r in candidates}
        cells = {}
        for row in candidates:
            cells.setdefault((row["seed"], row["patient_id"]), []).append(row)
        common = {key for key, rs in cells.items() if len(rs) == len(gammas) and all(r["qualified"] for r in rs)}
        for gamma in gammas:
            current = [r for r in candidates if r["gamma"] == gamma]
            paired = [r for r in current if (r["seed"], r["patient_id"]) in common]
            for cls in (None, *sorted({r["oct_class"] for r in current})):
                matched = [r for r in paired if cls is None or r["oct_class"] == cls]
                available = [r for r in current if cls is None or r["oct_class"] == cls]
                for level in ("patient_seed", "patient_mean"):
                    values = [(r["pred_h_projection"], r["true_h_projection"]) for r in matched]
                    if level == "patient_mean":
                        complete_patients = {pid for pid in {r["patient_id"] for r in matched}
                                             if {r["seed"] for r in matched if r["patient_id"] == pid} == expected_seeds}
                        values = [(float(np.mean([r['pred_h_projection'] for r in matched if r['patient_id'] == pid])),
                                   float(np.mean([r['true_h_projection'] for r in matched if r['patient_id'] == pid])))
                                  for pid in sorted(complete_patients)]
                    pred, actual = np.asarray(values, dtype=float).reshape(-1, 2).T
                    valid = len(values) >= 3 and np.ptp(pred) > 0 and np.ptp(actual) > 0
                    output.append({"condition": condition, "gamma": gamma, "oct_class": cls, "level": level,
                                   "n_total_cells": len(available), "n_qualified_cells": sum(r['qualified'] for r in available),
                                   "n_common_cells": len(matched), "n": len(values),
                                   "spearman": float(spearmanr(pred, actual).statistic) if valid else None,
                                   "mae": float(np.mean(np.abs(pred - actual))) if len(values) else None,
                                   "sign_agreement": float(np.mean(np.sign(pred) == np.sign(actual))) if len(values) else None,
                                   "mean_prediction": float(np.mean(pred)) if len(values) else None,
                                   "mean_true_projection": float(np.mean(actual)) if len(values) else None})
    return output


def projection_plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, condition in zip(axes, ("full", "dose01")):
        values = [r for r in rows if r['condition'] == condition and r['qualified']]
        for gamma in sorted({r['gamma'] for r in values}):
            selected = [r for r in values if r['gamma'] == gamma]
            ax.scatter([r['true_h_projection'] for r in selected], [r['pred_h_projection'] for r in selected],
                       s=14, alpha=.7, label=f"gamma={gamma:g}, n={len(selected)}")
        if values:
            limits = [r[key] for r in values for key in ('true_h_projection', 'pred_h_projection')]
            ax.plot([min(limits), max(limits)], [min(limits), max(limits)], color='gray', linestyle=':')
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No qualified cells', ha='center', transform=ax.transAxes)
        ax.set(xlabel='h dot true parameter change', ylabel='h dot predicted parameter change', title=condition)
        ax.grid(alpha=.2)
    fig.suptitle('Qualified cells by gamma. Coverage may differ. Use common-cell tables for comparisons.', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('projection_overview.' + suffix), dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "convergence", "damping"], required=True)
    p.add_argument("--full_dir", default="results/cross_stage_calibration_v4_1/shadow3_full")
    p.add_argument("--dose_dir", default="results/cross_stage_calibration_v4_1/shadow3_dose01")
    p.add_argument("--data_dir", default="/u/yli103/oct2/data")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--damping_grid", type=float, nargs="+", default=[.03, .1, .3, .7, 1., 2.])
    p.add_argument("--krylov_steps", type=int, nargs=2, default=[50, 100])
    p.add_argument("--residual_tol", type=float, default=1e-3)
    p.add_argument("--stability_tol", type=float, default=.01)
    p.add_argument("--damping_attack", type=float, default=.2)
    p.add_argument("--j00_tol", type=float, default=1e-6)
    p.add_argument("--hvp_batch", type=int, default=64)
    p.add_argument("--lanczos_iters", type=int, default=50)
    p.add_argument("--probe_seeds", type=int, nargs="+", default=[1701, 1702])
    p.add_argument("--ritz_tol", type=float, default=.005)
    p.add_argument("--gradient_every", type=int, default=5)
    p.add_argument("--dropout_mc_reps", type=int, default=16)
    p.add_argument("--require_cuda", action="store_true")
    a = p.parse_args()
    if not 1 <= a.krylov_steps[0] < a.krylov_steps[1]:
        p.error("Require two increasing positive Krylov depths")
    if min(a.hvp_batch, a.lanczos_iters, a.gradient_every) < 1 or a.dropout_mc_reps < 2:
        p.error("Invalid observation or iteration settings")
    for values in (a.seeds, a.damping_grid, a.probe_seeds):
        if len(set(values)) != len(values):
            p.error("Duplicate seed or gamma")
    for v in [*a.damping_grid, a.residual_tol, a.stability_tol, a.ritz_tol, a.damping_attack, a.j00_tol]:
        if not math.isfinite(v) or v <= 0:
            p.error("Require finite positive damping and tolerances")
    return a


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this invocation")
    if args.mode == "preflight":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    output = Path(args.out_dir).resolve()
    for source in (Path(args.full_dir).resolve(), Path(args.dose_dir).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("Output overlaps original truth")
    output.mkdir(parents=True, exist_ok=False)
    score = legacy.module_from_path("score07_v11", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    started = time.time()
    manifest = {"schema": "stage1_diagnostics_v1.1", "status": "running", "args": vars(args),
                "torch": torch.__version__, "python": platform.python_version(),
                "device": "cpu" if args.mode == "preflight" else str(pilot.DEVICE),
                "source_sha256": {str(path): legacy.sha(path) for path in source_files()},
                "scope": "Exploratory diagnostics. No best-gamma selection or causal chaos decomposition."}
    legacy.write_json(output / "manifest.json", manifest)
    try:
        if args.mode == "preflight":
            context = metadata_context(args, pilot)
        else:
            context = legacy.load_context(args, pilot)
            context["splits"] = legacy.read_json(context["full"] / "splits/fresh_patient_split.json")
        manifest["training_setup"] = {key: getattr(context["pargs"], key) for key in
                                      ("shadow_epochs", "save_epoch_checkpoints", "shadow_lr", "shadow_batch_size")}
        fingerprints = {str(path): legacy.sha(path) for path in input_files(context, args)}
        legacy.write_json(output / "input_sha256.json", fingerprints)
        rows, details = [], []
        if args.mode == "preflight":
            rows, info = checkpoint_preflight(context, args, pilot, output)
            details.append(info)
        else:
            for seed in args.seeds:
                LOG.info("START mode=%s seed=%s", args.mode, seed)
                if args.mode == "convergence":
                    result, info = legacy.convergence_seed(args, context, pilot, seed, output)
                    for row in result:
                        if row.get("dropout_mc_reps", 0) > 1:
                            row["dropout_mc_gradient_mean_se_norm_estimate"] = row["dropout_mc_gradient_rms_spread"] / math.sqrt(row["dropout_mc_reps"] - 1)
                    legacy.write_csv(output / f"training_curve_seed{seed}.csv", result)
                else:
                    result, info = damping_seed(args, context, pilot, score, seed, output)
                rows.extend(result)
                details.append(info)
                legacy.write_csv(output / "all_rows.csv", rows)
        if not all(legacy.sha(Path(path)) == value for path, value in fingerprints.items()):
            raise RuntimeError("Original input changed during diagnostics")
        if args.mode in ("convergence", "damping"):
            legacy.make_plots(args.mode, rows, output)
        if args.mode == "damping":
            summary = projection_summary(rows, args.damping_grid)
            legacy.write_csv(output / "projection_summary.csv", summary)
            manifest["projection_summary"] = summary
            projection_plot(rows, output)
        manifest.update(status="complete", elapsed_seconds=time.time() - started,
                        n_rows=len(rows), per_seed_extra=details, input_files_unchanged=True)
        legacy.write_json(output / "manifest.json", manifest)
        LOG.info("COMPLETE -> %s", output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
