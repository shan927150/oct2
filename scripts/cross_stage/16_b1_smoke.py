#!/usr/bin/env python3
"""B1: audited short-prefix forward tangents, dual to pathway-2 (73)-(74).

G1 replays the ENTIRE frozen training horizon against saved model/Adam/RNG states.
G2 measures functional/native fidelity without calling approximate equality exact.
G3 requires an absent-patient tangent to be finite and exactly zero in all states.
G4 compares matched-map finite differences and tangents in parameters AND predictions.
Float64 is a separate numerical control. An unresolved dose sweep is a scientific
result, not proof of an incorrect derivative, chaos, or a universal precision floor.
G5 measures the first-step epsilon mechanism including dg/dalpha and learning rate.
No training hyperparameters, truth directories, or Stage-2 code are changed.
"""
from __future__ import annotations
import argparse
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
from torch.func import jvp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core
import stage1_trajectory_core as trajectory

spec = importlib.util.spec_from_file_location("b1_e0", HERE / "13_e0_residual_decomposition.py")
e0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e0
spec.loader.exec_module(e0)
LOG = logging.getLogger("b1")
SCHEMA = "pathway2_b1_smoke_v2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("full_dir", "dose_dir", "data_dir", "out_dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--patients", type=int, nargs="+", default=[807, 2085, 1369])
    p.add_argument("--prefix_epochs", type=int, nargs="+", default=[0, 1, 5],
                   help="0 = first update; positive integers = whole epochs; adds first exposure")
    p.add_argument("--exponents", type=int, nargs="+", default=[10, 14, 18, 22, 24])
    p.add_argument("--exponents64", type=int, nargs="+", default=[10, 14, 18, 22, 24, 28, 32])
    p.add_argument("--derivative_tol", type=float, default=1e-2)
    p.add_argument("--cosine_min", type=float, default=0.9999)
    p.add_argument("--prediction_batch", type=int, default=64)
    p.add_argument("--require_cuda", action="store_true")
    args = p.parse_args()
    if min(args.prefix_epochs) < 0 or args.prediction_batch < 1:
        p.error("Nonnegative whole-epoch prefixes and positive prediction batch required")
    if not 0 < args.derivative_tol < 1 or not 0 < args.cosine_min <= 1:
        p.error("Invalid derivative/cosine tolerances")
    for key, dtype in (("exponents", torch.float32), ("exponents64", torch.float64)):
        vals = getattr(args, key)
        if len(set(vals)) != len(vals) or len(vals) < 2:
            p.error("Each dose ladder needs at least two distinct exponents")
        for exponent in vals:
            trajectory.exact_alpha(exponent, dtype)
    if len(set(args.patients)) != len(args.patients):
        p.error("Duplicate patients")
    return args


def checkpoint_match(snapshot, reference):
    meta = reference["metadata"]
    return {"parameters_bitwise_equal": core.exact_tree(snapshot["state_dict"], reference["state_dict"]),
            "optimizer_bitwise_equal": core.exact_tree(snapshot["optimizer_state"], meta["optimizer_state"]),
            "rng_bitwise_equal": core.exact_tree(snapshot["rng_states"], meta["rng_states"])}


def state_match(value, reference):
    metrics = {field: trajectory.compare(value[field], reference[field]) for field in ("theta", "m", "v")}
    metrics["step_equal"] = value["step"] == reference["step"]
    metrics["bitwise_equal"] = metrics["step_equal"] and all(metrics[f]["bitwise_equal"] for f in ("theta", "m", "v"))
    return metrics


def resolved_band(rows, tol, cosine):
    """Two consecutive SAMPLED doses must pass jointly, not two unrelated minima."""
    rows = sorted(rows, key=lambda r: r["exponent"])
    if not rows:
        raise ValueError("Empty dose sweep")
    if not rows[0]["exposed"]:
        zero = all(r[f+"_absolute_error"] == 0 and r[f+"_reference_norm"] == 0
                   for r in rows for f in ("param", "pred", "moment1", "moment2"))
        return {"status": "ZERO_CONTROL" if zero else "FAILED_ZERO_CONTROL", "resolved": False}
    def good(row):
        return all(row[f+"_relative_error"] is not None and row[f+"_relative_error"] <= tol
                   and row[f+"_cosine"] is not None and row[f+"_cosine"] >= cosine
                   for f in ("param", "pred"))
    pairs = [[a["alpha"], b["alpha"]] for a, b in zip(rows, rows[1:]) if good(a) and good(b)]
    errors = [r["param_relative_error"] for r in rows if r["param_relative_error"] is not None]
    return {"status": "RESOLVED_SAMPLED_BAND" if pairs else "UNRESOLVED_AT_TESTED_DOSES",
            "resolved": bool(pairs), "passing_adjacent_pairs": pairs,
            "best_param_relative_error": min(errors) if errors else None,
            "error_decreased_over_sweep": errors[-1] < errors[0] if len(errors) > 1 else None,
            "note": "Joint parameter/prediction agreement at adjacent sampled doses; no continuous interval, alpha=1, or later-prefix guarantee."}


def run(args, context, legacy, pilot, output):
    p, X, y = context["pargs"], context["X"], context["y"]
    seed, device = args.seed, pilot.DEVICE
    orders = legacy.verify_orders(context, seed)
    split = legacy.read_json(context["full"] / "splits/fresh_patient_split.json")["shadow_models"][p.affected_shadow]
    panel = {int(v["patient_id"]): list(map(int, v["raw_indices"])) for v in context["patients"]}
    if set(args.patients) - panel.keys():
        raise RuntimeError("Requested patient absent from the frozen panel")
    cfg = {"lr": p.shadow_lr, "batch_size": p.shadow_batch_size,
           "wd": context["config"]["oct_config"]["target_l2"], "deterministic": p.deterministic}
    per_epoch = math.ceil(len(orders[0]) / cfg["batch_size"])
    full_plan = trajectory.batch_plan(orders, cfg["batch_size"])
    if len(orders) != p.shadow_epochs or max(args.prefix_epochs) > p.shadow_epochs:
        raise RuntimeError("Requested horizon differs from/outside frozen configuration")
    first = {pid: next((k for k, (_, idx) in enumerate(full_plan, 1)
                       if set(idx.tolist()) & set(panel[pid])), None) for pid in args.patients}
    if None in first.values():
        raise RuntimeError("A panel patient never appears in the frozen trajectory")
    requested = {1 if ep == 0 else ep*per_epoch for ep in args.prefix_epochs}
    prefixes = sorted(requested | set(first.values()) | {1})
    steps = max(prefixes)
    plan = full_plan[:steps]
    frozen = sorted(set(map(int, p.save_epoch_checkpoints)))
    if not frozen or frozen[-1] != p.shadow_epochs or min(frozen) < 1:
        raise RuntimeError("G1 requires saved epoch checkpoints including the frozen final epoch")
    gates = {"G1": {"status": "running", "original_epochs": p.shadow_epochs,
                    "checked_against_original_epoch_checkpoints": []}}
    def save_gates():
        legacy.write_json(output / "b1_gates.json", gates)
    save_gates()
    LOG.info("G1 full %d-epoch replay; derivatives only at steps %s; first exposure %s",
             p.shadow_epochs, prefixes, first)
    model, optimizer, native, masks, _ = trajectory.faithful_run(
        pilot, X, y, orders, seed, cfg, 0., (), len(full_plan), prefixes=prefixes,
        record_masks=steps, audit_steps=[ep*per_epoch for ep in frozen])
    for epoch in frozen:
        ref = legacy.load_payload(context["full"] / "checkpoints" /
                                  f"baseline_seed{seed}_epochs/epoch{epoch:03d}.pt", pilot)
        row = {"epoch": epoch, **checkpoint_match(native[epoch*per_epoch], ref)}
        gates["G1"]["checked_against_original_epoch_checkpoints"].append(row)
        save_gates()
        if not all(row[k] for k in row if k != "epoch"):
            raise RuntimeError(f"G1 original replay mismatch at epoch {epoch}: {row}")
    final_ref = legacy.load_payload(context["full"] / "checkpoints" /
                                    f"shadow_{p.affected_shadow}_baseline_seed{seed}.pt", pilot)
    final_equal = core.exact_tree(model.state_dict(), final_ref["state_dict"])
    gates["G1"].update(final_parameters_bitwise_equal=final_equal, mask_count=len(masks))
    if not final_equal or len(masks) != steps:
        raise RuntimeError("G1 final baseline/mask count mismatch")
    gates["G1"]["status"] = "PASS_FULL_REPLAY"
    save_gates()
    del model, optimizer, final_ref, ref

    pilot.seed_everything(seed, cfg["deterministic"])
    model = pilot.build_model("cnn", X.shape[1], 128, int(np.max(y))+1)
    theta0 = trajectory.flat(model)
    trajectory.swap_in_recorder(model)
    bases, zeros, g2, g3 = {}, {}, [], []
    outside = sorted(set(range(len(y))) - set(map(int, split["train_idx"])))
    if not outside:
        raise RuntimeError("No outside-training row for G3")
    for lane, dtype in (("functional32", torch.float32), ("functional64", torch.float64)):
        th0 = theta0.to(dtype)
        bases[lane] = trajectory.functional_trajectory(model, X, y, cfg, device, th0,
            plan, masks, (), 0., prefixes)
        zeros[lane] = trajectory.functional_trajectory(model, X, y, cfg, device, th0,
            plan, masks, outside[:8], 0., prefixes, tangent=True)
        for k in prefixes:
            if lane == "functional32":
                g2.append({"prefix_steps": k, **state_match(bases[lane][k], native[k])})
            zero = all(torch.count_nonzero(zeros[lane][k][f"u_{f}"]).item() == 0 for f in ("theta", "m", "v"))
            same = state_match(zeros[lane][k], bases[lane][k])["bitwise_equal"]
            g3.append({"lane": lane, "prefix_steps": k, "all_state_tangents_exactly_zero": zero,
                       "primal_bitwise_equal": same})
            if not zero or not same:
                raise RuntimeError(f"G3 zero direction failed: {g3[-1]}")
    gates.update(G2=g2, G3=g3, G4=[], G5=[])
    save_gates()
    LOG.info("G1/G3 passed; G2 functional/native all-state bitwise equality = %s",
             all(r["bitwise_equal"] for r in g2))
    rows_interface = np.concatenate([split["train_idx"], split["test_idx"]]).astype(np.int64)
    def predict(theta, u=None):
        return trajectory.prediction_jvp(model, X, rows_interface, theta, u, device, args.prediction_batch)
    base_pred = {lane: {k: predict(sn[k]["theta"])[0] for k in prefixes}
                 for lane, sn in {**bases, "native32": native}.items()}
    rows, cross_precision, perturbed_fidelity = [], [], []
    for pid in args.patients:
        LOG.info("patient %d: starting two-precision tangent/FD sweeps", pid)
        idx = panel[pid]
        tangents, pred_tangents = {}, {}
        for lane, dtype in (("functional32", torch.float32), ("functional64", torch.float64)):
            tan = trajectory.functional_trajectory(model, X, y, cfg, device, theta0.to(dtype),
                plan, masks, idx, 0., prefixes, tangent=True)
            tangents[lane] = tan
            pred_tangents[lane] = {}
            for k in prefixes:
                if not state_match(tan[k], bases[lane][k])["bitwise_equal"]:
                    raise RuntimeError("Tangent propagation changed the functional primal")
                _, pred_tangents[lane][k] = predict(tan[k]["theta"], tan[k]["u_theta"])
                gates["G5"].append({"patient_id": pid, "lane": lane, "prefix_steps": k,
                    "exposed": k >= first[pid], **trajectory.concentration(tan[k]["u_theta"]),
                    "gradient_ad_primal_relative_gap": tan[k]["max_gradient_ad_primal_relative_gap"]})
        # This attribution uses g, q and u from ONE precision and ONE first update.
        gradient, _ = trajectory.make_gradient(model, cfg)
        th = theta0.double()
        trajectory.dropout_module(model, (trajectory.RecordedDropout,)).mask = masks[0].to(device=device, dtype=th.dtype)
        bi = plan[0][1]
        xb = torch.as_tensor(X[bi], dtype=th.dtype, device=device)
        yb = torch.as_tensor(y[bi], dtype=torch.long, device=device)
        removed = torch.as_tensor(np.isin(bi, idx), device=device)
        al = torch.zeros((), dtype=th.dtype, device=device)
        g = gradient(th, al, xb, yb, removed)
        g_ad, q = jvp(lambda a: gradient(th, a, xb, yb, removed), (al,), (torch.ones_like(al),))
        gates["G5"].append({"patient_id": pid, "lane": "first_step_float64_attribution",
            "prefix_steps": 1, "exposed": first[pid] == 1,
            "gradient_ad_primal_relative_gap": trajectory.compare(g_ad, g)["relative_error"],
            **trajectory.first_step_epsilon(g, q, tangents["functional64"][1]["u_theta"], cfg["lr"])})
        for k in prefixes:
            cross_precision.append({"patient_id": pid, "prefix_steps": k,
                "baseline": trajectory.compare(bases["functional32"][k]["theta"], bases["functional64"][k]["theta"]),
                "tangent": trajectory.compare(tangents["functional32"][k]["u_theta"], tangents["functional64"][k]["u_theta"]),
                "note": "Different numerical trajectories; difference is a sensitivity diagnostic, not an error bound for the original run."})
        for lane, dtype, exponents in (("functional32", torch.float32, args.exponents),
                                       ("functional64", torch.float64, args.exponents64)):
            for exponent in sorted(exponents):
                alpha = trajectory.exact_alpha(exponent, dtype)
                pert = trajectory.functional_trajectory(model, X, y, cfg, device, theta0.to(dtype),
                    plan, masks, idx, alpha, prefixes)
                variants = [(lane, pert, bases[lane])]
                if lane == "functional32":
                    # Independent native dropout replay, same seed/order; no inferred masks.
                    nm, no, native_pert, _, _ = trajectory.faithful_run(pilot, X, y, orders, seed,
                        cfg, alpha, idx, steps, prefixes=prefixes)
                    del nm, no
                    variants.append(("native32", native_pert, native))
                    for k in prefixes:
                        perturbed_fidelity.append({"patient_id": pid, "prefix_steps": k,
                            "exponent": exponent, **state_match(pert[k], native_pert[k])})
                for label, variant, baseline in variants:
                    for k in prefixes:
                        row = {"seed": seed, "patient_id": pid, "lane": label, "prefix_steps": k,
                               "prefix_epochs": k/per_epoch, "first_exposure_step": first[pid],
                               "exposed": k >= first[pid], "exponent": exponent, "alpha": alpha}
                        for field, short in (("theta", "param"), ("m", "moment1"), ("v", "moment2")):
                            metrics = trajectory.compare(trajectory.fd(variant[k][field], baseline[k][field], alpha),
                                                         tangents[lane][k]["u_"+field])
                            row.update({short+"_"+key: value for key, value in metrics.items()})
                        pred, _ = predict(variant[k]["theta"])
                        metrics = trajectory.compare(trajectory.fd(pred, base_pred[label][k], alpha), pred_tangents[lane][k])
                        row.update({"pred_"+key: value for key, value in metrics.items()})
                        row["reference"] = "same functional map" if label != "native32" else "functional32 tangent candidate; requires G2 and perturbed fidelity"
                        rows.append(row)
                legacy.write_csv(output / "b1_rows.csv", rows)
                LOG.info("patient=%d lane=%s alpha=2^-%d measured %d prefixes", pid, lane, exponent, len(prefixes))
        save_gates()
    for pid in args.patients:
        for k in prefixes:
            for lane in ("functional32", "functional64", "native32"):
                band = resolved_band([r for r in rows if r["patient_id"] == pid and r["prefix_steps"] == k
                                      and r["lane"] == lane], args.derivative_tol, args.cosine_min)
                if band["status"] == "FAILED_ZERO_CONTROL":
                    raise RuntimeError("Unexposed prefix has a nonzero paired response")
                gates["G4"].append({"patient_id": pid, "prefix_steps": k, "lane": lane, **band})
    save_gates()
    faithful = all(r["bitwise_equal"] for r in g2 + perturbed_fidelity)
    summary = {"gates": gates, "cross_precision": cross_precision,
               "perturbed_native_functional_fidelity": perturbed_fidelity,
               "original_numerical_trajectory_matched_at_sampled_states": faithful,
               "ready_for_B2": False,
               "note": "Stage-1 prefixes only. B2 requires the full original Stage-1 horizon and Stage-2 trajectory hypergradient (42); a Stage-2 implicit h would remain a hybrid. No alpha=1 validity claim."}
    return summary, rows, prefixes, per_epoch


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = Path(args.out_dir).resolve()
    for source in map(lambda x: Path(x).resolve(), (args.full_dir, args.dose_dir, args.data_dir, HERE.parents[1])):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("Output overlaps an input/code directory")
    output.mkdir(parents=True, exist_ok=False)
    legacy, _, pilot = e0.load_modules()
    started = time.time()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    code_paths = [Path(__file__), HERE/"stage1_trajectory_core.py", HERE/"stage1_diagnostic_core.py",
                  HERE/"05_end_to_end_patient_loo_pilot.py", HERE/"11_stage1_diagnostics.py",
                  HERE/"13_e0_residual_decomposition.py", HERE.parents[1]/"src/models.py"]
    source_hash = {str(q): legacy.sha(q) for q in code_paths}
    manifest = {"schema": SCHEMA, "status": "running", "args": vars(args), "git_commit": commit,
                "python": platform.python_version(), "torch": torch.__version__, "device": str(pilot.DEVICE),
                "tf32": trajectory.tf32_settings(), "source_sha256": source_hash,
                "scope": "Frozen-configuration replay + Stage-1 prefix derivative diagnostics; high precision is a separate control."}
    legacy.write_json(output/"manifest.json", manifest)
    fingerprints = {}
    try:
        ctx_args = argparse.Namespace(full_dir=args.full_dir, dose_dir=args.dose_dir, mode="damping",
                                      data_dir=args.data_dir, seeds=[args.seed])
        cfg, _, panel = legacy.validate_sources(Path(args.full_dir), Path(args.dose_dir), pilot, [args.seed])
        p = argparse.Namespace(**cfg["args"])
        metadata = {"full": Path(args.full_dir), "dose": Path(args.dose_dir), "pargs": p, "patients": panel["patients"]}
        paths = legacy.input_paths(metadata, ctx_args) + [Path(args.full_dir)/"checkpoints"/
            f"baseline_seed{args.seed}_epochs/epoch{ep:03d}.pt" for ep in p.save_epoch_checkpoints]
        fingerprints = {str(q.resolve()): legacy.sha(q) for q in paths}
        legacy.write_json(output/"input_sha256.json", fingerprints)
        context = legacy.load_context(ctx_args, pilot)
        manifest["loaded_data_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(context[k]).tobytes()).hexdigest()
                                          for k in ("X", "y")}
        legacy.write_json(output/"manifest.json", manifest)
        summary, rows, prefixes, per_epoch = run(args, context, legacy, pilot, output)
        legacy.write_json(output/"b1_summary.json", summary)
        legacy.write_json(output/"b1_rows.json", {"rows": rows, "prefixes": prefixes,
                                                "batches_per_epoch": per_epoch})
        manifest.update(status="complete", n_rows=len(rows))
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        changed = [q for q, h in {**fingerprints, **source_hash}.items()
                   if not Path(q).is_file() or legacy.sha(Path(q)) != h]
        manifest.update(input_files_unchanged=bool(fingerprints) and not changed, changed_files=changed,
                        elapsed_seconds=time.time()-started, tf32_after=trajectory.tf32_settings())
        if changed:
            manifest["status"] = "failed_inputs_or_code_changed"
        legacy.write_json(output/"manifest.json", manifest)
        if changed:
            raise RuntimeError("Input/code hashes changed; result rejected")
    print(json.dumps({"execution_status": manifest["status"], "original_numerical_fidelity":
                      summary["original_numerical_trajectory_matched_at_sampled_states"],
                      "G4": summary["gates"]["G4"], "ready_for_B2": False}, indent=2))


if __name__ == "__main__":
    main()
