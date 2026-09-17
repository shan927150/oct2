#!/usr/bin/env python3
"""B0: is there a numerically resolvable local deletion-weight derivative of the ORIGINAL algorithm?

Reads finished 05 truth directories that differ only in --deletion_weight and forms

    D_alpha = (theta_T(alpha) - theta_T(0)) / alpha

for every (seed, patient, alpha).  A local derivative exists in a resolvable band if
D_alpha stops moving as alpha shrinks, in direction and in magnitude, before the
difference sinks into float noise.  The readout is therefore a ladder of
*adjacent-dose* comparisons, not a single number:

    cos(D_a, D_a')            direction stability between neighbouring doses
    ||D_a - D_a'|| / ||D_a'|| magnitude stability
    ||theta_T(a)-theta_T(0)|| absolute displacement, compared against the no-op floor

The same three quantities are reported in three spaces: raw parameters, the fixed
Stage-1 probe h (an original-attack direction, held constant, not re-derived here),
and the affected-shadow prediction matrix P.  No training and no Hessian: this
script only reads checkpoints and interface arrays.

Nothing here decides that a derivative does not exist.  When no band is resolvable
the verdict is "unresolved at these doses and this precision", which is a statement
about the probe, not about the algorithm.
"""
from __future__ import annotations
import argparse
import logging
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

LOG = logging.getLogger("b0_dose")
SCHEMA = "pathway2_b0_dose_derivative_v1"


def load_modules():
    import importlib.util
    spec = importlib.util.spec_from_file_location("b0_legacy11", HERE / "11_stage1_diagnostics.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    score = legacy.module_from_path("b0_score07", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    return legacy, score, pilot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dose_dirs", nargs="+", required=True,
                   help="finished 05 directories, one per alpha (order is irrelevant; alpha is read "
                        "from each experiment_config.json)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--patients", type=int, nargs="*", default=None,
                   help="default: every patient present in ALL supplied dose directories")
    p.add_argument("--h_probe", default=None,
                   help="h_seed*.pt written by the v1.1 damping mode; supplies the fixed Stage-1 "
                        "probe direction. Optional: without it the h columns are omitted rather "
                        "than filled with a substitute direction.")
    p.add_argument("--direction_tol", type=float, default=0.05,
                   help="pre-registered engineering gate: 1 - cos between adjacent doses")
    p.add_argument("--magnitude_tol", type=float, default=0.05,
                   help="pre-registered engineering gate: relative change between adjacent doses")
    p.add_argument("--require_cuda", action="store_true")
    args = p.parse_args()
    if len(args.dose_dirs) < 2:
        raise ValueError("Need at least two doses to compare adjacent rungs")
    if not 0 < args.direction_tol < 1 or not 0 < args.magnitude_tol < 1:
        raise ValueError("Tolerances are fractions in (0, 1)")
    return args


def parameter_vector(payload):
    state = payload["state_dict"]
    if not all(torch.is_floating_point(t) for t in state.values()):
        raise RuntimeError("Expected the buffer-free v4.1 SmallCNN state dictionary")
    return torch.cat([v.detach().cpu().double().reshape(-1) for v in state.values()])


def read_dose_directory(legacy, pilot, root: Path, seeds):
    """Validate one 05 output directory and return its alpha plus per-cell payload paths."""
    root = Path(root).resolve()
    config = legacy.read_json(root / "experiment_config.json")
    summary = legacy.read_json(root / "experiment_summary.json")
    panel = legacy.read_json(root / "selected_patients.json")
    pilot.require_training_numerics(config, str(root))
    a = config["args"]
    legacy.assert_equal(summary["status"], "complete", f"{root.name}: truth is incomplete")
    legacy.assert_equal(a["deletion_mode"], "fixed_mask", f"{root.name}: expected fixed_mask")
    legacy.assert_equal(a.get("removal_epochs"), None, f"{root.name}: windowed truth is not comparable")
    legacy.assert_equal(a["deterministic"], True, f"{root.name}: strict determinism is required")
    alpha = float(a["deletion_weight"])
    if not 0 < alpha <= 1:
        raise RuntimeError(f"{root.name}: alpha outside (0, 1]")
    missing = [s for s in seeds if s not in a["seeds"]]
    if missing:
        raise RuntimeError(f"{root.name}: seeds {missing} were not trained here")
    return {"root": root, "alpha": alpha, "config": config, "summary": summary,
            "panel": panel, "shadow": int(a["affected_shadow"]),
            "patients": {int(p["patient_id"]): p for p in panel["patients"]}}


def cross_validate(doses):
    """Every dose must describe the same split, panel and Stage-1 training contract."""
    reference = doses[0]
    keys = ("seeds", "affected_shadow", "split_seed", "selection_seed", "target_seed",
            "fixed_shadow_seed", "shadow_epochs", "shadow_lr", "shadow_batch_size",
            "attack_epochs", "deletion_mode", "window_membership", "deterministic",
            "n_total_samples", "target_data_size", "shadow_data_size", "n_shadow", "classes")
    seen = {}
    for entry in doses:
        if entry["alpha"] in seen:
            raise RuntimeError(f"Two directories share alpha={entry['alpha']}: "
                               f"{seen[entry['alpha']].name} and {entry['root'].name}")
        seen[entry["alpha"]] = entry["root"]
        if entry is reference:
            continue
        for key in keys:
            if entry["config"]["args"].get(key) != reference["config"]["args"].get(key):
                raise RuntimeError(f"{entry['root'].name}: config differs from "
                                   f"{reference['root'].name} in {key}")
        if entry["summary"]["split_sha256"] != reference["summary"]["split_sha256"]:
            raise RuntimeError(f"{entry['root'].name}: different split")
        for pid, patient in entry["patients"].items():
            other = reference["patients"].get(pid)
            if other is not None and sorted(patient["raw_indices"]) != sorted(other["raw_indices"]):
                raise RuntimeError(f"patient {pid}: raw indices differ between doses")


def baseline_vector(legacy, pilot, doses, seed):
    """theta_T(0): identical across doses by construction; verified bitwise here."""
    vectors, rng = [], None
    for entry in doses:
        path = entry["root"] / "checkpoints" / f"shadow_{entry['shadow']}_baseline_seed{seed}.pt"
        payload = legacy.load_payload(path, pilot)
        legacy.assert_equal(payload["metadata"]["seed"], seed, f"{path.name}: seed mismatch")
        fingerprint = payload["metadata"]["metrics"]["post_train_rng_sha256"]
        if rng is None:
            rng, reference = fingerprint, payload
        elif fingerprint != rng or not core.exact_tree(payload["state_dict"], reference["state_dict"]):
            raise RuntimeError(f"seed {seed}: baselines differ between dose directories "
                               f"({entry['root'].name}); the doses are not paired")
        vectors.append(parameter_vector(payload))
    return vectors[0], rng


def prediction_matrix(root: Path, seed: int, pid: int):
    """Affected-shadow prediction rows saved by 05 for this cell (baseline and perturbed)."""
    path = root / "runs" / f"seed{seed}_patient{pid}_interface.npz"
    if not path.is_file():
        return None, None
    with np.load(path) as payload:
        return (torch.as_tensor(payload["p_baseline"], dtype=torch.float64).reshape(-1),
                torch.as_tensor(payload["p_loo"], dtype=torch.float64).reshape(-1))


def stability(current, previous):
    """Direction and magnitude agreement between two dose rungs."""
    if previous is None or current is None:
        return {"cosine": None, "relative_change": None}
    scale = core.norm(previous)
    return {"cosine": core.cosine(current, previous),
            "relative_change": core.norm(current - previous) / scale if scale else None}


def verdict(rows, direction_tol, magnitude_tol):
    """Lowest adjacent pair that passes both engineering gates, if any."""
    unresolved = [r for r in rows if not r.get("above_storage_resolution", True)]
    passing = [r for r in rows if r["parameter_cosine_vs_larger"] is not None
               and r.get("above_storage_resolution", True)
               and (1 - r["parameter_cosine_vs_larger"]) <= direction_tol
               and r["parameter_relative_change_vs_larger"] is not None
               and r["parameter_relative_change_vs_larger"] <= magnitude_tol]
    floor_note = ([f"alpha={r['alpha']:g} is within 100x of the float32 checkpoint floor and was "
                   "excluded from the gate" for r in unresolved] or None)
    if not passing:
        return {"resolved_band": False,
                "statement": ("No adjacent dose pair met the pre-registered direction and magnitude "
                              "gates. This is unresolved at these doses and this precision; it is "
                              "not evidence that a local derivative does not exist."),
                "smallest_passing_alpha": None, "largest_passing_alpha": None,
                "below_storage_resolution": floor_note}
    alphas = sorted({r["alpha"] for r in passing} | {r["larger_alpha"] for r in passing})
    return {"resolved_band": True,
            "statement": ("Adjacent doses agree within the pre-registered engineering tolerance over "
                          "this band. The band is a numerical observation about the probe, not a "
                          "proof of local linearity over the full deletion."),
            "smallest_passing_alpha": min(alphas), "largest_passing_alpha": max(alphas),
            "n_passing_pairs": len(passing), "below_storage_resolution": floor_note}


def run(args, legacy, score, pilot, doses, output):
    ladder = sorted(doses, key=lambda e: -e["alpha"])          # large -> small
    shared = set.intersection(*(set(e["patients"]) for e in ladder))
    patients = sorted(set(args.patients) & shared) if args.patients else sorted(shared)
    if args.patients:
        missing = sorted(set(args.patients) - shared)
        if missing:
            raise RuntimeError(f"Patients {missing} are not present in every dose directory")
    if not patients:
        raise RuntimeError("No patient is present in every supplied dose directory")
    LOG.info("dose ladder alpha=%s patients=%s seeds=%s",
             [e["alpha"] for e in ladder], patients, args.seeds)

    probe = None
    if args.h_probe:
        payload = torch.load(Path(args.h_probe).resolve(), weights_only=False, map_location="cpu")
        stacked = torch.cat([torch.as_tensor(v).reshape(-1, v.shape[-1]).double()
                             for v in payload["h_by_class"].values()])
        probe = stacked.mean(0)
        LOG.info("fixed h probe from %s: %d rows, ||h||=%.4g",
                 args.h_probe, stacked.shape[0], core.norm(probe))

    rows = []
    for seed in args.seeds:
        theta0, rng = baseline_vector(legacy, pilot, ladder, seed)
        for pid in patients:
            patient = ladder[0]["patients"][pid]
            previous = {"parameter": None, "prediction": None, "alpha": None}
            for entry in ladder:
                alpha, root = entry["alpha"], entry["root"]
                path = root / "checkpoints" / f"shadow_{entry['shadow']}_seed{seed}_patient{pid}.pt"
                payload = legacy.load_payload(path, pilot)
                md = payload["metadata"]
                legacy.assert_equal(md["seed"], seed, f"{path.name}: seed mismatch")
                legacy.assert_equal(float(md["deletion_weight"]), alpha, f"{path.name}: alpha mismatch")
                legacy.assert_equal(sorted(md["excluded_indices"]), sorted(patient["raw_indices"]),
                                    f"{path.name}: patient mismatch")
                legacy.assert_equal(md["metrics"]["post_train_rng_sha256"], rng,
                                    f"{path.name}: fixed_mask RNG fingerprint differs from the baseline")
                theta_a = parameter_vector(payload)
                d = theta_a - theta0
                D = d / alpha
                # Checkpoints are stored in float32, so theta carries ~eps*||theta|| of
                # representation noise and the difference of two checkpoints carries about
                # sqrt(2) of it. Below this the ladder measures storage, not training.
                floor = float(np.finfo(np.float32).eps) * np.sqrt(2.0) * max(
                    core.norm(theta0), core.norm(theta_a))
                p0, p1 = prediction_matrix(root, seed, pid)
                Dp = (p1 - p0) / alpha if p0 is not None else None
                row = {
                    "seed": seed, "patient_id": pid, "oct_class": int(patient["oct_class"]),
                    "n_images": int(patient["n_images"]), "alpha": alpha,
                    "displacement_norm": core.norm(d), "derivative_norm": core.norm(D),
                    "larger_alpha": previous["alpha"],
                    "float32_resolution_floor": floor,
                    "displacement_over_floor": core.norm(d) / floor if floor else None,
                    "above_storage_resolution": bool(core.norm(d) > 100 * floor),
                }
                for space, current, name in (("parameter", D, "parameter"),
                                             ("prediction", Dp, "prediction")):
                    result = stability(current, previous[space])
                    row[f"{name}_cosine_vs_larger"] = result["cosine"]
                    row[f"{name}_relative_change_vs_larger"] = result["relative_change"]
                if Dp is not None:
                    row["prediction_derivative_norm"] = core.norm(Dp)
                if probe is not None:
                    row["h_projection"] = float(torch.dot(probe, d))
                    row["h_projection_derivative"] = float(torch.dot(probe, D))
                rows.append(row)
                previous = {"parameter": D, "prediction": Dp, "alpha": alpha}
                LOG.info("seed=%d patient=%d alpha=%g ||d||=%.4g ||D||=%.4g cos_vs_larger=%s",
                         seed, pid, alpha, row["displacement_norm"], row["derivative_norm"],
                         "n/a" if row["parameter_cosine_vs_larger"] is None
                         else f"{row['parameter_cosine_vs_larger']:.4f}")
    legacy.write_csv(output / "b0_rows.csv", rows)
    summary = verdict(rows, args.direction_tol, args.magnitude_tol)
    per_cell = {}
    for row in rows:
        key = f"seed{row['seed']}_patient{row['patient_id']}"
        per_cell.setdefault(key, []).append(row)
    legacy.write_json(output / "b0_summary.json", {
        "ladder": [{"alpha": e["alpha"], "directory": str(e["root"])} for e in ladder],
        "patients": patients, "seeds": args.seeds,
        "gates": {"direction_tol": args.direction_tol, "magnitude_tol": args.magnitude_tol,
                  "note": "Pre-registered engineering tolerances, not theoretical constants."},
        "verdict": summary,
        "per_cell_verdict": {k: verdict(v, args.direction_tol, args.magnitude_tol)
                             for k, v in per_cell.items()},
        "h_probe": str(Path(args.h_probe).resolve()) if args.h_probe else None,
        "h_probe_role": ("Fixed original-attack Stage-1 direction used as a diagnostic probe. It is "
                         "not re-derived at the perturbed endpoints and is not a trajectory h."),
        "scope": ("Local-derivative resolvability of the original finite-Adam algorithm. Says nothing "
                  "about alpha=1 extrapolation or about the cross-stage endpoint."),
    })
    return rows, summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Delta invocation")
    output = Path(args.out_dir).resolve()
    for source in args.dose_dirs:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("B0 output overlaps a truth directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    legacy, score, pilot = load_modules()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {"schema": SCHEMA, "status": "running", "args": vars(args),
                "python": platform.python_version(), "torch": torch.__version__,
                "git_commit": commit,
                "source_sha256": {str(p): legacy.sha(p) for p in (
                    HERE / "05_end_to_end_patient_loo_pilot.py", HERE / "11_stage1_diagnostics.py",
                    HERE / "stage1_diagnostic_core.py", Path(__file__))},
                "method": "checkpoint differences only; no training, no Hessian, no attack"}
    legacy.write_json(output / "manifest.json", manifest)
    try:
        doses = [read_dose_directory(legacy, pilot, root, args.seeds) for root in args.dose_dirs]
        cross_validate(doses)
        inputs = []
        for entry in doses:
            inputs.extend(entry["root"] / name for name in (
                "experiment_config.json", "experiment_summary.json", "selected_patients.json"))
        fingerprints = {str(p): legacy.sha(p) for p in inputs}
        legacy.write_json(output / "input_sha256.json", fingerprints)
        rows, summary = run(args, legacy, score, pilot, doses, output)
        after = {str(p): legacy.sha(p) for p in inputs}
        manifest.update(status="complete" if after == fingerprints else "complete_but_inputs_changed",
                        elapsed_seconds=time.time() - started, n_rows=len(rows),
                        input_files_unchanged=(after == fingerprints), verdict=summary)
        legacy.write_json(output / "manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001
        manifest.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise
    LOG.info("B0 complete: %d rows, resolved_band=%s, %.1fs",
             len(rows), summary["resolved_band"], time.time() - started)


if __name__ == "__main__":
    main()
