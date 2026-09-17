#!/usr/bin/env python3
"""B0 orchestration only: reuse 05's unchanged training function and frozen orders.

One original-algorithm replay per seed, then two patients x four doses. Outputs
contain Stage-1 perturbations and interfaces, NOT new cross-stage J10 truth.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core
spec = importlib.util.spec_from_file_location("b0_train_e0", HERE / "13_e0_residual_decomposition.py")
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)
DOSES = {"dose003": .03, "dose001": .01, "dose0003": .003, "dose0001": .001}


def train_cells(args, context, legacy, pilot, output):
    p, X, y = context["pargs"], context["X"], context["y"]
    seed, shadow = args.seed, p.affected_shadow
    orders = legacy.verify_orders(context, seed)
    splits = legacy.read_json(context["full"] / "splits/fresh_patient_split.json")
    split = splits["shadow_models"][shadow]
    patients = {int(v["patient_id"]): v for v in context["patients"]}
    if set(args.patients) - patients.keys():
        raise RuntimeError("Requested patient is not in the frozen panel")
    base_path = context["full"] / "checkpoints" / f"shadow_{shadow}_baseline_seed{seed}.pt"
    base = legacy.load_payload(base_path, pilot)
    rng = base["metadata"]["metrics"]["post_train_rng_sha256"]
    milestones = sorted(set(p.save_epoch_checkpoints))
    if p.shadow_epochs not in milestones:
        raise RuntimeError("Frozen baseline must have a final optimizer/RNG checkpoint")
    common = dict(X=X, y=y, train_orders=orders, eval_indices=split["test_idx"],
                  seed=seed, n_hidden=128, lr=p.shadow_lr, batch_size=p.shadow_batch_size,
                  weight_decay=context["config"]["oct_config"]["target_l2"],
                  deterministic=p.deterministic, deletion_mode=p.deletion_mode)

    # No new baseline: this replay must reproduce the original endpoint and saved
    # Adam/RNG states before the first perturbed training is allowed to start.
    replay_dir = output / "replay"
    model, metrics = pilot.train_classifier_from_orders(
        **common, epoch_checkpoint_dir=replay_dir, epoch_checkpoints=milestones)
    if not core.exact_tree(model.state_dict(), base["state_dict"]):
        raise RuntimeError("B0 no-op parameters differ from original baseline")
    legacy.assert_equal(metrics["post_train_rng_sha256"], rng, "B0 no-op RNG mismatch")
    for epoch in milestones:
        reference = legacy.load_payload(context["full"] / "checkpoints" /
                    f"baseline_seed{seed}_epochs/epoch{epoch:03d}.pt", pilot)
        replay = legacy.load_payload(replay_dir / f"epoch{epoch:03d}.pt", pilot)
        for key in ("optimizer_state", "rng_states"):
            if not core.exact_tree(reference["metadata"][key], replay["metadata"][key]):
                raise RuntimeError(f"B0 no-op epoch {epoch}: {key} mismatch")
        if not core.exact_tree(reference["state_dict"], replay["state_dict"]):
            raise RuntimeError(f"B0 no-op epoch {epoch}: parameters mismatch")

    # Read exact original query rows/probabilities. The model's eval predictions
    # must match them, preventing a row/order/dataset change from entering B0.
    interfaces = {}
    for pid in args.patients:
        with np.load(context["full"] / "runs" / f"seed{seed}_patient{pid}_interface.npz") as z:
            interfaces[pid] = {k: z[k].copy() for k in z.files}
        itf = interfaces[pid]
        expected_rows = np.concatenate([split["train_idx"], split["test_idx"]])
        if not np.array_equal(itf["raw_index"], expected_rows):
            raise RuntimeError("Original interface row order differs from frozen split")
        pred = pilot.get_predictions(model, X[itf["raw_index"]])
        if not np.array_equal(pred, itf["p_baseline"]):
            raise RuntimeError("B0 no-op interface is not bitwise equal to original")
    noop = {"seed": seed, "parameters_exact": True, "optimizer_rng_epochs_exact": milestones,
            "interface_exact": True, "displacement_norm": 0.0,
            "post_train_rng_sha256": rng, "original_baseline_sha256": legacy.sha(base_path)}
    legacy.write_json(output / "noop_replay.json", noop)

    for condition, alpha in DOSES.items():
        root = output / f"shadow{shadow}_{condition}"
        (root / "checkpoints").mkdir(parents=True, exist_ok=False)
        cfg = copy.deepcopy(context["config"])
        cfg["args"].update(seeds=[seed], deletion_weight=alpha, output_dir=str(root),
                            data_dir=args.data_dir, loo_patients=list(args.patients))
        cfg["scope"] = "stage1_only_probe; no Stage-2 retraining or J10/J11 endpoint"
        legacy.write_json(root / "experiment_config.json", cfg)
        for name in ("selected_patients.json", "splits/fresh_patient_split.json",
                     f"stage1_order_seed{seed}.npz"):
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(context["full"] / name, destination)
        shutil.copyfile(base_path, root / "checkpoints" / base_path.name)
        legacy.write_json(root / "noop_replay.json", noop)
        for pid in args.patients:
            patient = patients[pid]
            idx = list(map(int, patient["raw_indices"]))
            logging.info("B0 seed=%d patient=%d alpha=%g epochs=%d", seed, pid, alpha, len(orders))
            perturbed, pm = pilot.train_classifier_from_orders(
                **common, excluded_indices=idx, deletion_weight=alpha,
                epoch_checkpoint_dir=root / "checkpoints" / f"seed{seed}_patient{pid}_epochs",
                epoch_checkpoints=[p.shadow_epochs])
            legacy.assert_equal(pm["post_train_rng_sha256"], rng, "B0 fixed_mask RNG mismatch")
            pilot.save_model(root / "checkpoints" / f"shadow_{shadow}_seed{seed}_patient{pid}.pt",
                             perturbed, {"seed": seed, "excluded_indices": idx, "deletion_weight": alpha,
                                         "removal_window": None, "deletion_mode": "fixed_mask", "metrics": pm})
            itf = interfaces[pid]
            saved = {k: v for k, v in itf.items() if k not in ("p_loo", "loo_membership")}
            saved["p_loo"] = pilot.get_predictions(perturbed, X[itf["raw_index"]])
            saved["loo_membership"] = itf["baseline_membership"].copy()
            (root / "runs").mkdir(exist_ok=True)
            np.savez_compressed(root / "runs" / f"seed{seed}_patient{pid}_interface.npz", **saved)
        legacy.write_json(root / "experiment_summary.json", {
            "status": "stage1_probe_complete", "scope": cfg["scope"],
            "split_sha256": context["summary"]["split_sha256"], "seeds": [seed],
            "completed_patients": args.patients, "noop_replay": noop,
            "effective_float32_deletion_weight": float(1.0 - np.float32(1.0-alpha))})
    return noop


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--dose_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--patients", type=int, nargs="+", default=[807, 2085])
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if len(args.patients) != len(set(args.patients)):
        raise ValueError("Duplicate patients")
    output = Path(args.out_dir).resolve()
    for source in (args.full_dir, args.dose_dir, args.data_dir, HERE.parents[1]):
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise RuntimeError("B0 output overlaps protected inputs/code")
    output.mkdir(parents=True, exist_ok=False)
    legacy, _, pilot = e0.load_modules()
    started, inputs = time.time(), {}
    manifest = {"status": "running", "args": vars(args), "torch": torch.__version__,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
                "training_function": "05.train_classifier_from_orders (unchanged)",
                "n_training_runs": 1 + len(DOSES)*len(args.patients), "stage2_retrained": False}
    legacy.write_json(output / "manifest.json", manifest)
    try:
        ctx_args = argparse.Namespace(full_dir=args.full_dir, dose_dir=args.dose_dir,
                    mode="damping", data_dir=args.data_dir, seeds=[args.seed])
        # Enumerate input files before reading model/data values for computation.
        v11 = legacy.module_from_path("b0_train_v11", HERE / "12_stage1_diagnostics_v11.py")
        context = v11.metadata_context(ctx_args, pilot)
        paths = legacy.input_paths(context, ctx_args)
        paths += [context["full"] / "checkpoints" / f"baseline_seed{args.seed}_epochs/epoch{e:03d}.pt"
                  for e in context["pargs"].save_epoch_checkpoints]
        paths += [context["full"] / "runs" / f"seed{args.seed}_patient{p}_interface.npz" for p in args.patients]
        inputs = {str(p): legacy.sha(p) for p in paths}
        legacy.write_json(output / "input_sha256.json", inputs)
        context = legacy.load_context(ctx_args, pilot)
        manifest["loaded_data_sha256"] = {
            key: hashlib.sha256(memoryview(context[key]).cast("B")).hexdigest() for key in ("X", "y")}
        manifest["shadow_epochs"] = context["pargs"].shadow_epochs
        manifest["noop"] = train_cells(args, context, legacy, pilot, output)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        changed = [p for p, h in inputs.items() if not Path(p).is_file() or legacy.sha(Path(p)) != h]
        manifest.update(elapsed_seconds=time.time()-started, changed_inputs=changed,
                        input_files_unchanged=not changed)
        if changed:
            manifest["status"] = "failed_inputs_changed"
        legacy.write_json(output / "manifest.json", manifest)
        if changed:
            raise RuntimeError("Original B0 input files changed")


if __name__ == "__main__":
    main()
