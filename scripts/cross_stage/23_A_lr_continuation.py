#!/usr/bin/env python3
"""Route A convergence-control probe: paired continuation of the five affected
baselines from their Step-1 E100 checkpoints.

Two arms per seed, sharing the identical starting state (parameters, Adam
moments, RNG stream) and the identical epoch orders:

  control  keep the original constant learning rate (asks "does merely
           training longer help?")
  decay    piecewise-constant smaller learning rates (asks "does shrinking
           the step move the endpoint toward the Eq. 56 stationarity premise?")

Scope: baselines only.  No patient deletion, no attack training, no truth and
no damping scan.  The decay arm is a NEW trial recipe: its endpoint must never
be presented as "the original training converged", and any later LOO built on
it must continue baseline/full/dose01 each from their own E100 states with
this same frozen schedule.

The primary readout is the full-train-set eval CE + L2 gradient norm (the
objective whose Hessian Eq. 56 inverts), NOT the parameter update: with a tiny
learning rate the parameters move little even far from a stationary point.
A pre-declared exploratory gradient target (default 1e-3) is recorded, but a
single low value never certifies convergence.  Training runs with dropout
while the gradient is measured in eval mode; if the decay arm stalls at a
gradient far above target, report that objective mismatch instead of silently
changing dropout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conv = load_module("a20_curve_for_lr23", "20_A_convergence_curve.py")
legacy = conv.legacy
LOG = logging.getLogger("a23_lr_continuation")
WD = legacy.WD
SCHEMA = "pathway2_A_lr_continuation_v1"
ARMS = ("control", "decay")


def parse_schedule(text):
    """"20@1e-4,20@1e-5" -> [(20, 1e-4), (20, 1e-5)]; segments apply after the resume epoch."""
    segments = []
    for part in text.split(","):
        length, lr = part.strip().split("@")
        length, lr = int(length), float(lr)
        if length < 1 or not lr > 0:
            raise ValueError(f"Invalid schedule segment: {part}")
        segments.append((length, lr))
    return segments


def lr_for_offset(offset, base_lr, segments):
    """Learning rate for continuation epoch ``resume + offset`` (offset >= 1)."""
    if not segments:
        return base_lr
    position = 0
    for length, lr in segments:
        position += length
        if offset <= position:
            return lr
    raise ValueError(f"Offset {offset} beyond the declared schedule")


def source_files():
    return [HERE / name for name in (
        "05_end_to_end_patient_loo_pilot.py", "11_stage1_diagnostics.py",
        "stage1_diagnostic_core.py", "20_A_convergence_curve.py",
        "23_A_lr_continuation.py", "23_A_lr_report.py",
    )] + [REPO / "experiments/pathway2/frozen_baseline.json"]


def load_resume_state(step1_dir, spec, resume, pilot):
    """Validate the Step-1 run and return its epoch-``resume`` checkpoint payload."""
    manifest = legacy.read_json(step1_dir / "manifest.json")
    if manifest.get("schema") not in ("pathway2_A_convergence_curve_v1", "pathway2_A_convergence_curve_v2"):
        raise RuntimeError(f"{step1_dir}: not a Step-1 convergence run")
    if manifest.get("status") != "complete" or manifest.get("replay_exact") is not True:
        raise RuntimeError(f"{step1_dir}: Step-1 run is incomplete or not an exact replay of the original")
    legacy.assert_equal(manifest["run"], spec["name"], "Step-1 run name mismatch")
    legacy.assert_equal(int(manifest["seed"]), spec["seed"], "Step-1 seed mismatch")
    step1_epochs = int(manifest["extended_epochs"])
    if not int(manifest["original_epochs"]) < resume <= step1_epochs:
        raise RuntimeError(f"--resume_epoch {resume} is not a saved extension epoch of {step1_dir}")
    checkpoint = step1_dir / "checkpoints" / f"epoch{resume:03d}.pt"
    payload = legacy.load_payload(checkpoint, pilot)
    md = payload["metadata"]
    legacy.assert_equal(int(md["epoch"]), resume, "Checkpoint epoch mismatch")
    legacy.assert_equal(int(md["seed"]), spec["seed"], "Checkpoint seed mismatch")
    legacy.assert_equal(int(md["extended_epochs"]), step1_epochs, "Checkpoint horizon mismatch")
    return payload, checkpoint, step1_epochs, manifest


def continue_run(spec, arm, context, args, pilot, output, payload, step1_epochs):
    p = context["pargs"]
    X, y = context["X"], context["y"]
    resume, extra = int(args.resume_epoch), int(args.extra_epochs)
    total = resume + extra
    base_lr = float(p.shadow_lr)
    segments = parse_schedule(args.decay_schedule) if arm == "decay" else []
    if segments and sum(length for length, _ in segments) != extra:
        raise RuntimeError("--decay_schedule lengths must sum to --extra_epochs")

    orders = pilot.make_epoch_orders(spec["train"], total, spec["seed"] + 700000)
    md = payload["metadata"]
    if conv.order_sha(orders[:step1_epochs]) != md["epoch_order_sha256"]:
        raise RuntimeError("Continuation epoch orders do not extend the Step-1 orders")
    np.savez_compressed(output / "epoch_orders.npz", raw_index_order=orders)

    model = legacy.model_from_payload(payload, pilot, X, y).to(pilot.DEVICE)
    if not core.exact_tree({k: v.cpu() for k, v in model.state_dict().items()}, payload["state_dict"]):
        raise RuntimeError("Model state changed while moving to the device")
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=WD)
    optimizer.load_state_dict(md["optimizer_state"])
    torch.set_rng_state(md["rng_states"]["torch_cpu"])
    if md["rng_states"].get("torch_cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint carries CUDA RNG state but CUDA is unavailable; "
                               "the continuation would not be the paired trajectory")
        torch.cuda.set_rng_state_all(md["rng_states"]["torch_cuda"])
    fingerprint = core.rng_fingerprint()
    start_check = {"kind": "resume_state", "epoch": resume,
                   "rng_fingerprint_equal": fingerprint == md.get("rng_fingerprint"),
                   "optimizer_state_equal": core.exact_tree(optimizer.state_dict(), md["optimizer_state"]),
                   "rng_fingerprint": fingerprint}
    start_check["passed"] = start_check["rng_fingerprint_equal"] and start_check["optimizer_state_equal"]
    legacy.write_json(output / "resume_checks.json", [start_check])
    if not start_check["passed"]:
        raise RuntimeError(f"Resume state does not reproduce the Step-1 checkpoint: {start_check}")

    observer_spec = {**spec, "name": f"{spec['name']}_{arm}", "role": f"{spec['role']}_{arm}"}
    observer_args = argparse.Namespace(
        epochs=total, gradient_every=args.gradient_every, dropout_mc_every=args.dropout_mc_every,
        dropout_mc_reps=args.dropout_mc_reps, hvp_batch=args.hvp_batch, eval_batch=args.eval_batch)
    observer = conv.EpochObserver(observer_spec, context, observer_args, resume, output / "curve.csv")

    lr_log = []
    started = time.time()
    with core.preserve_rng_and_modes(model):
        observer(model, {"epoch": resume, "online_train_ce": None, "epoch_update_norm": None,
                         "relative_epoch_update": None, "lr": None, "arm": arm})
    for epoch in range(resume + 1, total + 1):
        lr = lr_for_offset(epoch - resume, base_lr, segments)
        for group in optimizer.param_groups:
            group["lr"] = lr
        lr_log.append({"epoch": epoch, "lr": lr})
        before = core.flat(model).clone()
        model.train()
        total_loss, seen = 0., 0
        for batch_idx in pilot._iter_batches(orders[epoch - 1], p.shadow_batch_size):
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=pilot.DEVICE)
            yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=pilot.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            per_example = F.cross_entropy(model(xb), yb, reduction="none")
            weights = torch.ones_like(per_example)
            loss = (weights * per_example).sum() / per_example.numel()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_idx)
            seen += len(batch_idx)
        update_norm = core.norm(core.flat(model) - before)
        row = {"epoch": epoch, "online_train_ce": total_loss / seen, "epoch_update_norm": update_norm,
               "relative_epoch_update": update_norm / core.norm(before) if core.norm(before) else None,
               "lr": lr, "arm": arm}
        with core.preserve_rng_and_modes(model):
            observer(model, row)
        if epoch in args.checkpoint_epochs:
            pilot.save_model(output / "checkpoints" / f"epoch{epoch:03d}.pt", model, {
                "run": observer_spec["name"], "role": observer_spec["role"], "seed": spec["seed"],
                "epoch": epoch, "arm": arm, "resume_epoch": resume, "total_epochs": total,
                "learning_rate_at_epoch": lr, "decay_schedule": args.decay_schedule if arm == "decay" else None,
                "base_lr": base_lr, "epoch_order_sha256": conv.order_sha(orders),
                "optimizer_state": optimizer.state_dict(), "rng_states": conv.rng_payload(),
                "rng_fingerprint": core.rng_fingerprint(), "source": SCHEMA})
    legacy.write_json(output / "lr_log.json", lr_log)
    return observer.records, {"resume_check": start_check, "final_rng_sha256": core.rng_fingerprint(),
                              "training_and_observation_seconds": time.time() - started,
                              "epoch_order_sha256": conv.order_sha(orders)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full_dir", required=True, help="original shadow*_full directory (read only)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--step1_root", required=True, help="completed convergence_E100 output root (read only)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--array_index", type=int, default=None, help="0..9 = seed index * 2 + arm (0=control,1=decay)")
    ap.add_argument("--run", default=None, help="e.g. shadow3_seed42")
    ap.add_argument("--arm", choices=ARMS, default=None)
    ap.add_argument("--resume_epoch", type=int, default=100)
    ap.add_argument("--extra_epochs", type=int, default=40)
    ap.add_argument("--decay_schedule", default="20@1e-4,20@1e-5",
                    help="segments 'LEN@LR,...' applied after the resume epoch; must sum to --extra_epochs")
    ap.add_argument("--gradient_target", type=float, default=1e-3,
                    help="pre-declared exploratory eval-gradient target; recorded, never a certificate")
    ap.add_argument("--checkpoint_epochs", type=int, nargs="*", default=None,
                    help="default: resume+20 and resume+extra")
    ap.add_argument("--gradient_every", type=int, default=1)
    ap.add_argument("--dropout_mc_every", type=int, default=5)
    ap.add_argument("--dropout_mc_reps", type=int, default=16)
    ap.add_argument("--hvp_batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--require_cuda", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this invocation")
    if args.extra_epochs < 2:
        raise ValueError("--extra_epochs must be at least 2")
    if args.checkpoint_epochs is None:
        args.checkpoint_epochs = sorted({args.resume_epoch + args.extra_epochs // 2,
                                         args.resume_epoch + args.extra_epochs})
    if min(args.gradient_every, args.dropout_mc_every, args.hvp_batch, args.eval_batch) < 1 \
            or args.dropout_mc_reps < 2:
        raise ValueError("iteration and batch settings must be positive")
    full = Path(args.full_dir).resolve()
    step1_root = Path(args.step1_root).resolve()
    out_root = Path(args.out_root).resolve()
    for protected in (full, step1_root, REPO):
        if out_root == protected or out_root.is_relative_to(protected) or protected.is_relative_to(out_root):
            raise RuntimeError(f"Output root overlaps a protected directory: {protected}")

    def git(*cmd):
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *cmd], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    actual_commit = git("rev-parse", "HEAD")
    expected_commit = os.environ.get("OCT_A_CODE")
    if expected_commit and actual_commit != expected_commit:
        raise RuntimeError(f"Queued code commit {expected_commit} changed to {actual_commit}")

    score = legacy.module_from_path("score07_a23", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    ns = argparse.Namespace(full_dir=str(full), dose_dir=None, mode="convergence", data_dir=args.data_dir,
                            seeds=json.loads((full / "experiment_config.json").read_text())["args"]["seeds"])
    context = legacy.load_context(ns, pilot)
    context["splits"] = legacy.read_json(full / "splits" / "fresh_patient_split.json")
    specs = conv.run_specs(context)
    if args.array_index is not None:
        if args.run is not None or args.arm is not None:
            raise ValueError("Pass either --array_index or --run/--arm, not both")
        if not 0 <= args.array_index < 2 * len(specs):
            raise ValueError(f"--array_index must be in 0..{2 * len(specs) - 1}")
        spec = specs[args.array_index // 2]
        arm = ARMS[args.array_index % 2]
    else:
        if args.run is None or args.arm is None:
            raise ValueError("Pass --array_index, or both --run and --arm")
        matches = [s for s in specs if s["name"] == args.run]
        if len(matches) != 1:
            raise ValueError(f"Unknown run {args.run}; choose from {[s['name'] for s in specs]}")
        spec, arm = matches[0], args.arm

    step1_dir = step1_root / spec["name"]
    payload, checkpoint, step1_epochs, step1_manifest = load_resume_state(
        step1_dir, spec, int(args.resume_epoch), pilot)
    output = out_root / f"{spec['name']}_{arm}"
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(path): legacy.sha(path) for path in
              [checkpoint, step1_dir / "manifest.json",
               full / "experiment_config.json", full / "splits/fresh_patient_split.json"]}
    legacy.write_json(output / "input_sha256.json", inputs)
    source_sha = {str(path.relative_to(REPO)): legacy.sha(path) for path in source_files()}
    legacy.write_json(output / "source_sha256.json", source_sha)
    dataset_before = conv.dataset_fingerprint(context["X"], context["y"])
    legacy.write_json(output / "dataset_fingerprint.json", dataset_before)
    manifest = {
        "schema": SCHEMA, "status": "running", "run": f"{spec['name']}_{arm}", "base_run": spec["name"],
        "arm": arm, "seed": spec["seed"], "resume_epoch": int(args.resume_epoch),
        "extra_epochs": int(args.extra_epochs), "total_epochs": int(args.resume_epoch + args.extra_epochs),
        "base_lr": float(context["pargs"].shadow_lr),
        "decay_schedule": args.decay_schedule if arm == "decay" else None,
        "gradient_target_exploratory": args.gradient_target,
        "step1_root": str(step1_root), "step1_extended_epochs": step1_epochs,
        "step1_git_commit": step1_manifest.get("git_commit"),
        "args": vars(args), "weight_decay": WD,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "device": str(pilot.DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": actual_commit, "queued_git_commit": expected_commit,
        "source_sha256": source_sha,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scope": "Paired convergence-control probe on baselines only. The decay arm is a new trial recipe "
                 "with its own budget; it never redefines the original training, the frozen truth, or E*. "
                 "A low gradient here is a diagnostic, not a convergence certificate, and no LOO/attack/"
                 "score conclusions follow from this run alone.",
    }
    legacy.write_json(output / "manifest.json", manifest)
    started = time.time()
    try:
        rows, extra_info = continue_run(spec, arm, context, args, pilot, output, payload, step1_epochs)
        if not all(legacy.sha(Path(path)) == value for path, value in inputs.items()):
            raise RuntimeError("An input changed during the run")
        if {str(path.relative_to(REPO)): legacy.sha(path) for path in source_files()} != source_sha:
            raise RuntimeError("A source file changed during the run")
        if conv.dataset_fingerprint(context["X"], context["y"]) != dataset_before:
            raise RuntimeError("Loaded dataset arrays changed during the run")
        final = rows[-1]
        manifest.update(extra_info, status="complete", elapsed_seconds=time.time() - started,
                        n_curve_rows=len(rows), input_files_unchanged=True, source_files_unchanged=True,
                        dataset_arrays_unchanged=True,
                        final_row={k: v for k, v in final.items() if not isinstance(v, (dict, list))})
        legacy.write_json(output / "manifest.json", manifest)
        LOG.info("COMPLETE %s_%s -> %s", spec["name"], arm, output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
