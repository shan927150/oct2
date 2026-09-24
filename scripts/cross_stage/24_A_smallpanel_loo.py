#!/usr/bin/env python3
"""Route A small-panel paired LOO continuation (meeting plan part B).

Question the lr-probe cannot answer: once the training tail is stabilised by
learning-rate decay, does the baseline-vs-deletion difference (the real Δθ and
its dose response) actually become more local/linear under this schedule?

Design (one array task = one cell):

  panel     the pre-registered core patients 807 (DME) and 2085 (DRUSEN)
  seeds     all five (42..46)
  dose      full (deletion_weight 1.0) and dose01 (0.1)
  arm       control (constant lr) and decay (lr schedule after E100)
  => 2 patients x 5 seeds x 2 doses x 2 arms = 40 continuation runs.

Every condition trains its OWN trajectory: the fixed-mask deletion is applied
from epoch 1, exactly as the original truth, and the learning-rate schedule
only changes AFTER the resume epoch.  Because make_epoch_orders is prefix
stable and the arithmetic matches the pilot, the first 100 epochs reproduce
the original E100 truth endpoint bitwise; that identity is checked against the
frozen l3_at_E100 truth model, so E101..E140 is its exact continuation.

Guardrails: baselines come from the lr-probe (do NOT restart full/dose from a
baseline checkpoint - that would be a different, warm-start intervention).  No
attack retraining, no damping, no overwrite of E95/E100.  The decay arm is a
new trial recipe; the endpoint is never called "the original training".
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
import route_a_smallpanel as common

_spec = importlib.util.spec_from_file_location("stage1_legacy_a24", HERE / "11_stage1_diagnostics.py")
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)
_cspec = importlib.util.spec_from_file_location("a20_curve_for_a24", HERE / "20_A_convergence_curve.py")
conv = importlib.util.module_from_spec(_cspec)
sys.modules[_cspec.name] = conv
_cspec.loader.exec_module(conv)

LOG = logging.getLogger("a24_smallpanel")
WD = legacy.WD
SCHEMA = common.SCHEMA
ARMS = ("control", "decay")
DOSES = {"full": 1.0, "dose01": 0.1}


def parse_schedule(text):
    segments = []
    for part in text.split(","):
        length, lr = part.strip().split("@")
        length, lr = int(length), float(lr)
        if length < 1 or not lr > 0:
            raise ValueError(f"Invalid schedule segment: {part}")
        segments.append((length, lr))
    return segments


def lr_for_epoch(epoch, resume, base_lr, segments):
    """Constant base_lr for epoch <= resume; segments apply after resume."""
    if epoch <= resume or not segments:
        return base_lr
    offset, position = epoch - resume, 0
    for length, lr in segments:
        position += length
        if offset <= position:
            return lr
    raise ValueError(f"Epoch {epoch} beyond declared schedule")


def source_files():
    return [HERE / name for name in (
        "05_end_to_end_patient_loo_pilot.py", "07_cross_stage_score_ladder.py", "11_stage1_diagnostics.py",
        "stage1_diagnostic_core.py", "20_A_convergence_curve.py",
        "21_A_truth_replay.py", "24_A_smallpanel_loo.py", "24_A_smallpanel_report.py",
        "24_A_smallpanel_control.py", "route_a_smallpanel.py",
    )] + [REPO / "experiments/pathway2/frozen_baseline.json"]


def cells(context, panel):
    p = context["pargs"]
    out = []
    for seed in p.seeds:
        for patient in panel:
            for dose in ("full", "dose01"):
                for arm in ARMS:
                    out.append({"seed": int(seed), "patient": patient, "dose": dose, "arm": arm})
    return out


def core_panel(context, patient_ids):
    by_id = {int(pat["patient_id"]): pat for pat in context["patients"]}
    missing = [pid for pid in patient_ids if pid not in by_id]
    if missing:
        raise ValueError(f"Requested panel patients not in frozen panel: {missing}")
    return [by_id[pid] for pid in patient_ids]


def weighted_objective_grad(model, X, y, indices, excluded_t, weight, batch, wd):
    """grad of the model's OWN training objective: (1/N) sum_i w_i CE_i + wd/2||theta||^2,
    eval mode (dropout off).  w_i = 1, except (1 - weight) on the deleted patient's rows
    (0 for full deletion).  This is the correct stationarity readout for a LOO model;
    the unweighted eval gradient would answer a different question."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    dt = params[0].dtype
    g = torch.zeros(sum(p.numel() for p in params), device=device, dtype=dt)
    n = len(indices)
    for start in range(0, n, batch):
        idx = np.asarray(indices[start:start + batch], dtype=np.int64)
        xb = torch.as_tensor(X[idx], device=device, dtype=dt)
        yb = torch.as_tensor(y[idx], device=device, dtype=torch.long)
        per = F.cross_entropy(model(xb), yb, reduction="none")
        w = torch.ones_like(per)
        if len(excluded_t):
            idx_t = torch.as_tensor(idx, dtype=torch.int64, device=device)
            removed = torch.isin(idx_t, excluded_t)
            w = torch.where(removed, w * (1.0 - weight), w)
        loss = (w * per).sum() / n
        g += torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, params)])
    g = g + wd * core.flat(model)
    return core.norm(g)


def save_interface(context, pilot, cfg, spec, output):
    """Paired affected-shadow predictions on unchanged member/nonmember rows."""
    sid, seed, arm = 3, spec['seed'], spec['arm']
    split = context['splits']['shadow_models'][sid]
    rows = np.asarray(split['train_idx'] + split['test_idx'], dtype=np.int64)
    values = {'raw_indices': rows, 'n_member': np.asarray(len(split['train_idx']))}
    for epoch in (100, 140):
        base = (Path(cfg['e100_root']) / 'shadow3_full/checkpoints' / f'shadow_3_baseline_seed{seed}.pt'
                if epoch == 100 else Path(cfg['probe_root']) / f'shadow3_seed{seed}_{arm}/checkpoints/epoch140.pt')
        for label, path in [('baseline', base), ('loo', output / f'checkpoints/epoch{epoch:03d}.pt')]:
            payload = legacy.load_payload(path, pilot)
            if label == 'baseline' and epoch == 140:
                common.check_baseline_metadata(payload, seed, arm, cfg['probe_facts'][f'{seed}/{arm}'])
            model = legacy.model_from_payload(payload, pilot, context['X'], context['y']).to(pilot.DEVICE)
            with core.preserve_rng_and_modes(model):
                values[f'{label}_{epoch}'] = pilot.get_predictions(model, context['X'][rows])
            del model
    np.savez_compressed(output / 'interface.npz', **values)


def train_with_schedule(context, pilot, spec, args, output):
    """Fixed-mask deletion training to E140 with a per-epoch learning rate.

    Mirrors pilot.train_classifier_from_orders arithmetic so the first
    resume_epoch epochs reproduce the original truth trajectory bitwise.
    """
    p = context["pargs"]
    X, y = context["X"], context["y"]
    seed, dose, arm = spec["seed"], spec["dose"], spec["arm"]
    resume, total = int(args.resume_epoch), int(args.total_epochs)
    base_lr = float(p.shadow_lr)
    segments = parse_schedule(args.decay_schedule) if arm == "decay" else []
    if segments and resume + sum(l for l, _ in segments) != total:
        raise RuntimeError("decay schedule + resume must sum to total_epochs")
    excluded = np.asarray(spec["patient"]["raw_indices"], dtype=np.int64)
    weight = DOSES[dose]

    train_idx = context["train"]
    orders = pilot.make_epoch_orders(train_idx, total, seed + 700000)
    if not np.array_equal(orders[:resume], pilot.make_epoch_orders(train_idx, resume, seed + 700000)):
        raise RuntimeError("epoch-order generator is not prefix stable")

    pilot.seed_everything(seed, p.deterministic)
    n_classes = int(np.max(y)) + 1
    model = pilot.build_model("cnn", X.shape[1], 128, n_classes).to(pilot.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=WD)
    excluded_t = torch.as_tensor(np.sort(excluded), dtype=torch.int64, device=pilot.DEVICE)

    curve, checks = [], []
    e100_ref = spec["e100_ref"]

    def observe(epoch, lr):
        with core.preserve_rng_and_modes(model):
            info, _ = core.objective_snapshot(model, X, y, train_idx, args.hvp_batch, WD, True)
            train_grad = weighted_objective_grad(model, X, y, train_idx, excluded_t, weight, args.hvp_batch, WD)
        row = {"seed": seed, "patient_id": spec["patient"]["patient_id"], "dose": dose, "arm": arm,
               "epoch": epoch, "lr": lr, "eval_ce": info["ce"], "eval_objective": info["objective"],
               # primary stationarity readout: gradient of THIS model's own (deletion-weighted) objective
               "weighted_eval_grad_norm": train_grad,
               # secondary, for comparison across conditions: gradient of the unweighted eval objective
               "unweighted_eval_grad_norm": info["grad_norm"], "param_norm": core.norm(core.flat(model))}
        curve.append(row)
        legacy.write_csv(output / "curve.csv", curve)

    final_epoch = resume if args.stop_at_resume else total
    for epoch in range(1, final_epoch + 1):
        lr = lr_for_epoch(epoch, resume, base_lr, segments)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        for batch_idx in pilot._iter_batches(orders[epoch - 1], p.shadow_batch_size):
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=pilot.DEVICE)
            yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=pilot.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            per_example = F.cross_entropy(model(xb), yb, reduction="none")
            weights = torch.ones_like(per_example)
            if weight != 1.0:
                idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=pilot.DEVICE)
                removed = torch.isin(idx_t, excluded_t)
                weights = torch.where(removed, weights * (1.0 - weight), weights)
            elif len(excluded):
                idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=pilot.DEVICE)
                removed = torch.isin(idx_t, excluded_t)
                weights = torch.where(removed, torch.zeros_like(weights), weights)
            loss = (weights * per_example).sum() / per_example.numel()
            loss.backward()
            optimizer.step()
        if epoch == resume:
            ref = legacy.load_payload(e100_ref, pilot)
            fp = core.rng_fingerprint()
            ref_fp = ref.get("metadata", {}).get("metrics", {}).get("post_train_rng_sha256")
            check = {"epoch": epoch, "kind": "e100_truth_endpoint", "reference": str(e100_ref),
                     "parameters_equal": core.exact_tree(model.state_dict(), ref["state_dict"]),
                     "post_train_rng_sha256": fp, "reference_post_train_rng_sha256": ref_fp,
                     "rng_fingerprint_equal": (ref_fp is not None and fp == ref_fp),
                     "reference_has_rng_fingerprint": ref_fp is not None}
            replay = legacy.module_from_path("truth_hash_a24", HERE / "21_A_truth_replay.py")
            actual_hash = replay.tree_sha(optimizer.state_dict())
            cert = legacy.read_json(e100_ref.parent.parent / "route_a_epoch50_replay_checks.json")
            expected_hash = common.optimizer_reference(cert, seed, spec["patient"]["patient_id"],
                                                       p.affected_shadow, resume)
            recorded = np.load(e100_ref.parent.parent / f"stage1_order_seed{seed}.npz")["raw_index_order"]
            check.update(optimizer_sha256=actual_hash, reference_optimizer_sha256=expected_hash,
                         optimizer_hash_equal=actual_hash == expected_hash,
                         recorded_orders_equal=np.array_equal(orders[:resume], recorded))
            check["passed"] = all(check[k] for k in ("parameters_equal", "rng_fingerprint_equal",
                                                      "optimizer_hash_equal", "recorded_orders_equal"))
            checks.append(check)
            legacy.write_json(output / "replay_checks.json", checks)
            if not check["passed"]:
                raise RuntimeError(f"E{resume} does not reproduce the frozen truth endpoint {e100_ref} "
                                   f"(params_equal={check['parameters_equal']}, "
                                   f"rng_equal={check['rng_fingerprint_equal']}); "
                                   "the continuation would not describe the original run")
        if epoch in (resume, total):
            observe(epoch, lr)
            pilot.save_model(output / "checkpoints" / f"epoch{epoch:03d}.pt", model, {
                "seed": seed, "patient_id": spec["patient"]["patient_id"], "dose": dose, "arm": arm,
                "deletion_weight": weight, "excluded_indices": sorted(int(v) for v in excluded),
                "epoch": epoch, "resume_epoch": resume, "total_epochs": total,
                "learning_rate_at_epoch": lr, "decay_schedule": args.decay_schedule if arm == "decay" else None,
                # full continuation state so a later stage can resume from this checkpoint exactly
                "optimizer_state": optimizer.state_dict(), "rng_states": conv.rng_payload(),
                "rng_fingerprint": core.rng_fingerprint(), "epoch_order_sha256": conv.order_sha(orders),
                "base_lr": base_lr, "source": SCHEMA})
        elif epoch % args.observe_every == 0:
            observe(epoch, lr)
    if not any(c["kind"] == "e100_truth_endpoint" for c in checks):
        raise RuntimeError("E100 replay check never ran")
    return curve, checks


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full_dir", required=True, help="original shadow*_full (read-only; split/panel/orders)")
    ap.add_argument("--config", help="frozen v3 attempt configuration; required for CUDA runs")
    ap.add_argument("--stop_at_resume", action="store_true", help="real-data smoke: verify and save E100, then stop")
    ap.add_argument("--e100_full_dir", required=True, help="l3_at_E100 .../shadow*_full (frozen E100 truth)")
    ap.add_argument("--e100_dose_dir", required=True, help="l3_at_E100 .../shadow*_dose01 (frozen E100 truth)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--array_index", type=int, default=None)
    ap.add_argument("--panel", type=int, nargs="+", default=[807, 2085])
    ap.add_argument("--resume_epoch", type=int, default=100)
    ap.add_argument("--total_epochs", type=int, default=140)
    ap.add_argument("--decay_schedule", default="20@1e-4,20@1e-5")
    ap.add_argument("--observe_every", type=int, default=5)
    ap.add_argument("--hvp_batch", type=int, default=64)
    ap.add_argument("--list_cells", action="store_true")
    ap.add_argument("--require_cuda", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this invocation")
    if args.require_cuda and not args.config:
        raise ValueError("CUDA runs require a frozen submission configuration")
    if min(args.hvp_batch, args.observe_every, args.resume_epoch) < 1 or args.total_epochs <= args.resume_epoch:
        raise ValueError("Invalid batch/observation/epoch setting")
    full = Path(args.full_dir).resolve()
    e100_full = Path(args.e100_full_dir).resolve()
    e100_dose = Path(args.e100_dose_dir).resolve()
    out_root = Path(args.out_root).resolve()
    cfg = None
    if args.config:
        cfgroot, cfg = common.load_config(Path(args.config).parent)
        expected_out = cfgroot / "smoke" if args.stop_at_resume else cfgroot
        actual = [str(full), str(e100_full), str(e100_dose), str(Path(args.data_dir).resolve()),
                  str(out_root), args.resume_epoch, args.total_epochs, args.decay_schedule, args.panel]
        expected = [cfg["full_dir"], str(Path(cfg["e100_root"]) / "shadow3_full"),
                    str(Path(cfg["e100_root"]) / "shadow3_dose01"), cfg["data_dir"], str(expected_out),
                    100, 140, common.SCHEDULE, common.PATIENTS]
        if actual != expected:
            raise ValueError("Runner arguments differ from the frozen submission")
    for protected in (full, e100_full, e100_dose, REPO):
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

    score = legacy.module_from_path("score07_a24", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    ns = argparse.Namespace(full_dir=str(full), dose_dir=None, mode="convergence", data_dir=args.data_dir,
                            seeds=json.loads((full / "experiment_config.json").read_text())["args"]["seeds"])
    context = legacy.load_context(ns, pilot)
    context["splits"] = legacy.read_json(full / "splits" / "fresh_patient_split.json")
    panel = core_panel(context, args.panel)
    affected = int(context["pargs"].affected_shadow)
    all_cells = cells(context, panel)
    if args.list_cells:
        for i, c in enumerate(all_cells):
            print(i, c["seed"], c["patient"]["patient_id"], c["dose"], c["arm"])
        return
    if args.array_index is None or not 0 <= args.array_index < len(all_cells):
        raise ValueError(f"--array_index must be in 0..{len(all_cells) - 1}")
    spec = all_cells[args.array_index]
    pid = spec["patient"]["patient_id"]
    ref_dir = e100_full if spec["dose"] == "full" else e100_dose
    spec["e100_ref"] = ref_dir / "checkpoints" / f"shadow_{affected}_seed{spec['seed']}_patient{pid}.pt"
    if not spec["e100_ref"].is_file():
        raise RuntimeError(f"Missing frozen E100 truth endpoint: {spec['e100_ref']}")
    certpath = ref_dir / "route_a_epoch50_replay_checks.json"
    common.optimizer_reference(legacy.read_json(certpath), spec["seed"], pid, affected, args.resume_epoch)

    name = f"seed{spec['seed']}_patient{pid}_{spec['dose']}_{spec['arm']}"
    output = out_root / name
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(p): legacy.sha(p) for p in [
        spec["e100_ref"], full / "experiment_config.json", full / "splits/fresh_patient_split.json",
        full / "selected_patients.json", certpath, ref_dir / f"stage1_order_seed{spec['seed']}.npz"]}
    legacy.write_json(output / "input_sha256.json", inputs)
    source_sha = {str(p.relative_to(REPO)): legacy.sha(p) for p in source_files()}
    legacy.write_json(output / "source_sha256.json", source_sha)
    dataset_before = conv.dataset_fingerprint(context["X"], context["y"])
    if cfg and dataset_before != cfg["dataset_fingerprint"]:
        raise ValueError("Loaded OCT dataset differs from the paired probe baseline")
    legacy.write_json(output / "dataset_fingerprint.json", dataset_before)
    manifest = {
        "schema": SCHEMA, "status": "running", "cell": name, **{k: spec[k] for k in ("seed", "dose", "arm")},
        "patient_id": pid, "deletion_weight": DOSES[spec["dose"]], "affected_shadow": affected,
        "resume_epoch": args.resume_epoch, "total_epochs": args.total_epochs,
        "decay_schedule": args.decay_schedule if spec["arm"] == "decay" else None,
        "e100_reference": str(spec["e100_ref"]), "panel": args.panel,
        "all_cells": [f"seed{c['seed']}_patient{c['patient']['patient_id']}_{c['dose']}_{c['arm']}" for c in all_cells],
        "base_lr": float(context["pargs"].shadow_lr), "weight_decay": WD,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "device": str(pilot.DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": actual_commit, "queued_git_commit": expected_commit, "source_sha256": source_sha,
        "config_sha256": common.sha(args.config) if args.config else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scope": "Small-panel paired LOO continuation. Each condition follows its own trajectory from epoch 1 and "
                 "only changes lr after the resume epoch; the E100 endpoint is verified bitwise against the frozen "
                 "truth. Baselines come from the lr-probe. New trial recipe on the decay arm; not the original truth.",
    }
    legacy.write_json(output / "manifest.json", manifest)
    started = time.time()
    try:
        curve, checks = train_with_schedule(context, pilot, spec, args, output)
        if not all(legacy.sha(Path(k)) == v for k, v in inputs.items()):
            raise RuntimeError("An input changed during the run")
        if {str(p.relative_to(REPO)): legacy.sha(p) for p in source_files()} != source_sha:
            raise RuntimeError("A source file changed during the run")
        if conv.dataset_fingerprint(context["X"], context["y"]) != dataset_before:
            raise RuntimeError("Loaded dataset arrays changed during the run")
        if cfg:
            common.load_config(Path(args.config).parent)
            if not args.stop_at_resume:
                md = legacy.load_payload(output / "checkpoints/epoch140.pt", pilot)["metadata"]
                fact = cfg["probe_facts"][f"{spec['seed']}/{spec['arm']}"]
                if md["epoch_order_sha256"] != fact["epoch_order_sha256"] or md["rng_fingerprint"] != fact["final_rng_sha256"]:
                    raise ValueError("LOO and probe baseline did not use the same order/RNG stream")
                save_interface(context, pilot, cfg, spec, output)
        manifest.update(status="smoke_complete" if args.stop_at_resume else "complete", elapsed_seconds=time.time() - started,
                        replay_e100_passed=all(c["passed"] for c in checks),
                        final_row={k: v for k, v in curve[-1].items() if not isinstance(v, (dict, list))},
                        input_files_unchanged=True, source_files_unchanged=True, dataset_arrays_unchanged=True)
        manifest["checkpoint_sha256"] = {p.name: legacy.sha(p) for p in sorted((output / "checkpoints").glob("*.pt"))}
        manifest['artifact_sha256'] = {p.name: legacy.sha(p) for p in
            [output / 'curve.csv', output / 'replay_checks.json', output / 'interface.npz'] if p.exists()}
        legacy.write_json(output / "manifest.json", manifest)
        LOG.info("COMPLETE %s -> %s", name, output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
