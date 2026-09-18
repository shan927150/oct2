#!/usr/bin/env python3
"""Route A, meeting 2026-09-18, step 1: train each Stage 1 model longer and log every epoch.

Question: after how many epochs does every Stage 1 training curve flatten, and
does "flat loss" coincide with the Eq. 56 premise, i.e. a small full-batch
gradient of the eval objective whose Hessian the static formula inverts?

Scope: descriptive curves plus Hessian Ritz endpoints at saved epochs.  No
patient deletion, no attack training, no truth replacement.  The original
50-epoch results are read only.

Exactness: rows 0..T0-1 of ``make_epoch_orders(train, E_max, seed + 700000)``
equal the original T0-epoch order (the generator is consumed row by row), the
learning rate is constant and the RNG stream is not touched by the observer.
The first T0 epochs are checked bitwise (parameters, Adam state, RNG) against
the frozen checkpoints, so epochs T0+1..E_max are the exact continuation of
the original training.  Consequently, training any condition from scratch
for E epochs equals continuing its original run with its own Adam moments.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage1_legacy_a20", HERE / "11_stage1_diagnostics.py")
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)

LOG = logging.getLogger("a20_convergence")
WD = legacy.WD
SCHEMA = "pathway2_A_convergence_curve_v1"


# ---------------------------------------------------------------------------
# run specification
# ---------------------------------------------------------------------------

def run_specs(context):
    """Every Stage 1 model of the formal design, in a fixed array order."""
    p, splits, full = context["pargs"], context["splits"], context["full"]
    a = int(p.affected_shadow)
    ck = full / "checkpoints"
    specs = []
    for seed in p.seeds:
        specs.append({
            "name": f"shadow{a}_seed{int(seed)}", "role": "affected_shadow", "seed": int(seed),
            "train": splits["shadow_models"][a]["train_idx"], "heldout": splits["shadow_models"][a]["test_idx"],
            "reference": ck / f"shadow_{a}_baseline_seed{int(seed)}.pt",
            "epoch_dir": ck / f"baseline_seed{int(seed)}_epochs",
            "recorded_orders": full / f"stage1_order_seed{int(seed)}.npz"})
    specs.append({
        "name": "target", "role": "target", "seed": int(p.target_seed),
        "train": splits["target_train_idx"], "heldout": splits["target_test_idx"],
        "reference": ck / "target_fixed.pt", "epoch_dir": None, "recorded_orders": None})
    for sid in range(int(p.n_shadow)):
        if sid == a:
            continue
        specs.append({
            "name": f"shadow{sid}_fixed", "role": "fixed_shadow", "seed": int(p.fixed_shadow_seed) + sid,
            "train": splits["shadow_models"][sid]["train_idx"], "heldout": splits["shadow_models"][sid]["test_idx"],
            "reference": ck / f"shadow_{sid}_fixed.pt", "epoch_dir": None, "recorded_orders": None})
    for spec in specs:
        spec["train"] = np.asarray(spec["train"], dtype=np.int64)
        spec["heldout"] = np.asarray(spec["heldout"], dtype=np.int64)
    return specs


def order_sha(orders):
    return hashlib.sha256(np.ascontiguousarray(orders).tobytes()).hexdigest()[:16]


def reference_files(spec, T0, save_epochs):
    files = [spec["reference"]]
    if spec["recorded_orders"] is not None:
        files.append(spec["recorded_orders"])
    if spec["epoch_dir"] is not None:
        files.extend(spec["epoch_dir"] / f"epoch{e:03d}.pt" for e in sorted(save_epochs) if e <= T0)
    return files


# ---------------------------------------------------------------------------
# per-epoch observation (runs inside core.preserve_rng_and_modes)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rows(model, X, y, rows, batch):
    """Eval-mode probabilities, per-row CE and correctness without DataLoader RNG use."""
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    probs, ce, correct = [], [], []
    for start in range(0, len(rows), batch):
        idx = rows[start:start + batch]
        xb = torch.as_tensor(X[idx], dtype=dtype, device=device)
        yb = torch.as_tensor(y[idx], dtype=torch.long, device=device)
        logits = model(xb)
        ce.append(F.cross_entropy(logits, yb, reduction="none").double().cpu())
        probs.append(torch.softmax(logits, dim=1).double().cpu())
        correct.append((logits.argmax(1) == yb).cpu())
    return torch.cat(probs).numpy(), torch.cat(ce).numpy(), torch.cat(correct).numpy()


class EpochObserver:
    def __init__(self, spec, context, args, T0, output_csv):
        self.spec, self.args, self.T0 = spec, args, T0
        self.X, self.y = context["X"], context["y"]
        self.rows = np.concatenate([spec["train"], spec["heldout"]])  # make_interface row order
        self.n_train = len(spec["train"])
        self.labels = self.y[self.rows]
        self.previous = None
        self.anchor = None
        self.records = []
        self.output_csv = output_csv

    def __call__(self, model, row):
        epoch = int(row["epoch"])
        with_grad = epoch % self.args.gradient_every == 0 or epoch in (0, self.T0, self.args.epochs)
        info = core.observe_checkpoint(model, self.X, self.y, self.spec["train"], self.args.hvp_batch, WD, with_grad)
        probs, ce, correct = evaluate_rows(model, self.X, self.y, self.rows, self.args.eval_batch)
        n = self.n_train
        true_prob = probs[np.arange(len(self.rows)), self.labels]
        out = {"run": self.spec["name"], "role": self.spec["role"], "seed": self.spec["seed"], **row, **info,
               "param_norm": core.norm(core.flat(model)),
               "train_accuracy_eval": float(correct[:n].mean()),
               "heldout_ce": float(ce[n:].mean()), "heldout_accuracy": float(correct[n:].mean()),
               "generalization_gap_ce": float(ce[n:].mean() - ce[:n].mean()),
               "member_true_prob_mean": float(true_prob[:n].mean()),
               "nonmember_true_prob_mean": float(true_prob[n:].mean())}
        # Sanity check that both passes measure the same quantity. Different batch sizes can
        # pick different (TF32) cuDNN kernels, so only a gross disagreement is an error.
        other = float(ce[:n].mean())
        out["eval_ce_crosscheck_abs_diff"] = abs(out["eval_ce"] - other)
        if not math.isclose(out["eval_ce"], other, rel_tol=2e-2, abs_tol=1e-4):
            raise RuntimeError(f"Two eval CE paths disagree at epoch {epoch}: {out['eval_ce']} vs {other}")
        if self.previous is not None:
            tv = 0.5 * np.abs(probs - self.previous).sum(1)
            out.update(interface_tv_prev_mean=float(tv.mean()), interface_tv_prev_max=float(tv.max()),
                       interface_tv_prev_member_mean=float(tv[:n].mean()),
                       interface_tv_prev_nonmember_mean=float(tv[n:].mean()))
        if epoch == self.T0:
            self.anchor = probs.copy()
        if self.anchor is not None:
            tv0 = 0.5 * np.abs(probs - self.anchor).sum(1)
            out.update(interface_tv_from_T0_mean=float(tv0.mean()), interface_tv_from_T0_max=float(tv0.max()))
        self.previous = probs
        self.records.append(out)
        legacy.write_csv(self.output_csv, self.records)
        LOG.info("%s epoch=%d online_CE=%s eval_obj=%.6g grad=%s heldout_CE=%.4f",
                 self.spec["name"], epoch, row.get("online_train_ce"), info["eval_objective"],
                 info.get("eval_grad_norm"), out["heldout_ce"])


# ---------------------------------------------------------------------------
# replay + extension
# ---------------------------------------------------------------------------

def rng_payload():
    state = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def extend_run(spec, context, args, pilot, output):
    p = context["pargs"]
    X, y = context["X"], context["y"]
    T0, E = int(p.shadow_epochs), int(args.epochs)
    orders = pilot.make_epoch_orders(spec["train"], E, spec["seed"] + 700000)
    prefix = pilot.make_epoch_orders(spec["train"], T0, spec["seed"] + 700000)
    if not np.array_equal(orders[:T0], prefix):
        raise RuntimeError("Epoch-order generator is not prefix stable; the continuation would not be exact")
    if spec["recorded_orders"] is not None:
        recorded = np.load(spec["recorded_orders"])["raw_index_order"]
        if not np.array_equal(recorded, prefix):
            raise RuntimeError(f"{spec['name']}: recorded original order differs from the seed derivation")
    np.savez_compressed(output / "epoch_orders.npz", raw_index_order=orders)
    save_epochs = set(int(e) for e in p.save_epoch_checkpoints) if spec["epoch_dir"] is not None else set()
    strict = spec["role"] == "affected_shadow" or not args.allow_fixed_mismatch
    reference = legacy.load_payload(spec["reference"], pilot)
    legacy.assert_equal(int(reference["metadata"]["seed"]), spec["seed"], f"{spec['name']}: reference seed mismatch")
    checks = []

    def record(check):
        checks.append(check)
        legacy.write_json(output / "replay_checks.json", checks)
        if not check["passed"]:
            message = f"{spec['name']} epoch {check['epoch']}: not the original trajectory ({check})"
            if strict:
                raise RuntimeError(message + "; the extended curve would not describe the original run")
            LOG.warning(message)

    def on_epoch(epoch, model, optimizer):
        if epoch in save_epochs:
            payload = legacy.load_payload(spec["epoch_dir"] / f"epoch{epoch:03d}.pt", pilot)
            md = payload["metadata"]
            legacy.assert_equal(md["epoch_order_sha256"], order_sha(prefix), "Checkpoint order fingerprint mismatch")
            current = core.snapshot_rng()
            check = {"epoch": epoch, "kind": "original_epoch_checkpoint",
                     "parameters_equal": core.exact_tree(model.state_dict(), payload["state_dict"]),
                     "optimizer_equal": core.exact_tree(optimizer.state_dict(), md["optimizer_state"]),
                     "cpu_rng_equal": core.exact_tree(current["torch_cpu"], md["rng_states"]["torch_cpu"]),
                     "cuda_rng_equal": core.exact_tree(current["torch_cuda"], md["rng_states"].get("torch_cuda", []))}
            check["passed"] = all(check[k] for k in ("parameters_equal", "optimizer_equal",
                                                      "cpu_rng_equal", "cuda_rng_equal"))
            record(check)
        if epoch == T0:
            fingerprint = core.rng_fingerprint()
            check = {"epoch": epoch, "kind": "original_final_model",
                     "parameters_equal": core.exact_tree(model.state_dict(), reference["state_dict"]),
                     "post_train_rng_equal": fingerprint == reference["metadata"]["metrics"]["post_train_rng_sha256"],
                     "post_train_rng_sha256": fingerprint}
            check["passed"] = check["parameters_equal"] and check["post_train_rng_equal"]
            record(check)
        if epoch in args.checkpoint_epochs:
            pilot.save_model(output / "checkpoints" / f"epoch{epoch:03d}.pt", model, {
                "run": spec["name"], "role": spec["role"], "seed": spec["seed"], "epoch": epoch,
                "original_epochs": T0, "extended_epochs": E,
                "epoch_order_sha256": order_sha(orders), "prefix_order_sha256": order_sha(prefix),
                "optimizer_state": optimizer.state_dict(), "rng_states": rng_payload(),
                "rng_fingerprint": core.rng_fingerprint(), "source": SCHEMA})

    observer = EpochObserver(spec, context, args, T0, output / "curve.csv")
    started = time.time()
    model, optimizer, rng_fp = core.replay_baseline(
        pilot, X, y, orders, spec["seed"], p.shadow_lr, p.shadow_batch_size, WD, observer, on_epoch)
    elapsed = time.time() - started
    if not any(c["kind"] == "original_final_model" for c in checks):
        raise RuntimeError("Final original-model comparison never ran")
    replay_exact = all(c["passed"] for c in checks)
    return observer.records, {"replay_checks": checks, "replay_exact": replay_exact,
                              "final_rng_sha256": rng_fp, "training_and_observation_seconds": elapsed,
                              "epoch_order_sha256": order_sha(orders), "prefix_order_sha256": order_sha(prefix)}


def spectrum(spec, context, args, pilot, score, output):
    """Lanczos Ritz endpoints of H_bar = Hessian(eval CE) + wd I at saved epochs."""
    X, y = context["X"], context["y"]
    rows = []
    for epoch in args.spectrum_epochs:
        payload = legacy.load_payload(output / "checkpoints" / f"epoch{epoch:03d}.pt", pilot)
        model = legacy.model_from_payload(payload, pilot, X, y).to(pilot.DEVICE)
        started = time.time()
        H = score.make_hvp(model, X, y, spec["train"], pilot.DEVICE, WD, 0., args.hvp_batch)
        theta = core.flat(model)
        probes = [core.lanczos_probe(H, theta.numel(), args.lanczos_iters, s, theta.device, theta.dtype)
                  for s in args.probe_seeds]
        grad = score.shadow_loss_grad(model, X, y, spec["train"], pilot.DEVICE, WD, args.hvp_batch)
        lo = min(q["endpoints"]["min"]["value"] for q in probes)
        hi = max(q["endpoints"]["max"]["value"] for q in probes)
        row = {"run": spec["name"], "seed": spec["seed"], "epoch": epoch,
               "min_ritz": lo, "max_ritz": hi, "damping_floor": max(0., -lo),
               "min_ritz_residual_scaled_max": max(q["endpoints"]["min"]["residual_scaled"] for q in probes),
               "max_ritz_residual_scaled_max": max(q["endpoints"]["max"]["residual_scaled"] for q in probes),
               "eval_grad_norm_07": core.norm(grad), "param_norm": core.norm(theta),
               "hvp_calls_upper_bound": len(args.probe_seeds) * (args.lanczos_iters + 2),
               "seconds": time.time() - started}
        reference = (Path(args.v11_damping_dir) / f"spectrum_seed{spec['seed']}.json"
                     if args.v11_damping_dir and epoch == int(context["pargs"].shadow_epochs)
                     and spec["role"] == "affected_shadow" else None)
        if reference is not None and reference.is_file():
            old = legacy.read_json(reference)
            row["v11_min_ritz"] = old["undamped_min_ritz_estimate"]
            row["v11_max_ritz"] = old["undamped_max_ritz_estimate"]
            row["v11_grad_norm"] = old["baseline_eval_gradient_norm"]
            row["v11_same_probe_seeds"] = sorted(int(q["probe_seed"]) for q in old["probes"]) == sorted(args.probe_seeds)
        rows.append(row)
        legacy.write_json(output / f"spectrum_epoch{epoch:03d}.json", {**row, "probes": probes,
                          "warning": "Finite Krylov Ritz estimates; not an SPD certificate"})
        legacy.write_csv(output / "spectrum.csv", rows)
        LOG.info("%s spectrum epoch=%d min=%.5g max=%.5g grad=%.4g", spec["name"], epoch, lo, hi, row["eval_grad_norm_07"])
        del H, model
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full_dir", required=True, help="original shadow*_full directory (read only)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", required=True, help="shared output root; each run writes <out_root>/<run>")
    ap.add_argument("--run", default=None, help="run name, e.g. shadow3_seed42, target, shadow0_fixed")
    ap.add_argument("--array_index", type=int, default=None, help="index into the fixed run list")
    ap.add_argument("--list_runs", action="store_true")
    ap.add_argument("--epochs", type=int, default=100, help="extended total epochs E_max")
    ap.add_argument("--checkpoint_epochs", type=int, nargs="*", default=None,
                    help="default: T0 and every 10 epochs up to E_max")
    ap.add_argument("--spectrum_epochs", type=int, nargs="*", default=None,
                    help="default: T0, T0+10, ..., E_max (affected shadow runs only)")
    ap.add_argument("--spectrum_roles", nargs="*", default=["affected_shadow"],
                    choices=["affected_shadow", "target", "fixed_shadow"])
    ap.add_argument("--lanczos_iters", type=int, default=50)
    ap.add_argument("--probe_seeds", type=int, nargs="+", default=[1701, 1702])
    ap.add_argument("--hvp_batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--gradient_every", type=int, default=1)
    ap.add_argument("--allow_fixed_mismatch", action="store_true", default=True,
                    help="target/fixed shadows: record a bitwise mismatch instead of failing (default)")
    ap.add_argument("--strict_all", dest="allow_fixed_mismatch", action="store_false",
                    help="fail on any bitwise mismatch, including target/fixed shadows")
    ap.add_argument("--v11_damping_dir", default=None, help="optional v1.1 damping dir for the epoch-T0 spectrum cross-check")
    ap.add_argument("--require_cuda", action="store_true")
    return ap.parse_args(argv)


def resolve_defaults(args, T0):
    if args.epochs <= T0:
        raise ValueError(f"--epochs must exceed the original {T0}")
    if args.checkpoint_epochs is None:
        args.checkpoint_epochs = sorted({T0, *range(T0, args.epochs + 1, 10), args.epochs})
    if args.spectrum_epochs is None:
        args.spectrum_epochs = sorted({T0, *range(T0, args.epochs + 1, 10), args.epochs})
    args.checkpoint_epochs = sorted(set(args.checkpoint_epochs) | set(args.spectrum_epochs))
    if any(not 1 <= e <= args.epochs for e in args.checkpoint_epochs):
        raise ValueError("checkpoint/spectrum epochs must be within 1..E_max")
    if min(args.lanczos_iters, args.hvp_batch, args.eval_batch, args.gradient_every) < 1:
        raise ValueError("iteration and batch settings must be positive")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this invocation")
    full = Path(args.full_dir).resolve()
    out_root = Path(args.out_root).resolve()
    for protected in (full, REPO):
        if out_root == protected or out_root.is_relative_to(protected) or protected.is_relative_to(out_root):
            raise RuntimeError(f"Output root overlaps a protected directory: {protected}")
    score = legacy.module_from_path("score07_a20", HERE / "07_cross_stage_score_ladder.py")
    pilot = score.import_pilot_module()
    if args.list_runs:
        config = legacy.read_json(full / "experiment_config.json")
        split = legacy.read_json(full / "splits" / "fresh_patient_split.json")
        context = {"pargs": argparse.Namespace(**config["args"]), "splits": split, "full": full}
        for index, spec in enumerate(run_specs(context)):
            print(index, spec["name"], spec["role"], spec["seed"])
        return
    ns = argparse.Namespace(full_dir=str(full), dose_dir=None, mode="convergence", data_dir=args.data_dir,
                            seeds=json.loads((full / "experiment_config.json").read_text())["args"]["seeds"])
    context = legacy.load_context(ns, pilot)
    context["splits"] = legacy.read_json(full / "splits" / "fresh_patient_split.json")
    T0 = int(context["pargs"].shadow_epochs)
    resolve_defaults(args, T0)
    specs = run_specs(context)
    if (args.run is None) == (args.array_index is None):
        raise ValueError("Pass exactly one of --run or --array_index")
    if args.array_index is not None:
        if not 0 <= args.array_index < len(specs):
            raise ValueError(f"--array_index must be in 0..{len(specs) - 1}")
        spec = specs[args.array_index]
    else:
        matches = [s for s in specs if s["name"] == args.run]
        if len(matches) != 1:
            raise ValueError(f"Unknown run {args.run}; choose from {[s['name'] for s in specs]}")
        spec = matches[0]
    output = out_root / spec["name"]
    output.mkdir(parents=True, exist_ok=False)
    save_epochs = set(int(e) for e in context["pargs"].save_epoch_checkpoints)
    common_inputs = [full / n for n in ("experiment_config.json", "experiment_summary.json",
                                        "selected_patients.json", "splits/fresh_patient_split.json")]
    inputs = list(dict.fromkeys(common_inputs + reference_files(spec, T0, save_epochs)))
    fingerprints = {str(path): legacy.sha(path) for path in inputs}
    legacy.write_json(output / "input_sha256.json", fingerprints)

    def git(*cmd):
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *cmd], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    manifest = {
        "schema": SCHEMA, "status": "running", "run": spec["name"], "role": spec["role"], "seed": spec["seed"],
        "all_runs": [s["name"] for s in specs], "original_epochs": T0, "extended_epochs": args.epochs,
        "args": vars(args), "training_setup": {k: getattr(context["pargs"], k) for k in
                                                ("shadow_lr", "shadow_batch_size", "save_epoch_checkpoints", "deterministic")},
        "weight_decay": WD, "n_train": int(len(spec["train"])), "n_heldout": int(len(spec["heldout"])),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "device": str(pilot.DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scope": "Descriptive convergence curve of the original Stage 1 recipe extended to E_max epochs. "
                 "No deletion, attack or truth. Plateau choice is made by the report script and the team.",
    }
    legacy.write_json(output / "manifest.json", manifest)
    started = time.time()
    try:
        rows, replay = extend_run(spec, context, args, pilot, output)
        manifest.update(replay)
        spectrum_rows = []
        if spec["role"] in args.spectrum_roles and args.spectrum_epochs:
            spectrum_rows = spectrum(spec, context, args, pilot, score, output)
        if not all(legacy.sha(Path(path)) == value for path, value in fingerprints.items()):
            raise RuntimeError("An original input changed during the run")
        manifest.update(status="complete", elapsed_seconds=time.time() - started, n_curve_rows=len(rows),
                        n_spectrum_rows=len(spectrum_rows), input_files_unchanged=True,
                        final_row={k: v for k, v in rows[-1].items() if not isinstance(v, (dict, list))})
        legacy.write_json(output / "manifest.json", manifest)
        LOG.info("COMPLETE %s -> %s", spec["name"], output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time() - started)
        legacy.write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
