#!/usr/bin/env python3
"""Run the frozen 05 truth entry while checking every Stage-1 path at epoch 50.

The protected 05 source is imported unchanged.  Its Stage-1 training function
is replaced only for this Route-A invocation by an arithmetic-identical copy
that pauses after the original epoch T0, compares the current native model and
RNG with the corresponding frozen baseline/LOO endpoint, compares Adam state
where the original baseline epoch checkpoint contains it, and then continues
the same in-memory optimizer to E*.  Any mismatch aborts before the new truth
can be called complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(value):
    """Stable content fingerprint for regenerated optimizer/model/RNG trees."""
    digest = hashlib.sha256()

    def add(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())

    add(value)
    return digest.hexdigest()


def option_value(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit(f"Missing required forwarded option {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--original_epochs", type=int, required=True)
    ap.add_argument("--selection", required=True)
    known, forwarded = ap.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    output = Path(option_value(forwarded, "--output_dir")).resolve()
    extended_epochs = int(option_value(forwarded, "--shadow_epochs"))
    if extended_epochs <= known.original_epochs:
        raise SystemExit("Route A truth must extend beyond the original endpoint")
    if output.exists():
        raise SystemExit(f"Route A truth output must be new; refusing resume/overwrite: {output}")

    reference = Path(known.reference_dir).resolve()
    selection_path = Path(known.selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "pathway2_A_frozen_E_star_v1" or
            int(selection["E_star"]) != extended_epochs or
            selection.get("status") != "frozen_human_choice"):
        raise SystemExit("Forwarded E* does not match the frozen human selection")
    pilot = load_module("a21_pilot", HERE / "05_end_to_end_patient_loo_pilot.py")
    config = json.loads((reference / "experiment_config.json").read_text(encoding="utf-8"))
    ref_args = argparse.Namespace(**config["args"])
    if int(ref_args.shadow_epochs) != known.original_epochs:
        raise SystemExit("Reference truth does not use the declared original epoch")
    splits = json.loads((reference / "splits/fresh_patient_split.json").read_text(encoding="utf-8"))
    panel = json.loads((reference / "selected_patients.json").read_text(encoding="utf-8"))["patients"]
    patients = {frozenset(map(int, p["raw_indices"])): int(p["patient_id"]) for p in panel}
    checks = []
    anchor_checks = []
    affected_no_deletion_calls = {}
    report_path = output / "route_a_epoch50_replay_checks.json"
    a = int(ref_args.affected_shadow)

    def same_indices(x, y):
        return np.array_equal(np.sort(np.asarray(x, dtype=np.int64)),
                              np.sort(np.asarray(y, dtype=np.int64)))

    selected_checkpoints = {}
    for path, digest in selection["checkpoint_sha256"].items():
        path = Path(path)
        for seed in map(int, ref_args.seeds):
            if path.parent.parent.name == f"shadow{a}_seed{seed}":
                selected_checkpoints[seed] = (path, digest)
    if set(selected_checkpoints) != set(map(int, ref_args.seeds)):
        raise SystemExit("Frozen selection does not identify one E* checkpoint for every Stage-1 seed")

    def identify(train_indices, excluded_indices, seed):
        excluded = frozenset(map(int, excluded_indices))
        if seed == int(ref_args.target_seed) and same_indices(train_indices, splits["target_train_idx"]):
            return "target", reference / "checkpoints/target_fixed.pt", None
        for sid, split in enumerate(splits["shadow_models"]):
            if sid == a:
                continue
            if seed == int(ref_args.fixed_shadow_seed) + sid and same_indices(train_indices, split["train_idx"]):
                return "fixed_shadow", reference / f"checkpoints/shadow_{sid}_fixed.pt", None
        affected = splits["shadow_models"][a]["train_idx"]
        if seed in set(map(int, ref_args.seeds)) and same_indices(train_indices, affected):
            if not excluded:
                final = reference / f"checkpoints/shadow_{a}_baseline_seed{seed}.pt"
                epoch = reference / f"checkpoints/baseline_seed{seed}_epochs/epoch{known.original_epochs:03d}.pt"
                return "affected_baseline_or_noop", final, epoch
            pid = patients.get(excluded)
            if pid is None:
                raise RuntimeError(f"No frozen patient matches excluded raw indices {sorted(excluded)}")
            return "loo", reference / f"checkpoints/shadow_{a}_seed{seed}_patient{pid}.pt", None
        raise RuntimeError(f"Unrecognized Stage-1 training path: seed={seed}, n_train={len(train_indices)}")

    def verify_epoch(model, optimizer, train_orders, excluded_indices, seed, identity):
        role, final_path, epoch_path = identity
        with core.preserve_rng_and_modes(model):
            current = core.snapshot_rng()
            fingerprint = core.rng_fingerprint()
            final = torch.load(final_path, map_location="cpu", weights_only=False)
            pilot.require_training_numerics(final.get("metadata", {}), str(final_path))
            md = final["metadata"]
            check = {
                "role": role, "seed": int(seed), "epoch": known.original_epochs,
                "training_instance": final_path.stem,
                "reference": str(final_path), "reference_sha256": sha(final_path),
                "parameters_equal": core.exact_tree(model.state_dict(), final["state_dict"]),
                "post_train_rng_equal": fingerprint == md["metrics"]["post_train_rng_sha256"],
                "post_train_rng_sha256": fingerprint,
                "regenerated_optimizer_sha256": tree_sha(optimizer.state_dict()),
                "optimizer_equal": None, "checkpoint_rng_equal": None,
            }
            if role in ("affected_baseline_or_noop", "loo"):
                recorded = np.load(reference / f"stage1_order_seed{int(seed)}.npz")["raw_index_order"]
                check["recorded_order_prefix_equal"] = np.array_equal(
                    np.asarray(train_orders[:known.original_epochs]), recorded)
            else:
                check["recorded_order_prefix_equal"] = None
            if epoch_path is not None:
                epoch_payload = torch.load(epoch_path, map_location="cpu", weights_only=False)
                pilot.require_training_numerics(epoch_payload.get("metadata", {}), str(epoch_path))
                emd = epoch_payload["metadata"]
                check["epoch_checkpoint"] = str(epoch_path)
                check["epoch_checkpoint_sha256"] = sha(epoch_path)
                check["epoch_parameters_equal"] = core.exact_tree(model.state_dict(), epoch_payload["state_dict"])
                check["optimizer_equal"] = core.exact_tree(optimizer.state_dict(), emd["optimizer_state"])
                check["checkpoint_rng_equal"] = (
                    core.exact_tree(current["torch_cpu"], emd["rng_states"]["torch_cpu"]) and
                    core.exact_tree(current["torch_cuda"], emd["rng_states"].get("torch_cuda", [])))
            required = [check["parameters_equal"], check["post_train_rng_equal"]]
            if check["recorded_order_prefix_equal"] is not None:
                required.append(check["recorded_order_prefix_equal"])
            if epoch_path is not None:
                required += [check["epoch_parameters_equal"], check["optimizer_equal"], check["checkpoint_rng_equal"]]
            check["passed"] = all(required)
        checks.append(check)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"status": "running", "checks": checks}, indent=2) + "\n")
        if not check["passed"]:
            raise RuntimeError(f"Epoch-{known.original_epochs} replay failed: {check}")
        return check

    def verify_selected_anchor(model, optimizer, seed, identity):
        role, _, _ = identity
        if role != "affected_baseline_or_noop":
            return
        path, expected_sha = selected_checkpoints[int(seed)]
        if sha(path) != expected_sha:
            raise RuntimeError(f"Frozen E* checkpoint hash changed: {path}")
        with core.preserve_rng_and_modes(model):
            current = core.snapshot_rng()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            md = payload["metadata"]
            check = {
                "role": role, "seed": int(seed), "epoch": extended_epochs,
                "reference": str(path), "reference_sha256": expected_sha,
                "parameters_equal": core.exact_tree(model.state_dict(), payload["state_dict"]),
                "optimizer_equal": core.exact_tree(optimizer.state_dict(), md["optimizer_state"]),
                "rng_equal": (core.exact_tree(current["torch_cpu"], md["rng_states"]["torch_cpu"]) and
                              core.exact_tree(current["torch_cuda"], md["rng_states"].get("torch_cuda", []))),
            }
            check["passed"] = check["parameters_equal"] and check["optimizer_equal"] and check["rng_equal"]
        anchor_checks.append(check)
        if not check["passed"]:
            raise RuntimeError(f"E* baseline differs from the frozen Step-1 checkpoint: {check}")

    def checked_train(X, y, train_orders, eval_indices, seed, n_hidden, lr, batch_size,
                      weight_decay, deterministic, excluded_indices=(), removal_window=None,
                      trajectory_noise_seed=None, epoch_checkpoint_dir=None, epoch_checkpoints=(),
                      deletion_mode="filter_rechunk", deletion_weight=1.0):
        pilot.seed_everything(seed, deterministic)
        n_classes = int(np.max(y)) + 1
        model = pilot.build_model("cnn", X.shape[1], n_hidden, n_classes)
        if trajectory_noise_seed is not None:
            raise RuntimeError("Route A formal truth does not permit trajectory-noise replays")
        optimizer = pilot.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        excluded_all = set(int(v) for v in excluded_indices)
        identity = identify(train_orders[0], excluded_all, seed)
        if identity[0] == "affected_baseline_or_noop":
            occurrence = affected_no_deletion_calls.get(int(seed), 0)
            affected_no_deletion_calls[int(seed)] = occurrence + 1
            if occurrence == 0:
                final_path = reference / f"checkpoints/shadow_{a}_baseline_seed{int(seed)}.pt"
            elif occurrence <= int(ref_args.noop_replays):
                final_path = reference / f"checkpoints/shadow_{a}_noop_seed{int(seed)}_r{occurrence - 1}.pt"
            else:
                raise RuntimeError(f"Unexpected extra no-deletion replay for seed {seed}")
            identity = (identity[0], final_path, identity[2])
        if deletion_mode not in ("filter_rechunk", "fixed_mask"):
            raise ValueError(f"unknown deletion_mode {deletion_mode}")
        if not 0.0 < deletion_weight <= 1.0:
            raise ValueError("deletion_weight must be in (0, 1]")
        if deletion_mode == "filter_rechunk" and deletion_weight != 1.0:
            raise ValueError("--deletion_weight < 1 requires --deletion_mode fixed_mask")
        excluded_tensor = torch.as_tensor(sorted(excluded_all), dtype=torch.int64, device=pilot.DEVICE)
        n_epochs_full_exposure = 0
        n_epochs_partial_exposure = 0
        checkpoints = set(int(e) for e in epoch_checkpoints)
        epoch_check = None

        for epoch, base_order in enumerate(train_orders):
            in_window = removal_window is None or removal_window[0] <= epoch < removal_window[1]
            excluded = excluded_all if in_window else set()
            if not excluded or deletion_mode == "fixed_mask":
                order = base_order
            else:
                order = np.asarray([v for v in base_order if int(v) not in excluded], dtype=np.int64)
            if len(order) == 0:
                raise ValueError("LOO removed every Stage-1 training image")
            if excluded_all and not excluded:
                n_epochs_full_exposure += 1
            elif excluded_all and deletion_weight < 1.0:
                n_epochs_partial_exposure += 1
            model.train()
            for batch_idx in pilot._iter_batches(order, batch_size):
                xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=pilot.DEVICE)
                yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=pilot.DEVICE)
                optimizer.zero_grad(set_to_none=True)
                per_example = pilot.nn.functional.cross_entropy(model(xb), yb, reduction="none")
                weights = torch.ones_like(per_example)
                if deletion_mode == "fixed_mask" and excluded:
                    idx_t = torch.as_tensor(batch_idx, dtype=torch.int64, device=pilot.DEVICE)
                    removed = torch.isin(idx_t, excluded_tensor)
                    weights = torch.where(removed, weights * (1.0 - deletion_weight), weights)
                loss = (weights * per_example).sum() / per_example.numel()
                loss.backward()
                optimizer.step()
            completed = epoch + 1
            if completed == known.original_epochs:
                epoch_check = verify_epoch(model, optimizer, train_orders, excluded_all, seed, identity)
            if epoch_checkpoint_dir is not None and completed in checkpoints:
                rng_states = {"torch_cpu": torch.get_rng_state()}
                if torch.cuda.is_available():
                    rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()
                pilot.save_model(Path(epoch_checkpoint_dir) / f"epoch{completed:03d}.pt", model, {
                    "seed": seed, "epoch": completed, "excluded_indices": sorted(excluded_all),
                    "removal_window": list(removal_window) if removal_window else None,
                    "deletion_mode": deletion_mode, "deletion_weight": deletion_weight,
                    "epoch_order_sha256": hashlib.sha256(
                        np.ascontiguousarray(train_orders).tobytes()).hexdigest()[:16],
                    "optimizer_state": optimizer.state_dict(), "rng_states": rng_states})

        if epoch_check is None:
            raise RuntimeError(f"Training path never reached original epoch {known.original_epochs}")
        epoch_check["continued_optimizer_sha256_at_E_star"] = tree_sha(optimizer.state_dict())
        verify_selected_anchor(model, optimizer, seed, identity)

        rng_fp = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            for state in torch.cuda.get_rng_state_all():
                rng_fp.update(state.cpu().numpy().tobytes())
        post_train_rng_sha256 = rng_fp.hexdigest()[:16]
        all_train = np.asarray(train_orders[0], dtype=np.int64)
        without_patient = np.asarray([v for v in all_train if int(v) not in excluded_all], dtype=np.int64)
        fully_removed = bool(excluded_all) and deletion_weight == 1.0 and (
            removal_window is None or tuple(removal_window) == (0, len(train_orders)))
        effective_train = without_patient if fully_removed else all_train
        exposure = {
            "n_excluded_images": int(len(excluded_all)), "n_epochs_total": int(len(train_orders)),
            "n_epochs_with_full_exposure": int(n_epochs_full_exposure),
            "n_epochs_with_partial_exposure": int(n_epochs_partial_exposure),
            "removal_window": list(removal_window) if removal_window else None,
            "deletion_mode": deletion_mode, "deletion_weight": float(deletion_weight),
            "fully_removed": fully_removed,
        }
        metrics = {
            "train_accuracy": pilot.classification_accuracy(model, X, y, effective_train),
            "train_accuracy_excluding_patient": pilot.classification_accuracy(model, X, y, without_patient),
            "patient_images_accuracy": (pilot.classification_accuracy(model, X, y, sorted(excluded_all))
                                        if excluded_all else None),
            "heldout_accuracy": pilot.classification_accuracy(model, X, y, eval_indices),
            "n_train_images": int(len(effective_train)),
            "n_train_images_excluding_patient": int(len(without_patient)),
            "n_heldout_images": int(len(eval_indices)),
            "post_train_rng_sha256": post_train_rng_sha256, "exposure": exposure,
        }
        model.eval()
        return model, metrics

    pilot.train_classifier_from_orders = checked_train
    sys.argv = [str(HERE / "05_end_to_end_patient_loo_pilot.py"), *forwarded]
    try:
        pilot.main()
    except Exception:
        if report_path.parent.exists():
            report_path.write_text(json.dumps({"status": "failed", "checks": checks}, indent=2) + "\n")
        raise

    expected = {
        "target": 1,
        "fixed_shadow": int(ref_args.n_shadow) - 1,
        "affected_baseline_or_noop": len(ref_args.seeds) * (1 + int(ref_args.noop_replays)),
        "loo": len(ref_args.seeds) * len(panel),
    }
    actual = {role: sum(c["role"] == role for c in checks) for role in expected}
    summary = json.loads((output / "experiment_summary.json").read_text(encoding="utf-8"))
    passed = (summary.get("status") == "complete" and actual == expected and
              all(c["passed"] and c.get("regenerated_optimizer_sha256") and
                  c.get("continued_optimizer_sha256_at_E_star") for c in checks))
    payload = {
        "schema": "pathway2_A_epoch50_replay_v2", "status": "complete" if passed else "failed",
        "original_epochs": known.original_epochs, "extended_epochs": extended_epochs,
        "reference_dir": str(reference), "reference_config_sha256": sha(reference / "experiment_config.json"),
        "selection": str(selection_path), "selection_sha256": sha(selection_path),
        "expected_counts": expected, "actual_counts": actual, "checks": checks,
        "expected_E_star_anchor_checks": len(ref_args.seeds) * (1 + int(ref_args.noop_replays)),
        "E_star_anchor_checks": anchor_checks,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if len(anchor_checks) != payload["expected_E_star_anchor_checks"] or not all(c["passed"] for c in anchor_checks):
        passed = False
        payload["status"] = "failed"
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Route A replay panel incomplete: expected={expected}, actual={actual}, summary={summary.get('status')}")
    print(json.dumps({"status": "complete", "replay_checks": len(checks),
                      "E_star_anchor_checks": len(anchor_checks), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
