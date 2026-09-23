#!/usr/bin/env python3
"""Tests for the paired lr-continuation probe (23_A_lr_continuation.py).

Only the OCT image loader is mocked; training, checkpoints and the resume
checks run through the production code on tiny synthetic images.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lr23 = load("a23_lr_test", "23_A_lr_continuation.py")
report = load("a23_lr_report_test", "23_A_lr_report.py")
conv = lr23.conv
legacy = lr23.legacy
score = legacy.module_from_path("score07_a23_test", HERE / "07_cross_stage_score_ladder.py")
pilot = score.import_pilot_module()
torch.set_num_threads(min(2, torch.get_num_threads()))
T0, STEP1_E, RESUME, EXTRA = 2, 4, 4, 2
BASE_LR = 1e-3


def build_fixture(td):
    """Original T0-epoch truth directory for one affected shadow seed."""
    rng = np.random.default_rng(52)
    X = rng.random((16, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), 4).astype(np.int64)
    groups = np.arange(100, 116)
    split = {"target_train_idx": [8, 9, 10, 11], "target_test_idx": [12, 13, 14, 15],
             "shadow_models": [{"train_idx": [0, 1, 2, 3], "test_idx": [4, 5, 6, 7]}]}
    patient = {"patient_id": 101, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]}
    full = Path(td) / "full"
    with mock.patch.object(sys, "argv", ["pilot"]):
        p = pilot.parse_args()
    p.n_shadow, p.affected_shadow, p.seeds, p.classes, p.n_patients = 1, 0, [42], [1], 1
    p.shadow_epochs, p.shadow_batch_size, p.shadow_lr = T0, 4, BASE_LR
    p.save_epoch_checkpoints, p.deletion_mode = [1, T0], "fixed_mask"
    p.output_dir, p.n_total_samples = str(full), 16
    legacy.write_json(full / "splits/fresh_patient_split.json", split)
    split_hash = legacy.sha(full / "splits/fresh_patient_split.json")[:16]
    legacy.write_json(full / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
    legacy.write_json(full / "experiment_config.json", {
        "training_numerics": pilot.TRAINING_NUMERICS, "args": vars(p),
        "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
    legacy.write_json(full / "selected_patients.json", {"split_sha256": split_hash, "patients": [patient]})
    orders = pilot.make_epoch_orders(np.asarray(split["shadow_models"][0]["train_idx"]), T0, 700042)
    baseline, metrics = pilot.train_classifier_from_orders(
        X, y, orders, split["shadow_models"][0]["test_idx"], 42, 128, BASE_LR, 4, 1e-5, True,
        deletion_mode="fixed_mask", epoch_checkpoint_dir=full / "checkpoints/baseline_seed42_epochs",
        epoch_checkpoints=[1, T0])
    pilot.save_model(full / "checkpoints/shadow_0_baseline_seed42.pt", baseline, {"seed": 42, "metrics": metrics})
    np.savez_compressed(full / "stage1_order_seed42.npz", raw_index_order=orders)
    return {"X": X, "y": y, "groups": groups, "split": split, "full": full, "pargs": p}


def patched(fx):
    return (mock.patch.object(legacy, "module_from_path", return_value=score),
            mock.patch.object(score, "import_pilot_module", return_value=pilot),
            mock.patch.object(pilot, "load_dataset", return_value=(fx["X"], fx["y"], fx["groups"])))


def run_step1(fx, out_root):
    argv = ["--full_dir", str(fx["full"]), "--data_dir", "unused", "--out_root", str(out_root),
            "--array_index", "0", "--epochs", str(STEP1_E), "--checkpoint_epochs", str(T0), "3", str(STEP1_E),
            "--gradient_every", "1", "--dropout_mc_every", "1", "--dropout_mc_reps", "2",
            "--hvp_batch", "4", "--eval_batch", "5"]
    a, b, c = patched(fx)
    with a, b, c:
        conv.main(argv)
    return out_root / "shadow0_seed42"


def run_probe(fx, step1_root, out_root, arm, schedule=None, resume=RESUME, extra=EXTRA):
    argv = ["--full_dir", str(fx["full"]), "--data_dir", "unused", "--step1_root", str(step1_root),
            "--out_root", str(out_root), "--run", "shadow0_seed42", "--arm", arm,
            "--resume_epoch", str(resume), "--extra_epochs", str(extra),
            "--gradient_every", "1", "--dropout_mc_every", "1", "--dropout_mc_reps", "2",
            "--hvp_batch", "4", "--eval_batch", "5"]
    if schedule is not None:
        argv += ["--decay_schedule", schedule]
    a, b, c = patched(fx)
    with a, b, c:
        lr23.main(argv)
    return out_root / f"shadow0_seed42_{arm}"


class LrContinuationTests(unittest.TestCase):
    def test_control_equals_from_scratch_and_decay_is_paired(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            step1 = run_step1(fx, Path(td) / "runs/A/conv")
            probes = Path(td) / "runs/A/probe"
            control = run_probe(fx, step1.parent, probes, "control")
            manifest = legacy.read_json(control / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["resume_check"]["passed"])
            self.assertTrue(manifest["input_files_unchanged"])
            total = RESUME + EXTRA
            curve = report.read_csv(control / "curve.csv")
            self.assertEqual([r["epoch"] for r in curve], list(range(RESUME, total + 1)))
            self.assertTrue(all(r["eval_grad_norm"] is not None for r in curve))
            self.assertEqual(curve[0]["interface_tv_from_T0_mean"], 0.0)
            self.assertEqual([r["lr"] for r in curve[1:]], [BASE_LR] * EXTRA)
            # Bitwise: control continuation == training from scratch for RESUME+EXTRA epochs.
            X, y, split, p = fx["X"], fx["y"], fx["split"], fx["pargs"]
            orders = pilot.make_epoch_orders(np.asarray(split["shadow_models"][0]["train_idx"]), total, 700042)
            direct, metrics = pilot.train_classifier_from_orders(
                X, y, orders, split["shadow_models"][0]["test_idx"], 42, 128, BASE_LR, 4, 1e-5, True,
                deletion_mode="fixed_mask", epoch_checkpoint_dir=Path(td) / "direct", epoch_checkpoints=[total])
            saved = legacy.load_payload(control / "checkpoints" / f"epoch{total:03d}.pt", pilot)
            self.assertTrue(core.exact_tree(saved["state_dict"], direct.state_dict()))
            direct_ck = torch.load(Path(td) / "direct" / f"epoch{total:03d}.pt", weights_only=False)
            self.assertTrue(core.exact_tree(saved["metadata"]["optimizer_state"],
                                            direct_ck["metadata"]["optimizer_state"]))
            self.assertEqual(saved["metadata"]["rng_fingerprint"], metrics["post_train_rng_sha256"])
            # A decay arm whose schedule equals the base lr is bitwise identical: exact pairing.
            paired = run_probe(fx, step1.parent, Path(td) / "runs/A/paired", "decay",
                               schedule=f"{EXTRA}@{BASE_LR}")
            same = legacy.load_payload(paired / "checkpoints" / f"epoch{total:03d}.pt", pilot)
            self.assertTrue(core.exact_tree(same["state_dict"], saved["state_dict"]))
            self.assertTrue(core.exact_tree(same["metadata"]["optimizer_state"],
                                            saved["metadata"]["optimizer_state"]))
            # A genuinely smaller lr changes the endpoint, is logged, and moves less.
            decay = run_probe(fx, step1.parent, probes, "decay", schedule="1@1e-4,1@1e-5")
            decay_ck = legacy.load_payload(decay / "checkpoints" / f"epoch{total:03d}.pt", pilot)
            self.assertFalse(core.exact_tree(decay_ck["state_dict"], saved["state_dict"]))
            self.assertEqual(decay_ck["metadata"]["learning_rate_at_epoch"], 1e-5)
            dcurve = report.read_csv(decay / "curve.csv")
            self.assertEqual([r["lr"] for r in dcurve[1:]], [1e-4, 1e-5])
            self.assertLess(sum(r["epoch_update_norm"] for r in dcurve[1:]),
                            sum(r["epoch_update_norm"] for r in curve[1:]))
            # Report runs on the pair and marks the exploratory target without selecting anything.
            report.main(["--root", str(probes), "--tail", "2"])
            summary = json.loads((probes / "report/probe_summary.json").read_text())
            self.assertEqual(sorted(summary["per_run"]), ["shadow0_seed42_control", "shadow0_seed42_decay"])
            self.assertTrue(summary["pairing_complete"])
            self.assertIn("grad_ratio_decay_over_control", summary["per_seed_pairs"]["42"])
            self.assertTrue((probes / "report/lr_probe_curves.png").is_file())
            # Reruns must not overwrite an existing probe output.
            with self.assertRaises(FileExistsError):
                run_probe(fx, step1.parent, probes, "control")

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            step1 = run_step1(fx, Path(td) / "runs/A/conv")
            probes = Path(td) / "runs/A/refuse"
            with self.assertRaises(RuntimeError):
                run_probe(fx, step1.parent, probes, "control", resume=RESUME + 1)   # not a saved epoch
            with self.assertRaises(RuntimeError):
                run_probe(fx, step1.parent, Path(td) / "runs/A/bad_schedule", "decay",
                          schedule="1@1e-4")                                        # sum != extra_epochs
            manifest_path = step1 / "manifest.json"
            good = manifest_path.read_text()
            manifest = json.loads(good)
            manifest["replay_exact"] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                run_probe(fx, step1.parent, Path(td) / "runs/A/bad_step1", "control")
            manifest_path.write_text(good)
            with self.assertRaises(RuntimeError):
                run_probe(fx, step1.parent, fx["full"] / "inside", "control")       # output overlap


if __name__ == "__main__":
    unittest.main(verbosity=2)
