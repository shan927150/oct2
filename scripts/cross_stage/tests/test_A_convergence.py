#!/usr/bin/env python3
"""Tests for Route A step 1 (20_A_convergence_curve.py / 20_A_convergence_report.py).

Only the OCT image loader is mocked. Training, replay checks, checkpoints and
Lanczos probes run through the production code on tiny synthetic images.
"""
import csv
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


conv = load("a20_curve_test", "20_A_convergence_curve.py")
report = load("a20_report_test", "20_A_convergence_report.py")
legacy = conv.legacy
score = legacy.module_from_path("score07_a20_test", HERE / "07_cross_stage_score_ladder.py")
pilot = score.import_pilot_module()
torch.set_num_threads(min(2, torch.get_num_threads()))


def build_fixture(td, T0=2, target_lr=None, baseline_lr=None):
    """Original T0-epoch run: affected shadow 0 (seed 42), target, fixed shadow 1."""
    rng = np.random.default_rng(52)
    X = rng.random((24, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), 6).astype(np.int64)
    groups = np.arange(100, 124)
    split = {"target_train_idx": [12, 13, 14, 15], "target_test_idx": [16, 17, 18, 19],
             "shadow_models": [{"train_idx": [0, 1, 2, 3], "test_idx": [4, 5, 6, 7]},
                               {"train_idx": [8, 9, 10, 11], "test_idx": [20, 21, 22, 23]}]}
    patient = {"patient_id": 101, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]}
    full = Path(td) / "full"
    with mock.patch.object(sys, "argv", ["pilot"]):
        p = pilot.parse_args()
    p.n_shadow, p.affected_shadow, p.seeds, p.classes, p.n_patients = 2, 0, [42], [1], 1
    p.shadow_epochs, p.shadow_batch_size = T0, 4
    p.save_epoch_checkpoints, p.deletion_mode = [1, T0], "fixed_mask"
    p.output_dir, p.n_total_samples = str(full), 24
    legacy.write_json(full / "splits/fresh_patient_split.json", split)
    split_hash = legacy.sha(full / "splits/fresh_patient_split.json")[:16]
    legacy.write_json(full / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
    legacy.write_json(full / "experiment_config.json", {
        "training_numerics": pilot.TRAINING_NUMERICS, "args": vars(p),
        "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
    legacy.write_json(full / "selected_patients.json", {"split_sha256": split_hash, "patients": [patient]})

    def train(indices, heldout, seed, lr, ckdir=None):
        orders = pilot.make_epoch_orders(np.asarray(indices), T0, seed + 700000)
        model, metrics = pilot.train_classifier_from_orders(
            X, y, orders, heldout, seed, 128, lr, 4, 1e-5, True, deletion_mode="fixed_mask",
            epoch_checkpoint_dir=ckdir, epoch_checkpoints=[1, T0] if ckdir else ())
        return model, metrics, orders

    base, bm, orders = train(split["shadow_models"][0]["train_idx"], split["shadow_models"][0]["test_idx"], 42,
                             baseline_lr or p.shadow_lr, full / "checkpoints/baseline_seed42_epochs")
    pilot.save_model(full / "checkpoints/shadow_0_baseline_seed42.pt", base, {"seed": 42, "metrics": bm})
    np.savez_compressed(full / "stage1_order_seed42.npz", raw_index_order=orders)
    target, tm, _ = train(split["target_train_idx"], split["target_test_idx"], p.target_seed, target_lr or p.shadow_lr)
    pilot.save_model(full / "checkpoints/target_fixed.pt", target, {"seed": p.target_seed, "metrics": tm})
    fixed, fm, _ = train(split["shadow_models"][1]["train_idx"], split["shadow_models"][1]["test_idx"],
                         p.fixed_shadow_seed + 1, p.shadow_lr)
    pilot.save_model(full / "checkpoints/shadow_1_fixed.pt", fixed, {"seed": p.fixed_shadow_seed + 1, "metrics": fm})
    return {"X": X, "y": y, "groups": groups, "split": split, "full": full, "pargs": p}


def run_main(fx, out_root, *extra):
    argv = ["--full_dir", str(fx["full"]), "--data_dir", "unused", "--out_root", str(out_root),
            "--epochs", "4", "--lanczos_iters", "3", "--probe_seeds", "17", "--hvp_batch", "4",
            "--eval_batch", "5", *extra]
    with mock.patch.object(legacy, "module_from_path", return_value=score), \
         mock.patch.object(score, "import_pilot_module", return_value=pilot), \
         mock.patch.object(pilot, "load_dataset", return_value=(fx["X"], fx["y"], fx["groups"])):
        conv.main(argv)


class ConvergenceCurveTests(unittest.TestCase):
    def test_epoch_orders_are_prefix_stable(self):
        idx = np.arange(37) * 3
        for seed in (700042, 742007):
            self.assertTrue(np.array_equal(pilot.make_epoch_orders(idx, 9, seed)[:4],
                                           pilot.make_epoch_orders(idx, 4, seed)))

    def test_extension_equals_training_from_scratch_for_E_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            out_root = Path(td) / "runs" / "A" / "conv"
            run_main(fx, out_root, "--array_index", "0")
            out = out_root / "shadow0_seed42"
            manifest = legacy.read_json(out / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["replay_exact"])
            self.assertTrue(manifest["input_files_unchanged"])
            self.assertEqual(manifest["all_runs"], ["shadow0_seed42", "target", "shadow1_fixed"])
            kinds = [(c["kind"], c["epoch"]) for c in manifest["replay_checks"]]
            self.assertEqual(kinds, [("original_epoch_checkpoint", 1), ("original_epoch_checkpoint", 2),
                                     ("original_final_model", 2)])
            curve = report.read_csv(out / "curve.csv")
            self.assertEqual([r["epoch"] for r in curve], [0, 1, 2, 3, 4])
            self.assertTrue(all(r["eval_grad_norm"] is not None for r in curve))
            self.assertEqual(curve[2]["interface_tv_from_T0_mean"], 0.0)
            # Bitwise: saved epoch-4 state == direct 4-epoch training (observer and checkpoints
            # did not perturb the trajectory; continuation == from-scratch E-epoch run).
            X, y, split, p = fx["X"], fx["y"], fx["split"], fx["pargs"]
            orders4 = pilot.make_epoch_orders(np.asarray(split["shadow_models"][0]["train_idx"]), 4, 700042)
            direct, metrics = pilot.train_classifier_from_orders(
                X, y, orders4, split["shadow_models"][0]["test_idx"], 42, 128, p.shadow_lr, 4, 1e-5, True,
                deletion_mode="fixed_mask", epoch_checkpoint_dir=Path(td) / "direct", epoch_checkpoints=[4])
            saved = legacy.load_payload(out / "checkpoints/epoch004.pt", pilot)
            self.assertTrue(core.exact_tree(saved["state_dict"], direct.state_dict()))
            direct_ck = torch.load(Path(td) / "direct/epoch004.pt", weights_only=False)
            self.assertTrue(core.exact_tree(saved["metadata"]["optimizer_state"], direct_ck["metadata"]["optimizer_state"]))
            self.assertEqual(saved["metadata"]["rng_fingerprint"], metrics["post_train_rng_sha256"])
            original = legacy.load_payload(fx["full"] / "checkpoints/shadow_0_baseline_seed42.pt", pilot)
            self.assertTrue(core.exact_tree(legacy.load_payload(out / "checkpoints/epoch002.pt", pilot)["state_dict"],
                                            original["state_dict"]))
            spectrum = report.read_csv(out / "spectrum.csv")
            self.assertEqual([r["epoch"] for r in spectrum], [2, 4])
            self.assertTrue(all(r["damping_floor"] >= 0 for r in spectrum))
            # Output directories are never reused.
            with self.assertRaises(FileExistsError):
                run_main(fx, out_root, "--array_index", "0")

    def test_fixed_model_mismatch_is_recorded_and_affected_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td, target_lr=3e-3)   # reference target trained with another recipe
            out_root = Path(td) / "runs" / "A" / "conv"
            run_main(fx, out_root, "--run", "target")
            manifest = legacy.read_json(out_root / "target/manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["replay_exact"])
            run_main(fx, out_root, "--run", "shadow1_fixed")
            self.assertTrue(legacy.read_json(out_root / "shadow1_fixed/manifest.json")["replay_exact"])
            with self.assertRaises(RuntimeError):
                run_main(fx, Path(td) / "runs" / "A" / "strict", "--run", "target", "--strict_all")
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td, baseline_lr=3e-3)
            out_root = Path(td) / "runs" / "A" / "conv"
            with self.assertRaises(RuntimeError):
                run_main(fx, out_root, "--array_index", "0")
            self.assertEqual(legacy.read_json(out_root / "shadow0_seed42/manifest.json")["status"], "failed")

    def test_output_may_not_overlap_original(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            with self.assertRaises(RuntimeError):
                run_main(fx, fx["full"] / "extension", "--array_index", "0")


def fake_run(root, name, role, seed, ce_by_epoch, T0=50, E=100, grad=0.3):
    folder = Path(root) / name
    folder.mkdir(parents=True)
    rows = []
    for e in range(E + 1):
        rows.append({"run": name, "role": role, "seed": seed, "epoch": e,
                     "online_train_ce": None if e == 0 else ce_by_epoch(e),
                     "eval_objective": ce_by_epoch(max(e, 1)) / 3, "eval_grad_norm": grad,
                     "relative_epoch_update": .04, "heldout_ce": .5 + e / 1000, "heldout_accuracy": .9,
                     "generalization_gap_ce": .4, "interface_tv_prev_mean": None if e == 0 else .01,
                     "interface_tv_from_T0_mean": None if e < T0 else (e - T0) / 1000})
    with (folder / "curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "manifest.json").write_text(json.dumps({
        "schema": report.SCHEMA, "status": "complete", "run": name, "role": role, "seed": seed,
        "all_runs": ["shadow3_seed42", "shadow3_seed43", "target"], "original_epochs": T0,
        "extended_epochs": E, "replay_exact": True}))


class PlateauReportTests(unittest.TestCase):
    def test_plateau_rule(self):
        flat_after_60 = {k: (1. / k if k <= 60 else 1. / 60 * (1 - .01 * (k - 60) / 10)) for k in range(10, 101, 10)}
        start, _ = report.plateau_start(flat_after_60, 50, 100, 10, .10, 1e-9)
        self.assertEqual(start, 60)
        steady_decline = {k: .7 ** (k / 10) for k in range(10, 101, 10)}
        self.assertIsNone(report.plateau_start(steady_decline, 50, 100, 10, .10, 1e-9)[0])
        # One flat pair at the very end is not enough evidence.
        late = {**steady_decline, 100: steady_decline[90]}
        self.assertIsNone(report.plateau_start(late, 50, 100, 10, .10, 1e-9)[0])
        # A window difference inside 2 standard errors of minibatch noise counts as flat.
        noisy = {k: (1. if k < 60 else (.5 if (k // 10) % 2 else .6)) for k in range(10, 101, 10)}
        self.assertIsNone(report.plateau_start(noisy, 50, 100, 10, .10, 1e-9)[0])
        self.assertEqual(report.plateau_start(noisy, 50, 100, 10, .10, 1e-9,
                                              se={k: .05 for k in noisy})[0], 60)
        # Absolute tolerance lets a near-zero loss count as flat.
        tiny = {k: .001 * .5 ** (k / 10) for k in range(10, 101, 10)}
        self.assertEqual(report.plateau_start(tiny, 50, 100, 10, .10, .002)[0], 50)

    def test_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            fake_run(td, "shadow3_seed42", "affected_shadow", 42, lambda e: max(.02, .6 * .9 ** e))
            fake_run(td, "shadow3_seed43", "affected_shadow", 43,
                     lambda e: .02 + (.05 * (1 - (e - 50) / 10) if 50 <= e <= 60 else (.05 if e < 50 else 0)))
            fake_run(td, "target", "target", 42007, lambda e: .03)
            report.main(["--root", td])
            summary = json.loads((Path(td) / "report/plateau_report.json").read_text())
            self.assertEqual(summary["runs"]["shadow3_seed42"]["plateau_start_online_ce"], 50)
            self.assertEqual(summary["runs"]["shadow3_seed43"]["plateau_start_online_ce"], 70)
            self.assertEqual(summary["suggested_E_star"], 70)
            self.assertTrue(summary["fixed_models_flat_by_E_star"]["target"])
            for name in ("curves_affected_shadow.png", "curves_target_and_fixed_shadows.png",
                         "REPORT.md", "window_table.csv"):
                self.assertTrue((Path(td) / "report" / name).is_file(), name)

    def test_report_refuses_incomplete_panel(self):
        with tempfile.TemporaryDirectory() as td:
            fake_run(td, "shadow3_seed42", "affected_shadow", 42, lambda e: .02)
            with self.assertRaises(SystemExit):
                report.main(["--root", td])


if __name__ == "__main__":
    unittest.main(verbosity=2)
