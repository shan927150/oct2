#!/usr/bin/env python3
"""B0 regression: dose pairing, adjacent-rung stability, verdict wording and the 05/10 patches."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
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
spec = importlib.util.spec_from_file_location("b0_test_module", HERE / "14_b0_dose_derivative.py")
b0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b0
spec.loader.exec_module(b0)
legacy, score, pilot = b0.load_modules()
torch.set_num_threads(min(2, torch.get_num_threads()))

PATIENT = {"patient_id": 807, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]}


def write_dose_dir(root: Path, alpha: float, theta0, displacement, *, seed=42, rng="fp",
                   patient=PATIENT, split_hash="abc123", overrides=None):
    """A minimal but structurally faithful 05 output directory."""
    args = {"seeds": [seed], "affected_shadow": 0, "split_seed": 42005, "selection_seed": 42006,
            "target_seed": 42007, "fixed_shadow_seed": 42100, "shadow_epochs": 50, "shadow_lr": 1e-3,
            "shadow_batch_size": 128, "attack_epochs": 50, "deletion_mode": "fixed_mask",
            "window_membership": "value_only", "deterministic": True, "n_total_samples": 40000,
            "target_data_size": 2000, "shadow_data_size": 2000, "n_shadow": 5, "classes": [1, 2],
            "deletion_weight": alpha, "removal_epochs": None, **(overrides or {})}
    legacy.write_json(root / "experiment_config.json",
                      {"training_numerics": pilot.TRAINING_NUMERICS, "args": args,
                       "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
    legacy.write_json(root / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
    legacy.write_json(root / "selected_patients.json", {"split_sha256": split_hash, "patients": [patient]})
    metrics = {"post_train_rng_sha256": rng}
    for name, vector in (("baseline_seed%d" % seed, theta0),
                         ("seed%d_patient%d" % (seed, patient["patient_id"]), theta0 + alpha * displacement)):
        payload = {"state_dict": {"w": vector.clone().float()},
                   "metadata": {"seed": seed, "metrics": metrics, "deletion_weight": alpha,
                                "excluded_indices": patient["raw_indices"],
                                "training_numerics": pilot.TRAINING_NUMERICS}}
        torch.save(payload, root / "checkpoints" / f"shadow_0_{name}.pt")
    (root / "runs").mkdir(parents=True, exist_ok=True)
    p0 = np.full((4, 4), 0.25, dtype=np.float32)
    np.savez_compressed(root / "runs" / f"seed{seed}_patient{patient['patient_id']}_interface.npz",
                        p_baseline=p0, p_loo=p0 + np.float32(alpha) * np.float32(0.01))


def make_ladder(td: Path, alphas, *, curvature=0.0, **kw):
    """theta(alpha) = theta0 + alpha*v + curvature*alpha^2*u : a clean derivative plus a known bend."""
    torch.manual_seed(7)
    theta0 = torch.randn(64, dtype=torch.float64)
    v = torch.randn(64, dtype=torch.float64)
    u = torch.randn(64, dtype=torch.float64)
    roots = []
    for alpha in alphas:
        root = td / f"dose{alpha:g}"
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        write_dose_dir(root, alpha, theta0, v + curvature * alpha * u, **kw)
        roots.append(root)
    return roots


class B0Tests(unittest.TestCase):
    def run_b0(self, roots, out, extra=()):
        argv = ["b0", "--dose_dirs", *map(str, roots), "--out_dir", str(out), "--seeds", "42", *extra]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(b0, "load_modules", return_value=(legacy, score, pilot)):
            b0.main()
        return (legacy.read_json(out / "manifest.json"),
                legacy.read_json(out / "b0_summary.json"))

    def test_linear_ladder_resolves_and_reports_a_band(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = make_ladder(td, [0.1, 0.03, 0.01, 0.003])
            manifest, summary = self.run_b0(roots, td / "out")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["input_files_unchanged"])
            self.assertTrue(summary["verdict"]["resolved_band"])
            rows = list(csv_rows(td / "out" / "b0_rows.csv"))
            self.assertEqual(len(rows), 4)
            self.assertIsNone(rows[0]["larger_alpha"] or None)     # largest dose has no larger neighbour
            for row in rows[1:]:
                self.assertAlmostEqual(float(row["parameter_cosine_vs_larger"]), 1.0, places=9)
                # the residual disagreement is float32 checkpoint storage, not the ladder
                self.assertLess(float(row["parameter_relative_change_vs_larger"]), 1e-5)
                self.assertTrue(row["above_storage_resolution"] in ("True", True))
                self.assertGreater(float(row["displacement_over_floor"]), 100)
            # the derivative itself is dose-independent for a linear ladder
            norms = {float(r["derivative_norm"]) for r in rows}
            self.assertLess((max(norms) - min(norms)) / min(norms), 1e-5)

    def test_displacement_under_the_float32_floor_is_excluded_from_the_gate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # a dose so small that theta(alpha) - theta(0) is pure storage round-off
            roots = make_ladder(td, [0.1, 1e-9])
            _, summary = self.run_b0(roots, td / "out")
            self.assertFalse(summary["verdict"]["resolved_band"])
            self.assertIsNotNone(summary["verdict"]["below_storage_resolution"])
            self.assertIn("float32 checkpoint floor", summary["verdict"]["below_storage_resolution"][0])

    def test_strong_curvature_at_large_dose_is_reported_as_unresolved_not_absent(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = make_ladder(td, [1.0, 0.5], curvature=3.0)
            _, summary = self.run_b0(roots, td / "out")
            self.assertFalse(summary["verdict"]["resolved_band"])
            self.assertIn("not evidence that a local derivative does not exist",
                          summary["verdict"]["statement"])
            self.assertIsNone(summary["verdict"]["smallest_passing_alpha"])

    def test_unpaired_baselines_and_duplicate_alphas_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = make_ladder(td, [0.1, 0.01])
            # a different post-train RNG fingerprint means the doses are not the paired fixed_mask run
            shutil.rmtree(roots[1]); (roots[1] / "checkpoints").mkdir(parents=True)
            torch.manual_seed(7)
            theta0 = torch.randn(64, dtype=torch.float64); v = torch.randn(64, dtype=torch.float64)
            write_dose_dir(roots[1], 0.01, theta0, v, rng="different")
            with self.assertRaises(RuntimeError) as ctx:
                self.run_b0(roots, td / "out_rng")
            self.assertIn("baselines differ", str(ctx.exception))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = make_ladder(td, [0.1, 0.1000001])
            for root in roots:                       # force an exact duplicate alpha
                cfg = legacy.read_json(root / "experiment_config.json")
                cfg["args"]["deletion_weight"] = 0.1
                legacy.write_json(root / "experiment_config.json", cfg)
            with self.assertRaises(RuntimeError) as ctx:
                self.run_b0(roots, td / "out_dup")
            self.assertIn("share alpha", str(ctx.exception))

    def test_mismatched_training_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = make_ladder(td, [0.1])
            more = make_ladder(td / "second", [0.01], overrides={"shadow_epochs": 75})
            with self.assertRaises(RuntimeError) as ctx:
                self.run_b0(roots + more, td / "out")
            self.assertIn("shadow_epochs", str(ctx.exception))

    def test_05_restricts_the_loo_loop_without_touching_the_panel(self):
        """--loo_patients selects from the frozen panel and never re-selects it."""
        source = (HERE / "05_end_to_end_patient_loo_pilot.py").read_text()
        self.assertIn("are not in the frozen panel", source)
        self.assertIn("do not silently substitute another patient", source)
        self.assertIn('"covers_full_panel": args.loo_patients is None', source)
        with mock.patch.object(sys, "argv", ["pilot", "--loo_patients", "807", "2085"]):
            parsed = pilot.parse_args()
        self.assertEqual(parsed.loo_patients, [807, 2085])
        with mock.patch.object(sys, "argv", ["pilot"]):
            self.assertIsNone(pilot.parse_args().loo_patients)

    def test_10_dose_table_extends_without_changing_the_original_conditions(self):
        spec10 = importlib.util.spec_from_file_location("run10", HERE / "10_run_calibration.py")
        run10 = importlib.util.module_from_spec(spec10)
        sys.modules[spec10.name] = run10
        spec10.loader.exec_module(run10)
        self.assertEqual(run10.DOSE_CONDITIONS["dose01"], 0.1)
        self.assertEqual(run10.DOSE_CONDITIONS["dose025"], 0.25)
        self.assertEqual(run10.DOSE_CONDITIONS["dose05"], 0.5)
        self.assertEqual(run10.DOSE_CONDITIONS["dose003"], 0.03)
        self.assertEqual(run10.DOSE_CONDITIONS["dose0001"], 0.001)
        # every dose condition maps to a distinct directory suffix and a distinct alpha
        self.assertEqual(len(set(run10.DOSE_CONDITIONS.values())), len(run10.DOSE_CONDITIONS))


def csv_rows(path):
    import csv
    with Path(path).open() as handle:
        yield from csv.DictReader(handle)


if __name__ == "__main__":
    unittest.main()
