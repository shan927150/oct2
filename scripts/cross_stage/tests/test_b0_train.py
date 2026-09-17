#!/usr/bin/env python3
"""SmallCNN integration: frozen replay gate, all dose cells, read-only inputs."""
import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture = load("train_fixture", HERE / "tests/test_e0_residual_decomposition.py")
train = load("train_b0", HERE / "14_b0_train.py")
b0 = load("train_analysis_b0", HERE / "14_b0_dose_derivative.py")
legacy, score, pilot = train.e0.load_modules()
torch.set_num_threads(min(2, torch.get_num_threads()))


class B0TrainTests(unittest.TestCase):
    def test_original_training_function_is_structurally_unchanged(self):
        path = "scripts/cross_stage/05_end_to_end_patient_loo_pilot.py"
        original = subprocess.check_output(["git", "show", "ba8db97:" + path], cwd=HERE, text=True)
        current = (HERE / "05_end_to_end_patient_loo_pilot.py").read_text()
        def functions(source):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef)}
        a, b = functions(original), functions(current)
        for name in ("train_classifier_from_orders", "train_or_load_stage1", "make_epoch_orders", "seed_everything"):
            self.assertEqual(a[name], b[name], name)

    def test_replay_then_eight_cells_use_original_epochs_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            X, y, groups, full, dose = fixture.build_fixture(td)
            before = {str(p): legacy.sha(p) for root in (full, dose) for p in root.rglob('*') if p.is_file()}
            output = td / "b0"
            argv = ["b0", "--full_dir", str(full), "--dose_dir", str(dose), "--data_dir", str(td / "images"),
                    "--out_dir", str(output), "--seed", "42", "--patients", "101", "102"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(train.e0, "load_modules", return_value=(legacy, score, pilot)), \
                 mock.patch.object(pilot, "load_dataset", return_value=(X, y, groups)), \
                 mock.patch.object(pilot, "train_classifier_from_orders", wraps=pilot.train_classifier_from_orders) as call:
                train.main()
            self.assertEqual(call.call_count, 9)
            self.assertTrue(all(len(c.kwargs["train_orders"]) == 2 for c in call.call_args_list))
            after = {p: legacy.sha(Path(p)) for p in before}
            self.assertEqual(before, after)
            manifest = legacy.read_json(output / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["noop"]["optimizer_rng_epochs_exact"], [1, 2])
            roots = [full, dose] + [output / f"shadow0_{c}" for c in train.DOSES]
            with mock.patch.object(sys, "argv", ["b0", "--dose_dirs", *map(str, roots), "--out_dir", str(output / "analysis"),
                                                "--seeds", "42", "--patients", "101", "102"]):
                b0.main()
            result = legacy.read_json(output / "analysis/b0_summary.json")
            self.assertEqual(result["verdict"]["n_cells"], 2)

    def test_noop_mismatch_stops_before_any_perturbed_training(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            X, y, groups, full, dose = fixture.build_fixture(td)
            ctx = legacy.load_context
            args = argparse.Namespace(full_dir=str(full), dose_dir=str(dose), data_dir=str(td / "images"), mode="damping", seeds=[42])
            with mock.patch.object(pilot, "load_dataset", return_value=(X, y, groups)):
                context = ctx(args, pilot)
            path = full / "checkpoints/shadow_0_baseline_seed42.pt"
            payload = torch.load(path, weights_only=False)
            next(iter(payload["state_dict"].values())).view(-1)[0] += .1
            torch.save(payload, path)
            run_args = argparse.Namespace(seed=42, patients=[101, 102], data_dir=str(td / "images"))
            with mock.patch.object(pilot, "train_classifier_from_orders", wraps=pilot.train_classifier_from_orders) as call:
                with self.assertRaisesRegex(RuntimeError, "no-op parameters"):
                    train.train_cells(run_args, context, legacy, pilot, td / "b0")
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
