#!/usr/bin/env python3
"""Submission safety tests using mocked Slurm. No cluster calls are made."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "23_A_lr_control.py"
SPEC = importlib.util.spec_from_file_location("lr_control", PATH)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def attempt(root):
    cfg = dict(root=str(root), repo=str(control.REPO), git_commit="new", input_sha256={},
               baseline_root="/original", runs_root="/runs/A", data_dir="/data", step1_root="/step1",
               resume_epoch=100, extra_epochs=40, decay_schedule="20@1e-4,20@1e-5", account="account")
    control.write_new(root / "probe_config.json", cfg)
    return cfg


def marker(root):
    control.write_new(root / "test_submission.json", {"job_id": "123"})
    control.write_new(root / "gpu_tests_passed.json", dict(job_id="123", git_commit="new",
                       config_sha256=control.sha(root / "probe_config.json")))


class ControlTests(unittest.TestCase):
    def test_sacct_requires_the_exact_job_completed_zero(self):
        for output, expected in [("123|COMPLETED|0:0|", True), ("123|RUNNING|0:0|", False),
                                  ("123|FAILED|1:0|", False), ("123|COMPLETED|1:0|", False),
                                  ("123.batch|COMPLETED|0:0|", False), ("", False)]:
            with self.subTest(output=output), mock.patch.object(control, "command", return_value=output):
                self.assertEqual(control.job_success("123"), expected)

    def test_run_blocked_without_matching_pass_and_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = attempt(root)
            with mock.patch.object(control, "load_config", return_value=(root, cfg)), \
                    mock.patch.object(control, "command") as command:
                with self.assertRaises(FileNotFoundError):
                    control.submit(root, "run")
                command.assert_not_called()
                marker(root)
                with mock.patch.object(control, "job_success", return_value=False):
                    with self.assertRaisesRegex(ValueError, "COMPLETED 0:0"):
                        control.submit(root, "run")
                command.assert_not_called()
                (root / "gpu_tests_passed.json").write_text(json.dumps({"job_id": "wrong"}))
                with self.assertRaisesRegex(ValueError, "marker"):
                    control.submit(root, "run")
                command.assert_not_called()

    def test_test_submits_only_one_job_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = attempt(root)
            with mock.patch.object(control, "load_config", return_value=(root, cfg)), \
                    mock.patch.object(control, "preflight", return_value={"status": "PASS_FILE_INTEGRITY_ONLY"}), \
                    mock.patch.object(control, "command", return_value="123;cluster") as command:
                self.assertEqual(control.submit(root, "test"), "123")
                command.assert_called_once()
                self.assertNotIn("--array", " ".join(command.call_args.args[0]))
                self.assertFalse((root / "run_submission.json").exists())
                self.assertEqual(command.call_args.kwargs["env"]["PROJECT_DIR"], str(control.REPO))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    control.submit(root, "test")
                command.assert_called_once()

    def test_run_uses_requested_concurrency_after_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = attempt(root)
            marker(root)
            with mock.patch.object(control, "load_config", return_value=(root, cfg)), \
                    mock.patch.object(control, "job_success", return_value=True), \
                    mock.patch.object(control, "preflight", return_value={"status": "PASS_FILE_INTEGRITY_ONLY"}), \
                    mock.patch.object(control, "command", return_value="124") as command:
                self.assertEqual(control.submit(root, "run", 2), "124")
                self.assertIn("--array=0-9%2", command.call_args.args[0])

    def test_source_and_input_changes_block_submission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = attempt(root)
            with mock.patch.object(control, "git", return_value="old"):
                with self.assertRaisesRegex(ValueError, "commit changed"):
                    control.load_config(root)
            source = root / "source.pt"
            source.write_bytes(b"initial")
            cfg["input_sha256"] = {str(source): control.sha(source)}
            (root / "probe_config.json").write_text(json.dumps(cfg))
            source.write_bytes(b"changed")
            with mock.patch.object(control, "git", side_effect=lambda *args: "new" if args[0] == "rev-parse" else ""):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    control.load_config(root)

    def test_missing_approval_note_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual authorization"):
            control.prepare("   ")

    def test_prepare_freezes_inputs_without_submitting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            full = root / "baseline/results/cross_stage_calibration_v4_1/shadow3_full"
            (full / "splits").mkdir(parents=True)
            control.write_new(full / "experiment_config.json", {"args": {"seeds": control.SEEDS, "affected_shadow": 3}})
            control.write_new(full / "splits/fresh_patient_split.json", {"split": "test"})
            step1 = root / "step1"
            for seed in control.SEEDS:
                run = step1 / f"shadow3_seed{seed}"
                (run / "checkpoints").mkdir(parents=True)
                control.write_new(run / "manifest.json", dict(status="complete", replay_exact=True,
                                  seed=seed, extended_epochs=100))
                control.write_new(run / "input_sha256.json", {"input": "hash"})
                control.write_new(run / "dataset_fingerprint.json", {"X_sha256": "hash"})
                (run / "checkpoints/epoch100.pt").write_bytes(b"fixture")
            env = dict(OCT_BASELINE_ROOT=str(root / "baseline"), OCT_RUNS_ROOT=str(root / "runs/A"),
                       OCT_DATA_DIR=str(root / "data"), OCT_A_STEP1_ROOT=str(step1))
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(control, "git", side_effect=lambda *args: "new" if args[0] == "rev-parse" else ""), \
                    mock.patch.object(control, "preflight", return_value={"status": "PASS_FILE_INTEGRITY_ONLY"}), \
                    mock.patch.object(control, "command") as command:
                out = control.prepare("test authorization")
                cfg = control.read(out / "probe_config.json")
                self.assertEqual(len(cfg["input_sha256"]), 22)
                self.assertEqual(cfg["decay_schedule"], "20@1e-4,20@1e-5")
                self.assertFalse((out / "test_submission.json").exists())
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
