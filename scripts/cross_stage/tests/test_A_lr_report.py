#!/usr/bin/env python3
"""Regression tests for missing tasks, pairing evidence and tail-window claims."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "23_A_lr_report.py"
SPEC = importlib.util.spec_from_file_location("lr_report", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture():
    runs = {}
    for seed in report.EXPECTED_SEEDS:
        for arm in report.EXPECTED_ARMS:
            name = f"shadow3_seed{seed}_{arm}"
            m = dict(schema=report.SCHEMA, status="complete", run=name, base_run=f"shadow3_seed{seed}",
                     seed=seed, arm=arm, resume_epoch=100, total_epochs=140, n_curve_rows=41,
                     base_lr=.001, weight_decay=1e-5, gradient_target_exploratory=.001,
                     input_files_unchanged=True, source_files_unchanged=True, dataset_arrays_unchanged=True,
                     source_sha256={"core.py": "source"}, git_commit="commit",
                     epoch_order_sha256=f"orders{seed}", final_rng_sha256=f"end{seed}",
                     resume_check={"passed": True, "rng_fingerprint": f"start{seed}"},
                     args={"gradient_every": 1})
            curve = [dict(epoch=e, eval_grad_norm=.0008, relative_epoch_update=.001,
                          eval_objective=.003, heldout_ce=2., interface_tv_from_T0_mean=.02,
                          dropout_mc_mean_gradient_norm=.02, dropout_mc_gradient_rms_spread=.03,
                          dropout_mc_reps=16) for e in range(100, 141)]
            runs[name] = dict(manifest=m, curve=curve, input_sha256={f"checkpoint{seed}": "hash"},
                              dataset_fingerprint={"X_sha256": "data", "y_sha256": "labels"})
    return runs


def write_runs(root, runs):
    for name, run in runs.items():
        out = root / name
        out.mkdir()
        for field in ("manifest", "input_sha256", "dataset_fingerprint"):
            (out / f"{field}.json").write_text(json.dumps(run[field]))
        with (out / "curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(run["curve"][0]))
            writer.writeheader()
            writer.writerows(run["curve"])


class ReportTests(unittest.TestCase):
    def test_complete_panel_and_mc_error(self):
        result = report.summarize(fixture(), 10)
        self.assertTrue(result["pairing_complete"])
        self.assertEqual(len(result["per_run"]), 10)
        for row in result["per_run"].values():
            self.assertTrue(row["target_met_tail_median"])
            self.assertAlmostEqual(row["dropout_mc_last"]["mc_mean_gradient_rms_error_estimate"],
                                   .03 / math.sqrt(15))

    def test_missing_arm_and_missing_seed_are_not_complete(self):
        full = fixture()
        for names in (["shadow3_seed42_control"], ["shadow3_seed42_control", "shadow3_seed42_decay"],
                      [n for n in full if n != "shadow3_seed46_decay"]):
            with self.subTest(names=names):
                result = report.summarize({n: full[n] for n in names}, 10)
                self.assertFalse(result["pairing_complete"])
                self.assertTrue(result["problems"])
                self.assertTrue(all(r["target_met_tail_median"] is None for r in result["per_run"].values()))

    def test_bad_pairing_and_missing_epoch(self):
        for field in ("input_sha256", "dataset_fingerprint", "epoch_order_sha256", "final_rng_sha256"):
            runs = fixture()
            run = runs["shadow3_seed42_decay"]
            if field in run:
                run[field] = {"wrong": "value"}
            else:
                run["manifest"][field] = "wrong"
            with self.subTest(field=field):
                self.assertFalse(report.summarize(runs, 10)["pairing_complete"])
        for mode in ("missing", "duplicate", "nan"):
            runs = fixture()
            curve = runs["shadow3_seed42_control"]["curve"]
            if mode == "missing":
                curve.pop(-2)
            elif mode == "duplicate":
                curve[-2]["epoch"] = 140
            else:
                curve[-1]["eval_grad_norm"] = float("nan")
            with self.subTest(mode=mode):
                self.assertFalse(report.summarize(runs, 10)["pairing_complete"])

    def test_sparse_tail_does_not_reach_into_earlier_epochs(self):
        runs = fixture()
        for run in runs.values():
            run["manifest"]["args"]["gradient_every"] = 5
            for r in run["curve"]:
                if r["epoch"] % 5:
                    r["eval_grad_norm"] = None
        result = report.summarize(runs, 10)
        self.assertTrue(result["pairing_complete"])
        row = result["per_run"]["shadow3_seed42_control"]
        self.assertEqual(row["tail_gradient_count"], 2)
        self.assertIsNone(row["target_met_tail_median"])

    def test_main_refuses_missing_tasks_and_marks_partial_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs = fixture()
            write_runs(root, {"shadow3_seed42_control": runs["shadow3_seed42_control"]})
            with self.assertRaisesRegex(SystemExit, "Missing seed/arm"):
                report.main(["--root", str(root)])
            self.assertFalse((root / "report").exists())
            with mock.patch.object(report, "plot"):
                report.main(["--root", str(root), "--allow_incomplete"])
            summary = json.loads((root / "report/probe_summary.json").read_text())
            self.assertFalse(summary["pairing_complete"])
            self.assertIsNone(summary["per_run"]["shadow3_seed42_control"]["target_met_tail_median"])

    def test_small_fixture_requires_explicit_expected_panel(self):
        runs = {n: v for n, v in fixture().items() if v["manifest"]["seed"] == 42}
        self.assertTrue(report.summarize(runs, 10, [42])["pairing_complete"])
        with self.assertRaises(ValueError):
            report.summarize(runs, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
