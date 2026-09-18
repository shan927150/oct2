#!/usr/bin/env python3
"""Pure-CPU tests for manual E* freezing, Step-2 gates and matched comparison."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = load("a20_report_control_test", "20_A_convergence_report.py")
freeze = load("a20_freeze_control_test", "20_A_freeze_epoch.py")
control = load("a21_control_test", "21_A_step2_control.py")
compare = load("a22_compare_control_test", "22_A_compare_E.py")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_convergence(root):
    names = [f"shadow3_seed{seed}" for seed in range(42, 47)]
    source = {"scripts/cross_stage/20_A_convergence_curve.py": "source"}
    controlled = {"epochs": 100, "checkpoint_epochs": list(range(50, 101, 5)),
                  "gradient_every": 5, "dropout_mc_every": 25, "dropout_mc_reps": 16,
                  "hvp_batch": 64, "eval_batch": 256, "spectrum_epochs": []}
    for seed, name in zip(range(42, 47), names):
        folder = Path(root) / name
        (folder / "checkpoints").mkdir(parents=True)
        rows = [{"epoch": epoch, "seed": seed,
                 "online_train_ce": None if epoch == 0 else .02 + .5 * .95**epoch,
                 "eval_objective": .03 + .3 * .96**epoch, "eval_grad_norm": .3,
                 "relative_epoch_update": None if epoch == 0 else .04,
                 "heldout_ce": .4 + epoch / 10000, "heldout_accuracy": .8,
                 "generalization_gap_ce": .2,
                 "interface_tv_prev_mean": None if epoch == 0 else .01,
                 "interface_tv_from_T0_mean": None if epoch < 50 else (epoch - 50) / 1000}
                for epoch in range(101)]
        with (folder / "curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (folder / "checkpoints/epoch070.pt").write_bytes(f"checkpoint-{seed}".encode())
        write_json(folder / "manifest.json", {
            "schema": report.SCHEMA, "status": "complete", "run": name,
            "role": "affected_shadow", "seed": seed, "all_runs": names,
            "original_epochs": 50, "extended_epochs": 100, "git_commit": "fake-commit",
            "source_sha256": source, "args": controlled, "replay_exact": True,
            "input_files_unchanged": True, "source_files_unchanged": True,
            "dataset_arrays_unchanged": True,
        })
        write_json(folder / "dataset_fingerprint.json", {"X_sha256": "x", "y_sha256": "y"})
        write_json(folder / "input_sha256.json", {"/baseline/common.json": "input"})


def truth_complete(root, condition):
    folder = Path(root) / f"shadow3_{condition}"
    state = json.loads((Path(root) / "A_STEP2_STATE.json").read_text())
    write_json(folder / "experiment_summary.json", {"status": "complete"})
    expected = {"target": 1, "fixed_shadow": 4, "affected_baseline_or_noop": 10, "loo": 40}
    checks = []
    for role, count in expected.items():
        checks.extend({"role": role, "passed": True,
                       "regenerated_optimizer_sha256": f"adam-T0-{role}-{i}",
                       "continued_optimizer_sha256_at_E_star": f"adam-E-{role}-{i}"}
                      for i in range(count))
    anchors = [{"passed": True} for _ in range(10)]
    write_json(folder / "route_a_epoch50_replay_checks.json", {
        "schema": "pathway2_A_epoch50_replay_v2", "status": "complete",
        "original_epochs": state["original_epochs"], "extended_epochs": state["E_star"],
        "selection_sha256": state["selection_sha256"],
        "expected_counts": expected, "actual_counts": expected, "checks": checks,
        "expected_E_star_anchor_checks": 10, "E_star_anchor_checks": anchors,
    })


class ManualGateTests(unittest.TestCase):
    def test_freeze_prepare_and_manual_gates(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            convergence = base / "convergence"
            make_convergence(convergence)
            report.main(["--root", str(convergence)])
            descriptive = json.loads((convergence / "report/plateau_report.json").read_text())
            self.assertEqual(descriptive["selection_status"], "not_frozen")
            self.assertIsNone(descriptive["selected_E_star"])
            freeze.main(["--root", str(convergence), "--epoch", "70",
                         "--confirmed_by", "test review", "--note", "all five curves reviewed"])
            selection = convergence / "frozen_E_star.json"
            frozen = json.loads(selection.read_text())
            self.assertEqual(frozen["E_star"], 70)
            self.assertTrue(frozen["selection_was_not_automatic"])
            self.assertEqual(len(frozen["checkpoint_sha256"]), 5)

            baseline = base / "baseline"
            write_json(baseline / "results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json",
                       {"panel_complete": True})
            runs = base / "runs/A"
            runs.mkdir(parents=True)

            def fake_git(*args):
                if args == ("rev-parse", "HEAD"):
                    return "fake-commit"
                if args == ("status", "--porcelain", "--untracked-files=no"):
                    return ""
                raise AssertionError(args)

            preflight = {
                "status": "PASS_FILE_INTEGRITY_ONLY", "commit": "fake-commit",
                "baseline_root": str(baseline.resolve()), "proposed_output_root": str(runs.resolve()),
            }
            with mock.patch.object(control, "git", side_effect=fake_git), \
                 mock.patch.object(control, "run_preflight", return_value=preflight):
                control.prepare(argparse.Namespace(selection=str(selection), baseline_root=str(baseline),
                                                   runs_root=str(runs)))
            attempts = list(runs.glob("l3_at_E70_*"))
            self.assertEqual(len(attempts), 1)
            root = attempts[0]

            with mock.patch.object(control, "git", side_effect=fake_git), \
                 mock.patch.object(control, "run_preflight", return_value=preflight):
                with self.assertRaises(SystemExit):
                    control.check(argparse.Namespace(root=str(root), action="score"))
                control.check(argparse.Namespace(root=str(root), action="test"))
                write_json(root / "step2_gpu_smoke/SMOKE_COMPLETE.json", {
                    "schema": "pathway2_A_step2_gpu_smoke_v1", "status": "complete",
                    "git_commit": "fake-commit", "original_epochs": 2, "extended_epochs": 3,
                    "actual_counts": {"target": 1, "fixed_shadow": 1,
                                      "affected_baseline_or_noop": 2, "loo": 2},
                })
                control.check(argparse.Namespace(root=str(root), action="truth-full"))
                control.record(argparse.Namespace(root=str(root), action="truth-full", job_id="12345"))
                control.job_check(argparse.Namespace(root=str(root), action="truth-full"))
                with self.assertRaises(SystemExit):
                    control.check(argparse.Namespace(root=str(root), action="truth-full"))

                truth_complete(root, "full")
                control.check(argparse.Namespace(root=str(root), action="score"))
                score = root / "shadow3_full/score_ladder_A0.2_S1"
                write_json(score / "score_config.json", {
                    "damping_attack": .2, "damping_shadow": 1., "damping_shadow_grid": []})
                write_json(score / "ladder_summary.json", {})
                (score / "ladder_rows.csv").write_text("seed,patient_id,oct_class\n42,807,1\n")
                control.check(argparse.Namespace(root=str(root), action="compare-primary"))
                with self.assertRaises(SystemExit):
                    control.check(argparse.Namespace(root=str(root), action="approve-a2"))
                with self.assertRaises(SystemExit):
                    control.check(argparse.Namespace(root=str(root), action="truth-dose"))

                write_json(root / "compare_primary_vs_E50/comparison.json", {
                    "schema": "pathway2_A_matched_E_comparison_v2", "mode": "primary", "E_star": 70,
                    "provenance": {"status": "PASS"},
                })
                (root / "compare_primary_vs_E50/COMPARE.md").write_text("reviewed")
                control.check(argparse.Namespace(root=str(root), action="truth-dose"))
                truth_complete(root, "dose01")
                control.check(argparse.Namespace(root=str(root), action="preflight"))
                write_json(root / "diag_preflight/manifest.json", {
                    "status": "complete", "args": {"mode": "preflight"}})
                (root / "diag_preflight/checkpoint_ratios.csv").write_text("seed\n42\n")
                control.approve(argparse.Namespace(root=str(root), note="A1 reviewed; run A2 sensitivity"))
                control.check(argparse.Namespace(root=str(root), action="damping"))
                self.assertTrue((root / "A2_DAMPING_APPROVAL.json").is_file())

                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    control.status(argparse.Namespace(root=str(root)))
                status = json.loads(output.getvalue())
                self.assertEqual(status["stages"]["A1_truth_full"]["summary"], "complete")
                self.assertEqual(status["stages"]["A2_approval"], "present")


class MatchedComparisonTests(unittest.TestCase):
    @staticmethod
    def row(seed, patient, actual, shift=0.):
        row = {"seed": seed, "patient_id": patient, "oct_class": 1,
               "actual_value": actual + shift, "dtheta_cosine": .1 * seed + shift,
               "attack_solve_reliable": True, "cg_score_reliable": True,
               "cg_dtheta_reliable": True}
        for index, method in enumerate(compare.METHODS, start=1):
            row[method] = index * actual + shift
        return row

    def test_ladder_uses_same_cells_at_both_epochs(self):
        reference = [self.row(seed, 800 + seed, actual) for seed, actual in ((1, .1), (2, .2), (3, .4))]
        extended = [self.row(seed, 800 + seed, actual, .01) for seed, actual in ((1, .1), (2, .2), (3, .4))]
        result = compare.matched_ladder(reference, extended)
        self.assertEqual(result["n_total_cells_each"], 3)
        self.assertEqual(result["all_method_common"]["n"], 3)
        self.assertEqual(result["method_specific_matched"]["L3_lin_value"]["n_matched"], 3)
        with self.assertRaises(RuntimeError):
            compare.matched_ladder(reference, extended[:-1])

    def test_damping_uses_overlap_and_labels_new_gamma(self):
        with tempfile.TemporaryDirectory() as td:
            ref, new = Path(td) / "ref", Path(td) / "new"
            ref_grid = [.03, .1, .3, .7, 1., 2.]
            new_grid = [.01, .03, .1, .3, 1., 2.]
            for folder, grid, shift in ((ref, ref_grid, 0.), (new, new_grid, .01)):
                write_json(folder / "manifest.json", {
                    "status": "complete", "args": {"mode": "damping", "damping_grid": grid}})
                rows = []
                for condition in ("full", "dose01"):
                    for gamma in grid:
                        for seed in (42, 43, 44):
                            rows.append({"condition": condition, "gamma": gamma, "seed": seed,
                                         "patient_id": 800 + seed, "oct_class": 1, "qualified": True,
                                         "cosine": .2 + shift, "norm_ratio": .5,
                                         "pred_h_projection": seed / 100 + shift,
                                         "true_h_projection": seed / 90,
                                         "rhs_norm": 1e-4 + seed * 1e-7})
                with (folder / "all_rows.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            result = compare.matched_damping(ref, new)
            self.assertIn("full_gamma0.01", result["new_only_no_E50_comparator"])
            self.assertIn("full_gamma0.7", result["reference_only"])
            self.assertEqual(result["matched_overlap"]["full_gamma0.1"]["n_matched_qualified"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
