"""Regression tests for estimands, gates, and crossed/nested seed bookkeeping."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from formal_analysis import analyze_ladder, seed_variance_components

spec = importlib.util.spec_from_file_location("seed_stats", HERE / "09_seed_variance.py")
stats = importlib.util.module_from_spec(spec); spec.loader.exec_module(stats)


class FormalAnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=seed, patient_id=pid, oct_class=1, affected_shadow=0,
                     actual_value=pid/10, actual_full=pid/5, actual_relabel=pid/6,
                     L1_lin_value=pid/10, L2_lin_value=pid/10, L3_lin_value=pid/10,
                     L2_retrain_value=pid/10, L3_retrain_value=pid/10,
                     L2_hybrid_full=pid/5, L3_hybrid_full=pid/5,
                     frozen_self=pid/10, frozen_h=pid/10,
                     attack_solve_reliable=False, cg_score_reliable=False, cg_dtheta_reliable=True)
                for pid in (1, 2, 3) for seed in (42, 43)]

    def test_dependency_gates_and_common_samples(self):
        out = analyze_ladder(self.rows(), [42, 43])
        a = out["per_patient_seed"]
        for key in ("L1_lin_value", "L2_lin_value", "L3_lin_value", "frozen_h"):
            self.assertEqual(a[key+"~actual_value"]["n"], 0)
        for key in ("L2_retrain_value", "L3_retrain_value", "frozen_self"):
            self.assertEqual(a[key+"~actual_value"]["n"], 6)
        self.assertEqual(out["hybrid_increment_over_relabel"]["L3_hybrid_full"]["n_common"], 6)
        self.assertGreater(out["hybrid_increment_over_relabel"]["L3_hybrid_full"]["mae_reduction"], 0)
        rows = self.rows()
        for row in rows:
            row.update(attack_solve_reliable=True, cg_score_reliable=True, cg_dtheta_reliable=False)
        a = analyze_ladder(rows)["per_patient_seed"]
        self.assertEqual(a["L3_lin_value~actual_value"]["n"], 6)
        self.assertEqual(a["L3_retrain_value~actual_value"]["n"], 0)

    def test_complete_panel_and_tiny_coverage(self):
        rows = self.rows()[:-1]
        out = analyze_ladder(rows, [42, 43])
        self.assertEqual(out["per_patient_mean"]["L2_retrain_value~actual_value"]["n"], 2)
        tiny = analyze_ladder(rows[:1], [42, 43])
        self.assertEqual(tiny["n_rows_attack_solve_reliable"], 0)
        self.assertEqual(tiny["per_patient_seed"]["L2_retrain_value~actual_value"]["n"], 1)
        self.assertEqual(tiny["per_patient_mean"]["L2_retrain_value~actual_value"]["n"], 0)
        json.dumps(analyze_ladder([]), allow_nan=False)

    def test_grid_and_missing_values(self):
        rows = self.rows()
        for row in rows:
            row.update(attack_solve_reliable=True, cg_score_reliable=True,
                       L3_lin_value_gamma1=0.1, L3_lin_value_gamma1_reliable=False)
        rows[0]["L2_retrain_value"] = float("nan")
        out = analyze_ladder(rows)
        self.assertEqual(list(out["damping_sensitivity"]), ["L3_lin_value_gamma1"])
        self.assertEqual(out["damping_sensitivity"]["L3_lin_value_gamma1"]["n"], 0)
        self.assertEqual(out["per_patient_seed"]["L2_retrain_value~actual_value"]["n"], 5)
        json.dumps(out, allow_nan=False)

    def test_crossed_anova_known_components(self):
        r = np.array([-2., 0., 2.]); k = np.array([-3., 0., 3.])
        interaction = .1*np.outer([1., -2., 1.], [1., -2., 1.])
        x = r[:, None] + k[None, :] + interaction + 7
        out = seed_variance_components(x, "crossed_fixed_panel")
        self.assertAlmostEqual(out["raw_components"]["stage1"], 3.97)
        self.assertAlmostEqual(out["raw_components"]["attack"], 8.97)
        self.assertAlmostEqual(out["raw_components"]["interaction_and_cell_residual"], .09)
        self.assertAlmostEqual(out["variance_grand_mean"], 3.97/3 + 8.97/3 + .09/9)
        self.assertAlmostEqual(out["mean_paired_delta"], 7)

    def test_nested_and_unidentified_cases(self):
        out = seed_variance_components([[1, 2, 3], [5, 6, 7]], "nested_derived_from_stage1_seed")
        self.assertAlmostEqual(out["variance_grand_mean"], 4.)
        self.assertEqual(seed_variance_components([[1], [2]], "crossed_fixed_panel")["status"], "combined_only_K1")
        self.assertIsNone(seed_variance_components([[1, 2]], "crossed_fixed_panel")["se_grand_mean"])
        with self.assertRaises(ValueError):
            seed_variance_components([[1, np.nan], [2, 3]], "crossed_fixed_panel")
        out = seed_variance_components([[1, -1], [-1, 1]], "crossed_fixed_panel")
        self.assertLess(out["raw_components"]["stage1"], 0)
        self.assertEqual(out["components_nonnegative"]["stage1"], 0)

    def test_real_json_pairing_and_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"runs").mkdir()
            (root/"experiment_config.json").write_text(json.dumps({"args": {"seeds": [42, 43]}}))
            for r in (42, 43):
                conditions = {}
                for c, offset in (("J00", 0.), ("J10", .1), ("J01", .2), ("J11", .4)):
                    conditions[c] = {"per_class": {"1": {
                        "attack_seeds": [5101, 5102], "attack_seed_design": "crossed_fixed_panel",
                        "per_rep_cross_entropy": [r + offset, 2*r + offset],
                        "per_rep_auc": [.5 + offset, .55 + offset]}}}
                payload = {"seed": r, "patient": {"patient_id": 7, "oct_class": 1},
                           "affected_shadow": 0, "split_sha256": "abc", "conditions": conditions}
                (root/"runs"/f"seed{r}_patient7.json").write_text(json.dumps(payload))
            raw, fits = stats.read_effects(root)
            self.assertEqual(len(raw), 40)
            value = next(v for v in fits if v["metric"] == "cross_entropy" and v["effect"] == "value")
            self.assertAlmostEqual(value["mean_paired_delta"], .1)
            (root/"runs"/"seed43_patient7.json").unlink()
            self.assertTrue(all(v["status"] == "incomplete_grid" for v in stats.read_effects(root)[1]))
            path = root/"runs"/"seed42_patient7.json"
            payload = json.loads(path.read_text())
            payload["conditions"]["J10"]["per_class"]["1"]["attack_seeds"] = [5102, 5101]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                stats.read_effects(root)

    def test_launcher_dry_run_condition_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"panel").mkdir()
            proposal = {"panel_complete": True, "proposed_affected_shadows": [1],
                        "args": {"patients_per_class_per_shadow": 4}}
            (root/"panel"/"eligibility_preflight.json").write_text(json.dumps(proposal))
            base = [sys.executable, str(HERE/"10_run_calibration.py"), "--root", temp, "--dry_run"]
            late = subprocess.check_output(base + ["--phase", "truth", "--condition", "late"], text=True)
            self.assertIn("--removal_epochs 25:50", late)
            self.assertIn("--window_membership value_only", late)
            self.assertIn("--attack_seeds 5101 5102 5103 5104 5105", late)
            self.assertIn("09_seed_variance.py", late)
            dose = subprocess.check_output(base + ["--phase", "truth", "--condition", "dose025"], text=True)
            self.assertIn("--deletion_weight 0.25", dose)
            self.assertNotIn("--removal_epochs", dose)
            preflight = subprocess.check_output(base + ["--phase", "preflight", "--split_seed", "99"], text=True)
            self.assertIn("--split_seed 99", preflight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
