#!/usr/bin/env python3
"""Regression tests for float32 residuals, shifted indefinite solves and OCT wiring."""
import copy
import importlib.util
import json
import math
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
import stage1_diagnostic_core as core
import stage1_krylov_audit as audit
spec = importlib.util.spec_from_file_location("stage1_v11_test", HERE / "12_stage1_diagnostics_v11.py")
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
legacy = diag.legacy
score = legacy.module_from_path("score07_v11_test", HERE / "07_cross_stage_score_ladder.py")
pilot = score.import_pilot_module()
DEVICE = pilot.DEVICE
torch.set_num_threads(min(2, torch.get_num_threads()))


class Stage1V11Tests(unittest.TestCase):
    def test_delta_requires_actual_cuda(self):
        if os.environ.get("STAGE1_DIAG_REQUIRE_CUDA") == "1":
            self.assertTrue(torch.cuda.is_available())
            self.assertEqual(str(DEVICE).split(":")[0], "cuda")

    def test_shared_basis_matches_dense_spd_and_indefinite_systems(self):
        torch.manual_seed(81)
        U, _ = torch.linalg.qr(torch.randn(9, 9, dtype=torch.float64, device=DEVICE))
        eig = torch.tensor([-.6, -.2, -.03, .001, .08, .4, 1., 3., 10.], dtype=torch.float64, device=DEVICE)
        H = (U * eig) @ U.T
        b = torch.randn(9, dtype=H.dtype, device=DEVICE)
        kry = audit.lanczos_from_rhs(lambda x: H @ x, b, 9)
        for gamma in (.01, .1, .3, .7, 1., 2.):
            result = audit.solve_shift(lambda x: H @ x, b, kry, gamma, 9)
            expected = torch.linalg.solve(H + gamma * torch.eye(9, device=DEVICE, dtype=H.dtype), b)
            torch.testing.assert_close(result["x"], expected, rtol=1e-9, atol=1e-9)
            self.assertEqual(result["x"].device, b.device)
            self.assertLess(result["true_relative_residual"], 1e-10)
            self.assertTrue(audit.qualify_pair(result, result, kry['orthogonality_error'], 1e-3, .01)
                            ['linear_solve_qualified'])

    def test_float32_qualification_uses_real_not_projected_residual(self):
        torch.manual_seed(41)
        U, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))
        eig = torch.cat([torch.tensor([-.1], dtype=torch.float64), torch.logspace(-4, 1, 31, dtype=torch.float64)])
        H = ((U * eig) @ U.T).float().to(DEVICE)
        H = (H + H.T) / 2
        b = torch.randn(32).to(DEVICE)
        gamma = .100001
        kry = audit.lanczos_from_rhs(lambda x: H @ x, b, 32)
        solved = audit.solve_shift(lambda x: H @ x, b, kry, gamma, 32)
        expected = core.norm((H @ solved["x"]).double() + gamma * solved["x"].double() - b.double()) / core.norm(b)
        self.assertAlmostEqual(solved["true_relative_residual"], expected, places=12)
        # Deterministic decision test, independent of hardware-specific roundoff.
        misleading = {**solved, "projected_relative_residual": 0., "true_relative_residual": .02}
        gate = audit.qualify_pair(misleading, misleading, 0., 1e-3, .01)
        self.assertFalse(gate["linear_solve_qualified"])

    def test_depth_instability_is_rejected_even_with_small_residual(self):
        x = torch.tensor([1., 2.], device=DEVICE)
        long = {"x": x, "finite": True, "true_relative_residual": 1e-5}
        short = {**long, "x": 2 * x}
        self.assertFalse(audit.qualify_pair(short, long, 0., 1e-3, .01)["linear_solve_qualified"])

    def test_projected_singular_case_returns_least_squares_iterate(self):
        H = torch.tensor([[0., 1.], [1., 0.]], device=DEVICE, dtype=torch.float64)
        b = torch.tensor([1., 0.], device=DEVICE, dtype=H.dtype)
        kry = audit.lanczos_from_rhs(lambda x: H @ x, b, 2)
        one = audit.solve_shift(lambda x: H @ x, b, kry, 0., 1)
        two = audit.solve_shift(lambda x: H @ x, b, kry, 0., 2)
        self.assertTrue(one["finite"])
        self.assertAlmostEqual(one["true_relative_residual"], 1.)
        torch.testing.assert_close(two["x"], torch.tensor([0., 1.], device=DEVICE, dtype=H.dtype))

    def test_no_false_cdf_interval_or_damping_share_claim(self):
        H = torch.diag(torch.tensor([0., 2.], dtype=torch.float64, device=DEVICE))
        b = torch.ones(2, dtype=H.dtype, device=DEVICE)
        kry = audit.lanczos_from_rhs(lambda x: H @ x, b, 1)
        report = audit.weighted_spectrum(kry, 1, [1.])
        self.assertNotIn("mass_below_bracket", report)
        self.assertIn("No certified CDF interval", report["note"])
        w = .5 / (1000 - .5)
        b = torch.tensor([math.sqrt(w), math.sqrt(1-w)], dtype=H.dtype, device=DEVICE)
        H = torch.diag(torch.tensor([-.999, 1.], dtype=H.dtype, device=DEVICE))
        x = torch.linalg.solve(H + torch.eye(2, device=DEVICE), b)
        self.assertAlmostEqual(float(b @ x / (b @ b)), 1., places=9)
        self.assertLess(core.cosine(x, b), .05)

    def test_reverse_check_recovers_gamma_and_handles_constraints_and_zeros(self):
        H = torch.diag(torch.tensor([.2, 1., 3.], dtype=torch.float64, device=DEVICE))
        b = torch.tensor([.1, -.3, .8], dtype=H.dtype, device=DEVICE)
        d = torch.linalg.solve(H + 1.2 * torch.eye(3, device=DEVICE, dtype=H.dtype), .1 * b)
        r = audit.reverse_residual(lambda x: H @ x, d, b, .1, [.1, 1.2])
        self.assertAlmostEqual(r["implied_gamma_unconstrained"], 1.2, places=12)
        self.assertLess(r["nonnegative_residual"]["relative_to_rhs"], 1e-12)
        d = torch.linalg.solve(H - .1 * torch.eye(3, device=DEVICE, dtype=H.dtype), b)
        r = audit.reverse_residual(lambda x: H @ x, d, b, 1., [1.])
        self.assertAlmostEqual(r["implied_gamma_unconstrained"], -.1, places=12)
        self.assertEqual(r["best_nonnegative_gamma"], 0.)
        zero = audit.reverse_residual(lambda x: H @ x, torch.zeros_like(b), torch.zeros_like(b), .1, [1.])
        self.assertIsNone(zero["implied_gamma_unconstrained"])
        self.assertIsNone(zero["grid_residuals"]["1.0"]["relative_to_rhs"])

    def test_zero_rhs_is_reported_without_fake_direction(self):
        b = torch.zeros(3, device=DEVICE)
        kry = audit.lanczos_from_rhs(lambda x: x, b, 3)
        solved = audit.solve_shift(lambda x: x, b, kry, 1., 3)
        self.assertTrue(solved["rhs_zero"])
        self.assertFalse(audit.qualify_pair(solved, solved, 0., 1e-3, .01)["linear_solve_qualified"])

    def test_projection_summary_uses_paired_cells_and_separate_levels(self):
        rows = []
        for pid in (1, 2, 3, 4):
            for seed in (42, 43):
                for gamma in (1., 2.):
                    rows.append(dict(patient_id=pid, seed=seed, gamma=gamma, condition="full", oct_class=1,
                                     qualified=not (pid == 4 and seed == 42 and gamma == 2.),
                                     pred_h_projection=pid / gamma, true_h_projection=float(pid)))
        summary = diag.projection_summary(rows, [1., 2.])
        main = [r for r in summary if r['condition'] == 'full' and r['oct_class'] is None]
        self.assertEqual({r['n_common_cells'] for r in main}, {7})
        self.assertEqual({r['n'] for r in main if r['level'] == 'patient_seed'}, {7})
        self.assertEqual({r['n'] for r in main if r['level'] == 'patient_mean'}, {3})
        self.assertTrue(all(r['spearman'] == 1. for r in main))

    def test_real_checkpoint_h_replay_and_all_three_entrypoints(self):
        """Only the OCT image loader is mocked. All derivatives and attacks run."""
        X = np.random.default_rng(52).random((16, 1, 32, 32), dtype=np.float32)
        y = np.tile(np.arange(4), 4).astype(np.int64)
        groups = np.arange(100, 116)
        split = {"target_train_idx": [8, 9, 10, 11], "target_test_idx": [12, 13, 14, 15],
                 "shadow_models": [{"train_idx": [0, 1, 2, 3], "test_idx": [4, 5, 6, 7]}]}
        patient = {"patient_id": 101, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]}
        with tempfile.TemporaryDirectory() as td:
            full, dose = Path(td) / "full", Path(td) / "dose"
            with mock.patch.object(sys, "argv", ["pilot"]):
                p = pilot.parse_args()
            p.n_shadow, p.affected_shadow, p.seeds, p.classes, p.n_patients = 1, 0, [42], [1], 1
            p.shadow_epochs, p.attack_epochs = 2, 2
            p.shadow_batch_size, p.attack_batch_size = 4, 4
            p.save_epoch_checkpoints, p.deletion_mode = [1, 2], "fixed_mask"
            p.attack_seeds = [5101, 5102]
            p.output_dir, p.n_total_samples = str(full), 16
            for root, alpha in ((full, 1.), (dose, .1)):
                legacy.write_json(root / "splits/fresh_patient_split.json", split)
                split_hash = legacy.sha(root / "splits/fresh_patient_split.json")[:16]
                legacy.write_json(root / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
                legacy.write_json(root / "experiment_config.json", {
                    "training_numerics": pilot.TRAINING_NUMERICS,
                    "args": {**vars(p), "deletion_weight": alpha},
                    "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
                legacy.write_json(root / "selected_patients.json", {"split_sha256": split_hash, "patients": [patient]})
                legacy.write_json(root / "baseline_seed42.json", {})
            orders = pilot.make_epoch_orders(np.arange(4), 2, 700042)
            baseline, metrics = pilot.train_classifier_from_orders(
                X, y, orders, [4, 5, 6, 7], 42, 128, p.shadow_lr, 4, 1e-5, True,
                deletion_mode="fixed_mask", epoch_checkpoint_dir=full / "checkpoints/baseline_seed42_epochs",
                epoch_checkpoints=[1, 2])
            for root in (full, dose):
                pilot.save_model(root / "checkpoints/shadow_0_baseline_seed42.pt", baseline, {"seed": 42, "metrics": metrics})
                np.savez_compressed(root / "stage1_order_seed42.npz", raw_index_order=orders)
            target_orders = pilot.make_epoch_orders(np.arange(8, 12), 2, 700000 + p.target_seed)
            target, tm = pilot.train_classifier_from_orders(X, y, target_orders, [12, 13, 14, 15], p.target_seed,
                             128, p.shadow_lr, 4, 1e-5, True, deletion_mode="fixed_mask")
            pilot.save_model(full / "checkpoints/target_fixed.pt", target, {"seed": p.target_seed, "metrics": tm})
            interface = pilot.make_interface([baseline], split, X, y)
            queries = pilot.make_target_queries(target, split, X, y)
            tr, te = interface["classes"] == 1, queries["classes"] == 1
            ce = []
            for attack_seed in p.attack_seeds:
                attack = pilot.train_attack_model_deterministic(interface['x'][tr], interface['membership'][tr],
                         attack_seed, p.attack_epochs, p.attack_lr, p.attack_batch_size, n_hidden=64, deterministic=True)
                metric, _, _ = pilot.evaluate_attack_queries(attack, queries['x'][te], queries['membership'][te])
                ce.append(metric['cross_entropy'])
            for root, alpha in ((full, 1.), (dose, .1)):
                model, mm = pilot.train_classifier_from_orders(X, y, orders, [4, 5, 6, 7], 42, 128, p.shadow_lr, 4,
                                1e-5, True, excluded_indices=[1], deletion_mode="fixed_mask", deletion_weight=alpha)
                pilot.save_model(root / "checkpoints/shadow_0_seed42_patient101.pt", model,
                                 {"seed": 42, "excluded_indices": [1], "deletion_weight": alpha, "metrics": mm})
                legacy.write_json(root / "runs/seed42_patient101.json", {
                    "seed": 42, "patient": patient, "affected_shadow": 0, "split_sha256": split_hash,
                    "exposure": {"deletion_weight": alpha}, "conditions": {"J00": {"per_class": {"1": {
                        "attack_seeds": p.attack_seeds, "per_rep_cross_entropy": ce, "cross_entropy": float(np.mean(ce))}}}}})
            for mode in ("preflight", "convergence", "damping"):
                output = Path(td) / (mode + "_output")
                argv = ["diagnostic", "--mode", mode, "--full_dir", str(full), "--dose_dir", str(dose),
                        "--out_dir", str(output), "--seeds", "42", "--krylov_steps", "8", "16",
                        "--damping_grid", ".3", "10", "--lanczos_iters", "3", "--probe_seeds", "17",
                        "--hvp_batch", "4", "--dropout_mc_reps", "2"]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(legacy, "module_from_path", return_value=score), \
                     mock.patch.object(score, "import_pilot_module", return_value=pilot), \
                     mock.patch.object(pilot, "load_dataset", return_value=(X, y, groups)) as loader:
                    diag.main()
                    if mode == 'preflight':
                        loader.assert_not_called()
                manifest = legacy.read_json(output / "manifest.json")
                self.assertEqual(manifest['status'], 'complete')
                self.assertTrue(manifest['input_files_unchanged'])
                self.assertEqual(manifest['n_rows'], {'preflight': 1, 'convergence': 3, 'damping': 4}[mode])
                if mode == 'damping':
                    payload = torch.load(output / 'h_seed42.pt', weights_only=False, map_location='cpu')
                    self.assertEqual(payload['h_by_class'][1].shape, (2, 503044))
                    checks = legacy.read_json(output / 'h_checks_seed42.json')['classes']['1']['checks']
                    self.assertTrue(all(c['j00_max_abs_diff'] <= 1e-6 for c in checks))
                    self.assertTrue((output / 'projection_summary.csv').exists())
                    detail = legacy.read_json(output / 'damping_seed42.json')
                    self.assertTrue(any(r['linear_solve_qualified'] for r in detail['rows']))
                    # A two-epoch toy attack can fail the unchanged 07 float32
                    # eigenvalue screen. That must not be silently promoted to
                    # a qualified h projection, even when Stage 1 converges.
                    for row in detail['rows']:
                        if not row['attack_solve_qualified']:
                            self.assertFalse(row['qualified'])
                        if not row['qualified']:
                            self.assertIsNone(row['pred_h_projection'])
            # A changed attack seed panel must be rejected when h is reconstructed.
            context = diag.metadata_context(type('A', (), dict(full_dir=full, dose_dir=dose, seeds=[42]))(), pilot)
            context.update(X=X, y=y)
            corrupt = full / 'runs/seed42_patient101.json'
            r = legacy.read_json(corrupt)
            r['conditions']['J00']['per_class']['1']['attack_seeds'] = [1, 2]
            legacy.write_json(corrupt, r)
            args = type('A', (), dict(damping_attack=.2, j00_tol=1e-6))()
            with self.assertRaisesRegex(RuntimeError, 'attack seeds differ'):
                diag.h_for_seed(context, args, pilot, score, baseline, 42, Path(td))


if __name__ == '__main__':
    print(f"Stage 1 v1.1 tests: torch={torch.__version__}, device={DEVICE}", flush=True)
    unittest.main(verbosity=2)
