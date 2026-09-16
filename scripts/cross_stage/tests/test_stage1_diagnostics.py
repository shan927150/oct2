#!/usr/bin/env python3
"""CPU or CUDA tests of the independent diagnostic, with no OCT data needed."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


diag = load_module("stage1_diag_under_test", HERE / "11_stage1_diagnostics.py")
score = load_module("score07_for_stage1_tests", HERE / "07_cross_stage_score_ladder.py")
pilot = score.import_pilot_module()
DEVICE = pilot.DEVICE
torch.set_num_threads(min(2, torch.get_num_threads()))


class Stage1DiagnosticTests(unittest.TestCase):
    def test_cuda_is_not_silently_replaced(self):
        if os.environ.get("STAGE1_DIAG_REQUIRE_CUDA") == "1":
            self.assertTrue(torch.cuda.is_available(), "Delta test job requires CUDA")
            self.assertEqual(str(DEVICE).split(":")[0], "cuda")

    def test_cg_matches_dense_solve_and_alpha_scaling(self):
        M = torch.tensor([[3., .4, .1], [.4, 2., -.2], [.1, -.2, 1.]],
                         dtype=torch.float64, device=DEVICE)
        b = torch.tensor([.1, -.4, 1.], dtype=M.dtype, device=DEVICE)
        solved = score.conjugate_gradient(lambda x: M @ x, b, 10, 1e-12)
        expected = torch.linalg.solve(M, b)
        torch.testing.assert_close(solved["x"], expected, atol=1e-12, rtol=1e-12)
        self.assertLess(solved["final_rel_residual"], 1e-12)
        small = score.conjugate_gradient(lambda x: M @ x, .1 * b, 10, 1e-12)
        torch.testing.assert_close(small["x"], .1 * expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(core.solver_qualified(solved, {"passed": True}, 1e-8))
        bad = {**solved, "final_rel_residual": .1}
        self.assertFalse(core.solver_qualified(bad, {"passed": True}, 1e-3))
        self.assertFalse(core.solver_qualified(solved, {"passed": False}, 1e-3))

    def test_lanczos_detects_negative_edge_and_damping_screen(self):
        diagonal = torch.tensor([-.6, -.2, .03, .5, 2., 9.], dtype=torch.float64, device=DEVICE)
        probes = [core.lanczos_probe(lambda x: diagonal * x, len(diagonal), len(diagonal),
                                    seed, DEVICE, diagonal.dtype) for seed in (10, 11)]
        for p in probes:
            self.assertAlmostEqual(p["endpoints"]["min"]["value"], -.6, places=10)
            self.assertAlmostEqual(p["endpoints"]["max"]["value"], 9., places=10)
            self.assertLess(p["endpoints"]["min"]["residual_abs"], 1e-9)
        self.assertFalse(core.spectrum_screen(probes, .2, 1e-6)["passed"])
        self.assertTrue(core.spectrum_screen(probes, .7, 1e-6)["passed"])
        self.assertTrue(core.spectrum_screen(probes, .7, 1e-6)["not_spd_certificate"])
        unresolved = copy.deepcopy(probes)
        unresolved[0]["endpoints"]["max"]["residual_scaled"] = .5
        self.assertFalse(core.spectrum_screen(unresolved, 1., .005)["passed"])
        bad = score.conjugate_gradient(lambda x: diagonal * x,
                                      torch.tensor([1., 0., 0., 0., 0., 0.], device=DEVICE,
                                                   dtype=diagonal.dtype), 10, 1e-8)
        self.assertTrue(bad["nonpositive_curvature"])
        self.assertFalse(core.solver_qualified(bad, {"passed": True}, 1e-3))

    def test_hvp_matches_explicit_regularized_hessian(self):
        torch.manual_seed(9)
        model = torch.nn.Linear(2, 3).to(device=DEVICE, dtype=torch.float64)
        X = np.asarray([[.1, .7], [-.2, .4], [.5, .2], [1., -.3], [.8, .6]])
        y = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
        theta = core.flat(model).clone().requires_grad_(True)
        wd, gamma = .07, .2

        def loss(v):
            logits = F.linear(torch.as_tensor(X, device=DEVICE, dtype=v.dtype),
                              v[:6].reshape(3, 2), v[6:])
            return F.cross_entropy(logits, torch.as_tensor(y, device=DEVICE)) + wd / 2 * (v @ v)

        dense = torch.autograd.functional.hessian(loss, theta)
        vector = torch.linspace(-.3, .7, theta.numel(), device=DEVICE, dtype=theta.dtype)
        hvp = score.make_hvp(model, X, y, np.arange(len(y)), DEVICE, wd, gamma, batch=2)
        torch.testing.assert_close(hvp(vector), dense @ vector + gamma * vector, atol=1e-12, rtol=1e-10)
        observed, grad = core.objective_snapshot(model, X, y, np.arange(len(y)), 2, wd, True)
        expected_grad = torch.autograd.grad(loss(theta), theta)[0]
        torch.testing.assert_close(grad, expected_grad, atol=1e-12, rtol=1e-10)
        self.assertAlmostEqual(observed["objective"], float(loss(theta).detach()), places=12)

    def test_observer_preserves_rng_modes_parameters_and_grads(self):
        pilot.seed_everything(61, True)
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                                    torch.nn.Dropout(.2), torch.nn.Linear(4, 3)).to(DEVICE)
        model.train()
        model[0].eval()
        X = np.ones((5, 2), dtype=np.float32)
        y = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .3
        before = copy.deepcopy(model.state_dict())
        gradients = [p.grad.clone() for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        rng = core.rng_fingerprint()
        obs = core.observe_checkpoint(model, X, y, np.arange(5), 2, 1e-5, True, mc_reps=3)
        self.assertTrue(core.exact_tree(before, model.state_dict()))
        self.assertEqual(rng, core.rng_fingerprint())
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(g, p.grad) for g, p in zip(gradients, model.parameters())))
        self.assertTrue(np.isfinite(obs["dropout_mc_mean_gradient_norm"]))

    def test_baseline_replay_matches_original_parameters_adam_and_rng(self):
        rng = np.random.default_rng(8)
        X = rng.random((8, 1, 32, 32), dtype=np.float32)
        y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        train = np.arange(6)
        seed = 42
        orders = pilot.make_epoch_orders(train, 2, seed + 700000)
        seen = []
        with tempfile.TemporaryDirectory() as td:
            epochs = Path(td) / "epochs"
            original, metrics = pilot.train_classifier_from_orders(
                X, y, orders, [6, 7], seed, 128, .001, 4, 1e-5, True,
                epoch_checkpoint_dir=epochs, epoch_checkpoints=[1, 2], deletion_mode="fixed_mask")
            original_state = copy.deepcopy(original.state_dict())
            del original

            def check(epoch, model, optimizer):
                payload = torch.load(epochs / f"epoch{epoch:03d}.pt", weights_only=False, map_location="cpu")
                md = payload["metadata"]
                self.assertTrue(core.exact_tree(model.state_dict(), payload["state_dict"]))
                self.assertTrue(core.exact_tree(optimizer.state_dict(), md["optimizer_state"]))
                current = core.snapshot_rng()
                self.assertTrue(core.exact_tree(current["torch_cpu"], md["rng_states"]["torch_cpu"]))
                self.assertTrue(core.exact_tree(current["torch_cuda"], md["rng_states"].get("torch_cuda", [])))
                seen.append(epoch)

            def observe(model, row):
                info = core.observe_checkpoint(model, X, y, train, 3, 1e-5, True, mc_reps=2)
                self.assertTrue(np.isfinite(info["eval_objective"]))

            replay, _, fingerprint = core.replay_baseline(pilot, X, y, orders, seed, .001, 4, 1e-5,
                                                         observe, check)
            self.assertTrue(core.exact_tree(original_state, replay.state_dict()))
            self.assertEqual(fingerprint, metrics["post_train_rng_sha256"])
            self.assertEqual(seen, [1, 2])

    def test_zero_effect_geometry_is_undefined_not_false_success(self):
        zero = torch.zeros(3, device=DEVICE)
        g = core.geometry(zero, zero)
        self.assertIsNone(g["cosine"])
        self.assertIsNone(g["norm_ratio"])
        self.assertIsNone(g["relative_error"])

    def test_truth_pairing_rejects_changed_dose_and_panel(self):
        with tempfile.TemporaryDirectory() as td:
            roots = [Path(td) / x for x in ("full", "dose")]
            for root in roots:
                (root / "splits").mkdir(parents=True)
                (root / "splits/fresh_patient_split.json").write_text('{}\n')
            split_hash = diag.sha(roots[0] / "splits/fresh_patient_split.json")[:16]
            panel = {"split_sha256": split_hash, "patients": [
                {"patient_id": 7, "oct_class": 1, "raw_indices": [0, 2]}]}
            common = dict(deletion_mode="fixed_mask", removal_epochs=None, deterministic=True,
                          seeds=[42], window_membership="value_only")
            for root, alpha in zip(roots, (1., .1)):
                diag.write_json(root / "experiment_config.json", {
                    "training_numerics": pilot.TRAINING_NUMERICS, "args": {**common, "deletion_weight": alpha},
                    "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
                diag.write_json(root / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
                diag.write_json(root / "selected_patients.json", panel)
            diag.validate_sources(*roots, pilot, [42])
            dc = diag.read_json(roots[1] / "experiment_config.json")
            dc["args"]["deletion_weight"] = .25
            diag.write_json(roots[1] / "experiment_config.json", dc)
            with self.assertRaisesRegex(RuntimeError, "dose01"):
                diag.validate_sources(*roots, pilot, [42])
            dc["args"]["deletion_weight"] = .1
            diag.write_json(roots[1] / "experiment_config.json", dc)
            panel["patients"][0]["raw_indices"] = [1, 2]
            diag.write_json(roots[1] / "selected_patients.json", panel)
            with self.assertRaisesRegex(RuntimeError, "panel differs"):
                diag.validate_sources(*roots, pilot, [42])

    def test_both_cli_pipelines_with_real_synthetic_checkpoints(self):
        """Exercise artifact names, truth sign/scaling, plots, manifests and replay.

        Only OCT loading is replaced. Training, checkpoint reads, derivatives,
        CG and output writing use the real project code on a tiny dataset.
        """
        X = np.random.default_rng(82).random((8, 1, 32, 32), dtype=np.float32)
        y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        train = np.arange(6)
        orders = pilot.make_epoch_orders(train, 2, 42 + 700000)
        patient = {"patient_id": 7, "oct_class": 1, "class_name": "DME", "n_images": 1,
                   "raw_indices": [1]}
        p = Namespace(affected_shadow=0, shadow_epochs=2, save_epoch_checkpoints=[1, 2],
                      shadow_lr=.001, shadow_batch_size=4)
        with tempfile.TemporaryDirectory() as td:
            full, dose = Path(td) / "full", Path(td) / "dose"
            for root in (full, dose):
                root.mkdir()
                for name in ("experiment_config.json", "experiment_summary.json", "selected_patients.json",
                             "baseline_seed42.json", "splits/fresh_patient_split.json"):
                    diag.write_json(root / name, {})
                np.savez_compressed(root / "stage1_order_seed42.npz", raw_index_order=orders)
            baseline, base_metrics = pilot.train_classifier_from_orders(
                X, y, orders, [6, 7], 42, 128, .001, 4, 1e-5, True, deletion_mode="fixed_mask",
                epoch_checkpoint_dir=full / "checkpoints/baseline_seed42_epochs", epoch_checkpoints=[1, 2])
            for root in (full, dose):
                pilot.save_model(root / "checkpoints/shadow_0_baseline_seed42.pt", baseline,
                                 {"seed": 42, "metrics": base_metrics})
            del baseline
            for root, alpha in ((full, 1.), (dose, .1)):
                removed, metrics = pilot.train_classifier_from_orders(
                    X, y, orders, [6, 7], 42, 128, .001, 4, 1e-5, True,
                    excluded_indices=[1], deletion_mode="fixed_mask", deletion_weight=alpha)
                pilot.save_model(root / "checkpoints/shadow_0_seed42_patient7.pt", removed,
                                 {"seed": 42, "metrics": metrics, "excluded_indices": [1], "deletion_weight": alpha})
                diag.write_json(root / "runs/seed42_patient7.json", {
                    "seed": 42, "patient": patient, "split_sha256": "synthetic", "affected_shadow": 0,
                    "exposure": {"deletion_weight": alpha}, "conditions": {"J00": {"per_class": {"1": {
                        "attack_seeds": [5101], "per_rep_cross_entropy": [.2], "cross_entropy": .2}}}}})
            context = {"full": full, "dose": dose, "pargs": p, "summary": {"split_sha256": "synthetic"},
                       "patients": [patient], "X": X, "y": y, "train": train}
            for mode in ("damping", "convergence"):
                output = Path(td) / (mode + "_output")
                argv = ["11_stage1_diagnostics.py", "--mode", mode, "--full_dir", str(full),
                        "--dose_dir", str(dose), "--out_dir", str(output), "--seeds", "42",
                        "--damping_grid", "5", "20", "--lanczos_iters", "3", "--probe_seeds", "17",
                        "--cg_iters", "5", "--hvp_batch", "3", "--dropout_mc_reps", "2"]
                current = {**context, "dose": dose if mode == "damping" else None}
                with mock.patch.object(diag, "load_context", return_value=current), mock.patch.object(sys, "argv", argv):
                    diag.main()
                manifest = diag.read_json(output / "manifest.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertTrue(manifest["input_files_unchanged"])
                self.assertEqual(manifest["n_rows"], 4 if mode == "damping" else 3)
                self.assertTrue((output / (mode + "_overview.png")).is_file())
                if mode == "damping":
                    detail = diag.read_json(output / "damping_seed42.json")
                    for gamma in (5., 20.):
                        pair = [r for r in detail["raw_rows"] if r["gamma"] == gamma]
                        norms = [r["raw_metrics_do_not_use_if_unqualified"]["pred_norm"] for r in pair]
                        self.assertAlmostEqual(norms[1] / norms[0], .1, places=6)
                        if not pair[0]["qualified"]:
                            self.assertTrue(all(r["cosine"] is None for r in detail["rows"] if r["gamma"] == gamma))
                else:
                    self.assertTrue(diag.read_json(output / "replay_final_seed42.json")["passed"])


if __name__ == "__main__":
    print(f"Stage 1 diagnostics tests: torch={torch.__version__}, device={DEVICE}", flush=True)
    unittest.main(verbosity=2)
