#!/usr/bin/env python3
"""E0 regression: algebra closure, signs and independent forward-AD Hessian check.

Only the OCT image loader is mocked; the SmallCNN, 05 training, 07 gradients/HVP
and the E0 entry point all run on tiny real tensors.
"""
import importlib.util
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
spec = importlib.util.spec_from_file_location("e0_test_module", HERE / "13_e0_residual_decomposition.py")
e0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e0
spec.loader.exec_module(e0)
legacy, score, pilot = e0.load_modules()
DEVICE = pilot.DEVICE
torch.set_num_threads(min(2, torch.get_num_threads()))


def build_fixture(td: Path):
    """Two tiny fixed_mask truth directories (alpha=1 and 0.1) with real checkpoints."""
    X = np.random.default_rng(52).random((16, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), 4).astype(np.int64)
    groups = np.arange(100, 116)
    split = {"target_train_idx": [8, 9, 10, 11], "target_test_idx": [12, 13, 14, 15],
             "shadow_models": [{"train_idx": [0, 1, 2, 3], "test_idx": [4, 5, 6, 7]}]}
    patients = [{"patient_id": 101, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]},
                {"patient_id": 102, "oct_class": 2, "class_name": "DRUSEN", "n_images": 1, "raw_indices": [2]}]
    full, dose = td / "full", td / "dose"
    with mock.patch.object(sys, "argv", ["pilot"]):
        p = pilot.parse_args()
    p.n_shadow, p.affected_shadow, p.seeds, p.classes, p.n_patients = 1, 0, [42], [1, 2], 2
    p.shadow_epochs, p.attack_epochs, p.shadow_batch_size, p.attack_batch_size = 2, 2, 4, 4
    p.save_epoch_checkpoints = [1, 2]
    p.deletion_mode, p.attack_seeds, p.output_dir, p.n_total_samples = "fixed_mask", [5101], str(full), 16
    for root, alpha in ((full, 1.), (dose, .1)):
        legacy.write_json(root / "splits/fresh_patient_split.json", split)
        split_hash = legacy.sha(root / "splits/fresh_patient_split.json")[:16]
        legacy.write_json(root / "experiment_summary.json", {"status": "complete", "split_sha256": split_hash})
        legacy.write_json(root / "experiment_config.json", {
            "training_numerics": pilot.TRAINING_NUMERICS, "args": {**vars(p), "deletion_weight": alpha},
            "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
        legacy.write_json(root / "selected_patients.json", {"split_sha256": split_hash, "patients": patients})
        legacy.write_json(root / "baseline_seed42.json", {})
    orders = pilot.make_epoch_orders(np.arange(4), 2, 700042)
    baseline, metrics = pilot.train_classifier_from_orders(
        X, y, orders, [4, 5, 6, 7], 42, 128, p.shadow_lr, 4, 1e-5, True, deletion_mode="fixed_mask",
        epoch_checkpoint_dir=full / "checkpoints/baseline_seed42_epochs", epoch_checkpoints=[1, 2])
    for root in (full, dose):
        pilot.save_model(root / "checkpoints/shadow_0_baseline_seed42.pt", baseline, {"seed": 42, "metrics": metrics})
        np.savez_compressed(root / "stage1_order_seed42.npz", raw_index_order=orders)
    for patient in patients:
        pid, idx = patient["patient_id"], patient["raw_indices"]
        for root, alpha in ((full, 1.), (dose, .1)):
            model, mm = pilot.train_classifier_from_orders(
                X, y, orders, [4, 5, 6, 7], 42, 128, p.shadow_lr, 4, 1e-5, True,
                excluded_indices=idx, deletion_mode="fixed_mask", deletion_weight=alpha)
            pilot.save_model(root / f"checkpoints/shadow_0_seed42_patient{pid}.pt", model,
                             {"seed": 42, "excluded_indices": idx, "deletion_weight": alpha, "metrics": mm})
            legacy.write_json(root / f"runs/seed42_patient{pid}.json", {
                "seed": 42, "patient": patient, "affected_shadow": 0, "split_sha256": split_hash,
                "exposure": {"deletion_weight": alpha},
                "conditions": {"J00": {"per_class": {str(patient["oct_class"]): {
                    "attack_seeds": p.attack_seeds, "per_rep_cross_entropy": [0.5], "cross_entropy": 0.5}}}}})
            raw = np.arange(8)
            membership = np.asarray([1]*4 + [0]*4)
            np.savez_compressed(root / f"runs/seed42_patient{pid}_interface.npz",
                                raw_index=raw, oct_class=y[raw], baseline_membership=membership,
                                loo_membership=membership, is_deleted_patient=(groups[raw] == pid),
                                p_baseline=pilot.get_predictions(baseline, X[raw]),
                                p_loo=pilot.get_predictions(model, X[raw]))
    return X, y, groups, full, dose


class E0Tests(unittest.TestCase):
    def test_entrypoint_closure_shares_and_finite_difference_hessian(self):
        with tempfile.TemporaryDirectory() as td:
            X, y, groups, full, dose = build_fixture(Path(td))
            output = Path(td) / "e0_output"
            argv = ["e0", "--full_dir", str(full), "--dose_dir", str(dose), "--out_dir", str(output),
                    "--seeds", "42", "--core_patients", "101", "--hvp_batch", "4", "--dropout_mc_reps", "3",
                    "--gammas", "0", "0.3"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(e0, "load_modules", return_value=(legacy, score, pilot)), \
                 mock.patch.object(pilot, "load_dataset", return_value=(X, y, groups)):
                e0.main()
            manifest = legacy.read_json(output / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["input_files_unchanged"])
            payload = legacy.read_json(output / "e0_rows.json")
            rows = payload["rows"]
            self.assertEqual(len(rows), 2)                      # one core patient x two alphas
            self.assertEqual(len(payload["monitor"]), 2)         # whole panel monitored by default
            for row in rows:
                # the identity is exact; only float rounding remains
                self.assertLess(row["algebra_closure_rel"], 1e-4)
                self.assertAlmostEqual(row["share_stationarity"] + row["share_remainder"], 1.0, places=6)
                self.assertIn(row["condition"], ("full", "dose01"))
                self.assertGreater(row["d_norm"], 0)
                self.assertIsNotNone(row["residual_gamma0.3_over_alpha_b0"])
            self.assertTrue((output / "e0_rows.csv").exists() and (output / "signal_monitor.csv").exists())
            monitor = {m["patient_id"]: m for m in payload["monitor"]}
            self.assertEqual(monitor[101]["n_images"], 1)
            self.assertAlmostEqual(monitor[101]["cancellation_ratio"], 1.0, places=6)  # single image: no cancellation
            self.assertIsNotNone(monitor[101]["b_dropout_mc_norm"])

            # ---- independent checks of the conventions used inside run_seed
            train = np.array([0, 1, 2, 3])
            base = legacy.load_payload(full / "checkpoints/shadow_0_baseline_seed42.pt", pilot)
            model0 = legacy.model_from_payload(base, pilot, X, y)
            theta0 = core.flat(model0).clone()
            hvp = score.make_hvp(model0, X, y, train, DEVICE, legacy.WD, 0., 4)
            loo = legacy.load_payload(full / "checkpoints/shadow_0_seed42_patient101.pt", pilot)
            model_a = legacy.model_from_payload(loo, pilot, X, y)
            d = (core.flat(model_a) - theta0).to(DEVICE)
            Hd = e0.d64(hvp(d))
            # forward-over-reverse reference: jvp of the exact eval gradient along d, float64.
            # Finite differences need a step-size ladder and branch/roundoff checks:
            # one failed step is not enough to invalidate that validation method.
            from torch.func import functional_call, grad, jvp
            m64 = legacy.model_from_payload(base, pilot, X, y).double().eval()
            names = [n for n, _ in m64.named_parameters()]
            shapes = [q.shape for _, q in m64.named_parameters()]
            xb = torch.as_tensor(X[train], dtype=torch.float64, device=DEVICE)
            yb = torch.as_tensor(y[train], dtype=torch.long, device=DEVICE)

            def objective(theta_flat):
                pieces, k = {}, 0
                for name, shape in zip(names, shapes):
                    n = int(np.prod(shape)); pieces[name] = theta_flat[k:k + n].view(shape); k += n
                out = functional_call(m64, pieces, (xb,))
                ce = torch.nn.functional.cross_entropy(out, yb, reduction="sum") / len(train)
                return ce + legacy.WD / 2 * torch.dot(theta_flat, theta_flat)
            theta64, d64 = theta0.double().to(DEVICE), d.double().to(DEVICE)
            g_ref, Hd_ref = jvp(grad(objective), (theta64,), (d64,))
            self.assertLess(core.norm(e0.d64(Hd_ref) - Hd) / core.norm(Hd), 1e-4)
            g0 = e0.d64(score.shadow_loss_grad(model0, X, y, train, DEVICE, legacy.WD, 4))
            self.assertLess(core.norm(e0.d64(g_ref) - g0) / core.norm(g0), 1e-4)
            # g_alpha is the gradient of the fixed_mask weighted objective at theta_alpha
            alpha, idx = 1.0, [1]
            gL0 = e0.d64(score.shadow_loss_grad(model_a, X, y, train, DEVICE, legacy.WD, 4))
            b_a = e0.d64(e0.patient_gradients(score, model_a, X, y, idx, DEVICE).sum(0)) / len(train)
            params = [p for p in model_a.parameters() if p.requires_grad]
            w = torch.ones(len(train), device=DEVICE)
            w[[train.tolist().index(i) for i in idx]] = 1 - alpha
            xb = torch.as_tensor(X[train], device=DEVICE); yb = torch.as_tensor(y[train], device=DEVICE)
            model_a.eval()
            per = torch.nn.functional.cross_entropy(model_a(xb), yb, reduction="none")
            loss = (w * per).sum() / len(train) + legacy.WD / 2 * sum((p ** 2).sum() for p in params)
            direct = e0.d64(torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)]))
            self.assertLess(core.norm(direct - (gL0 - alpha * b_a)) / core.norm(direct), 1e-5)

    def test_output_must_not_overlap_truth_directories(self):
        with tempfile.TemporaryDirectory() as td:
            full = Path(td) / "full"; full.mkdir()
            argv = ["e0", "--full_dir", str(full), "--dose_dir", str(full), "--out_dir", str(full / "inside")]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(e0, "load_modules", return_value=(legacy, score, pilot)):
                with self.assertRaises(RuntimeError):
                    e0.main()


if __name__ == "__main__":
    unittest.main()
