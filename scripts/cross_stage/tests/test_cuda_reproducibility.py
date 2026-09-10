"""Pooling derivatives and exact trajectory checks on CPU and available CUDA.

CUDA is not silently replaced by CPU: the production-image replay uses the
same device as 05, and the log explicitly records whether CUDA was exercised.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_repro", HERE.parent / "05_end_to_end_patient_loo_pilot.py")
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
from models import DeterministicAdaptiveAvgPool2d, SmallCNN


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        pilot.seed_everything(7, deterministic=True)

    def test_pool_values_gradients_hvp_and_jvp(self):
        # The reference always differentiates on CPU: the CUDA reference
        # kernel is precisely the unsupported kernel that this fix removes.
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        gen = torch.Generator().manual_seed(10)
        for height, width in [(16, 16), (5, 7), (2, 2)]:
            raw = torch.randn(2, 3, height, width, generator=gen, dtype=torch.float64)
            direction = torch.randn(raw.shape, generator=gen, dtype=torch.float64)
            ref_x = raw.clone().requires_grad_(True)
            ref_pool = lambda x: F.adaptive_avg_pool2d(x, (4, 4))
            ref_y = ref_pool(ref_x)
            ref_g = torch.autograd.grad(ref_y.square().sum(), ref_x, create_graph=True)[0]
            ref_hv = torch.autograd.grad((ref_g * direction).sum(), ref_x)[0]
            _, ref_jv = torch.func.jvp(ref_pool, (raw,), (direction,))
            for device in devices:
                with self.subTest(shape=(height, width), device=str(device)):
                    pool = DeterministicAdaptiveAvgPool2d((4, 4)).to(device)
                    x = raw.to(device).clone().requires_grad_(True)
                    d = direction.to(device)
                    y = pool(x)
                    g = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
                    hv = torch.autograd.grad((g*d).sum(), x)[0]
                    _, jv = torch.func.jvp(pool, (x,), (d,))
                    for actual, expected in [(y, ref_y), (g, ref_g), (hv, ref_hv), (jv, ref_jv)]:
                        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=1e-11, atol=1e-12)

    def test_production_size_exact_replay(self):
        # OCT loader uses 128x128; feature maps are 16x16 before 4x4 pooling.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 1, 128, 128)).astype(np.float32)
        y = np.arange(8) % 4
        orders = pilot.make_epoch_orders(np.arange(8), 1, 123)

        def train(mode):
            model, metrics = pilot.train_classifier_from_orders(
                x, y, orders, eval_indices=np.arange(8), seed=7, n_hidden=8,
                lr=1e-3, batch_size=4, weight_decay=0., deterministic=True,
                deletion_mode=mode)
            return torch.cat([p.detach().flatten() for p in model.parameters()]), metrics["post_train_rng_sha256"]

        base, base_rng = train("fixed_mask")
        for mode in ["fixed_mask", "filter_rechunk"]:
            actual, actual_rng = train(mode)
            self.assertTrue(torch.equal(actual, base),
                            f"128x128 {mode} replay on {pilot.DEVICE}: max_abs={float((actual-base).abs().max()):.3e}")
            self.assertEqual(actual_rng, base_rng)
        print(f"128x128 exact training replay passed on {pilot.DEVICE}", flush=True)

    def test_strict_mode_and_legacy_checkpoint_rejection(self):
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        model = SmallCNN(1, 8, 4)
        old_keys = set(model.state_dict())
        model.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.assertEqual(old_keys, set(model.state_dict()))  # no weight-format migration
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"state_dict": model.state_dict(), "metadata": {}}, path)
            with self.assertRaisesRegex(RuntimeError, "Incompatible training numerics"):
                pilot.load_model(path, 1, 8, 4)
            model.pool = DeterministicAdaptiveAvgPool2d((4, 4))
            pilot.save_model(path, model, {"seed": 7})
            loaded = pilot.load_model(path, 1, 8, 4)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[key].cpu()))
        with self.assertRaisesRegex(RuntimeError, "Incompatible training numerics"):
            pilot.require_training_numerics({"schema_version": 4}, "old experiment_config.json")


if __name__ == "__main__":
    print(f"Reproducibility tests: torch={torch.__version__}, CUDA={torch.version.cuda}, "
          f"cuda_available={torch.cuda.is_available()}, training_device={pilot.DEVICE}", flush=True)
    if not torch.cuda.is_available():
        print("CUDA NOT TESTED in this process; rerun on a Delta GPU before OCT training.", flush=True)
    unittest.main(verbosity=2)
