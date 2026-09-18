#!/usr/bin/env python3
"""B1-smoke regression: mask recovery, functional Adam fidelity, zero AND non-zero directions."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b1 = _load("b1_test_module", HERE / "16_b1_smoke.py")
fixture = _load("b1_fixture", HERE / "tests/test_e0_residual_decomposition.py")
legacy, score, pilot = b1.e0.load_modules()
DEVICE = pilot.DEVICE
torch.set_num_threads(min(2, torch.get_num_threads()))
CFG = {"lr": 1e-3, "batch_size": 4, "wd": 1e-5, "deterministic": True}


def tiny_problem(n=16, epochs=3):
    X = np.random.default_rng(3).random((n, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), n // 4).astype(np.int64)
    orders = pilot.make_epoch_orders(np.arange(8), epochs, 700042)
    return X, y, orders


class B1Tests(unittest.TestCase):
    def test_recorded_mask_reproduces_dropout_and_is_constant_in_alpha(self):
        X, y, orders = tiny_problem()
        _, _, _, masks, n = b1.faithful_run(pilot, X, y, orders, 42, CFG, 1.0, (), 4,
                                            prefixes=[4], record_masks=True)
        self.assertEqual(len(masks), min(4, n))
        p = 0.2
        for m in masks:
            values = torch.unique(m)
            self.assertLessEqual(len(values), 2)
            for v in values.tolist():
                self.assertTrue(math.isclose(v, 0.0) or math.isclose(v, 1.0 / (1.0 - p), rel_tol=1e-6))
            self.assertFalse(m.requires_grad)

    def test_faithful_loop_matches_the_real_trainer_bitwise(self):
        """The loop written in 16 must be 05's loop, not merely a similar one."""
        X, y, orders = tiny_problem(epochs=2)
        per_epoch = math.ceil(len(orders[0]) / CFG["batch_size"])
        reference, _ = pilot.train_classifier_from_orders(
            X, y, orders, [8, 9, 10, 11], 42, 128, CFG["lr"], CFG["batch_size"], CFG["wd"], True,
            deletion_mode="fixed_mask")
        model, _, snap, _, _ = b1.faithful_run(pilot, X, y, orders, 42, CFG, 1.0, (),
                                               len(orders) * per_epoch,
                                               prefixes=[len(orders) * per_epoch])
        got = snap[len(orders) * per_epoch]["theta"].cpu()
        want = torch.cat([q.detach().cpu().reshape(-1) for q in reference.parameters()])
        self.assertTrue(torch.equal(got, want))

    def test_functional_adam_matches_the_faithful_loop(self):
        X, y, orders = tiny_problem(epochs=2)
        per_epoch = math.ceil(len(orders[0]) / CFG["batch_size"])
        steps = 2 * per_epoch
        prefixes = [1, per_epoch, steps]
        _, _, snap, masks, _ = b1.faithful_run(pilot, X, y, orders, 42, CFG, 1.0, (), steps,
                                               prefixes=prefixes, record_masks=True)
        pilot.seed_everything(42, True)
        model = pilot.build_model("cnn", X.shape[1], 128, 4)
        theta0 = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
        b1.swap_in_recorder(model)
        plan = b1.batch_plan(orders, CFG["batch_size"], steps)
        out = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0, plan, masks, (), 1.0,
                                       prefixes, tangent=False)
        for k in prefixes:
            for field in ("theta", "m", "v"):
                ref = snap[k][field].cpu().double()
                rel = core.norm(out[k][field].cpu().double() - ref) / core.norm(ref)
                self.assertLess(rel, 1e-6, f"{field} at prefix {k}: {rel:.3e}")
            self.assertEqual(out[k]["step"], snap[k]["step"])

    def test_absent_direction_gives_exactly_zero_tangent(self):
        """G3. Necessary, but on its own it proves only plumbing."""
        X, y, orders = tiny_problem(epochs=2)
        steps = 2 * math.ceil(len(orders[0]) / CFG["batch_size"])
        _, _, _, masks, _ = b1.faithful_run(pilot, X, y, orders, 42, CFG, 1.0, (), steps,
                                            prefixes=[steps], record_masks=True)
        pilot.seed_everything(42, True)
        model = pilot.build_model("cnn", X.shape[1], 128, 4)
        theta0 = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
        b1.swap_in_recorder(model)
        plan = b1.batch_plan(orders, CFG["batch_size"], steps)
        # rows 8..15 never appear in the recorded orders (which only use 0..7)
        out = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0, plan, masks,
                                       [12, 13], 0.0, [steps], tangent=True)
        self.assertEqual(int(torch.count_nonzero(out[steps]["u_theta"]).item()), 0)

    def _tangent_fixture(self, epochs=2):
        X, y, orders = tiny_problem(epochs=epochs)
        steps = epochs * math.ceil(len(orders[0]) / CFG["batch_size"])
        prefixes = [1, steps]
        _, _, base32, masks, _ = b1.faithful_run(pilot, X, y, orders, 42, CFG, 1.0, (), steps,
                                                 prefixes=prefixes, record_masks=True)
        pilot.seed_everything(42, True)
        model = pilot.build_model("cnn", X.shape[1], 128, 4)
        theta0 = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
        b1.swap_in_recorder(model)
        plan = b1.batch_plan(orders, CFG["batch_size"], steps)
        present = [int(orders[0][0])]
        tan = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0, plan, masks,
                                       present, 0.0, prefixes, tangent=True)
        tan64 = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0.double(), plan, masks,
                                         present, 0.0, prefixes, tangent=True)
        return (X, y, orders, steps, prefixes, base32, masks, model, theta0, plan,
                present, tan, tan64)

    def test_present_direction_tangent_matches_a_paired_finite_difference_in_float64(self):
        """G4 in miniature, in the precision that can answer the question.

        This is the non-zero direction a zero-tangent test cannot check. It is run in
        float64 -- tangent AND finite difference both -- because that is the only
        precision in which "the finite difference does not reproduce the tangent" means
        "the tangent is wrong". Comparing a float64 difference against a float32 tangent
        measures the tangent's own precision, which is the separate test below.
        """
        (X, y, orders, steps, prefixes, _base32, masks, model, theta0, plan,
         present, _tan, tan64) = self._tangent_fixture()
        theta0_64 = theta0.double()
        base64 = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0_64, plan, masks,
                                          present, 0.0, prefixes)
        for k in prefixes:
            u = tan64[k]["u_theta"].cpu().double()
            self.assertGreater(core.norm(u), 0.0)        # the direction must actually be excited
            errors, cosines = [], []
            for exponent in (12, 16, 20, 24):
                alpha = b1.exact_alpha(exponent)
                out = b1.functional_trajectory(model, X, y, CFG, DEVICE, theta0_64, plan, masks,
                                               present, alpha, [k])
                fd = ((out[k]["theta"] - base64[k]["theta"]) / alpha).cpu().double()
                errors.append(core.norm(fd - u) / core.norm(u))
                cosines.append(core.cosine(fd, u))
            self.assertLess(min(errors), 1e-2, f"prefix {k}: best float64 FD error {min(errors):.3e}")
            self.assertGreater(max(cosines), 1 - 1e-6, f"prefix {k}: best cosine {max(cosines):.6f}")
            # and it must be converging, not merely passing at one lucky alpha
            self.assertLess(errors[-1], errors[0], f"prefix {k}: float64 FD not converging: {errors}")

    def test_float32_cannot_resolve_the_step_one_tangent(self):
        """The companion finding, asserted rather than worked around.

        The real pipeline trains in float32. At the first Adam step the update is exactly
        -lr*g/(|g|+eps), so the alpha-sensitivity is eps/(|g|+eps)^2: it is carried by the
        coordinates whose gradient sits at Adam's epsilon. Both halves of a float32 paired
        finite difference therefore have a floor far above the float64 one. This test pins
        that floor down so a future change cannot quietly turn a measurement limit into a
        loosened tolerance -- if float32 ever does resolve it, this test fails and the
        gate should be tightened, not this assertion deleted.
        """
        (X, y, orders, steps, prefixes, base32, masks, model, theta0, plan,
         present, tan, tan64) = self._tangent_fixture()
        u = tan[1]["u_theta"].cpu().double()
        errors = []
        for exponent in (12, 16, 20, 24):
            alpha = b1.exact_alpha(exponent)
            _, _, snap, _, _ = b1.faithful_run(pilot, X, y, orders, 42, CFG, alpha, present,
                                               1, masks=masks, prefixes=[1])
            fd = ((snap[1]["theta"] - base32[1]["theta"]) / alpha).cpu().double()
            errors.append(core.norm(fd - u) / core.norm(u))
        self.assertGreater(min(errors), 1e-2,
                           f"float32 now resolves the tangent ({min(errors):.3e}); "
                           f"revisit the float64-only gate rather than relaxing this")

    def test_the_float32_tangent_itself_is_only_good_to_about_a_percent(self):
        """The other half of the same effect, and the one easiest to overlook.

        It is not only the finite difference that float32 degrades. eps/(|g|+eps)^2 is
        ~1e8 where |g| ~ eps, so float32 noise in g is amplified into the tangent too.
        The jvp computed at the production precision differs from the float64 jvp by
        roughly the same 1% that bounds the float32 finite difference -- which is why a
        float64 difference must be compared against a float64 tangent.
        """
        (X, y, orders, steps, prefixes, _b, masks, model, theta0, plan,
         present, tan, tan64) = self._tangent_fixture()
        u32 = tan[1]["u_theta"].cpu().double()
        u64 = tan64[1]["u_theta"].cpu().double()
        rel = core.norm(u32 - u64) / core.norm(u64)
        self.assertGreater(rel, 1e-4, f"float32 tangent unexpectedly clean: {rel:.3e}")
        self.assertLess(rel, 0.5, f"float32 tangent is not merely imprecise: {rel:.3e}")
        self.assertGreater(core.cosine(u32, u64), 0.9)   # same direction, wrong last digits

    def test_epsilon_structure_reports_where_the_tangent_lives(self):
        """G5. The tangent must be attributed, not just measured."""
        (X, y, orders, steps, prefixes, _b, masks, model, theta0, plan,
         present, tan, tan64) = self._tangent_fixture()
        g = b1.batch_gradient(model, X, y, CFG, DEVICE, theta0, plan, masks, 0, present, 0.0)
        report = b1.epsilon_structure(g, tan[1]["u_theta"])
        self.assertEqual(report["n_coordinates"], theta0.numel())
        for key in ("share_grad_below_10eps", "share_of_tangent_sq_in_top_0p1pct",
                    "largest_single_coordinate_share", "d_update_d_grad_max"):
            self.assertIn(key, report)
        # eps/(|g|+eps)^2 peaks at 1/eps as |g| -> 0 and is 1/(4 eps) at |g| = eps;
        # the report must carry that scale, not a clipped or rescaled version of it
        self.assertGreater(report["d_update_d_grad_max"], 1.0 / (4 * b1.ADAM_EPS))
        self.assertLessEqual(report["d_update_d_grad_max"], 1.0 / b1.ADAM_EPS)
        self.assertGreaterEqual(report["share_of_tangent_sq_in_top_1pct"],
                                report["share_of_tangent_sq_in_top_0p1pct"])
        self.assertTrue(0.0 <= report["largest_single_coordinate_share"] <= 1.0)

    def test_exact_alpha_floor(self):
        for k in range(1, 25):
            self.assertEqual(float(1.0 - np.float32(1.0 - b1.exact_alpha(k))), 2.0 ** -k)
        with self.assertRaises(RuntimeError):
            b1.exact_alpha(25)

    def test_end_to_end_run_gates_and_reports_both_precisions(self):
        """main() on a real (tiny) truth directory: every gate runs, both precisions land."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            X, y, groups, full, dose = fixture.build_fixture(td)
            before = {str(q): legacy.sha(q) for root in (full, dose)
                      for q in root.rglob("*") if q.is_file()}
            out = td / "b1"
            argv = ["b1", "--full_dir", str(full), "--dose_dir", str(dose),
                    "--data_dir", str(td / "images"), "--out_dir", str(out), "--seed", "42",
                    "--patients", "101", "102", "--prefix_epochs", "0", "1",
                    "--exponents", "12", "18", "24"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(b1.e0, "load_modules", return_value=(legacy, score, pilot)), \
                 mock.patch.object(pilot, "load_dataset", return_value=(X, y, groups)):
                b1.main()
            manifest = legacy.read_json(out / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["input_files_unchanged"])
            self.assertFalse(manifest["tf32"]["cudnn_allow_tf32"])
            after = {q: legacy.sha(Path(q)) for q in before}
            self.assertEqual(before, after)             # truth directories are read-only
            summary = legacy.read_json(out / "b1_summary.json")
            for gate in ("G1", "G2", "G3", "G4", "G5"):
                self.assertIn(gate, summary["gates"])
            self.assertTrue(all(r["exactly_zero"] for r in summary["gates"]["G3"]))
            self.assertTrue(all(r["passed"] for r in summary["gates"]["G4"]))
            payload = legacy.read_json(out / "b1_rows.json")
            rows = payload["rows"]
            # this fixture has one batch per epoch, so "one step" and "one epoch" coincide
            self.assertEqual(payload["batches_per_epoch"], 1)
            self.assertEqual(len(rows), 2 * len(payload["prefixes"]) * 3)
            for r in rows:
                for field in ("param_relative_error", "param_relative_error_float64",
                              "param_cosine_float64", "pred_relative_error"):
                    self.assertIn(field, r)
            # the two questions must be reported apart, and the epsilon attribution present
            self.assertEqual(set(summary["linear_range"]), set(summary["float32_resolvability"]))
            self.assertIn("epsilon_concentration", summary)
            self.assertGreater(summary["epsilon_concentration"]["worst_top_0p1pct_share_of_tangent_sq"], 0.0)

    def test_output_must_not_overlap_a_truth_directory(self):
        with tempfile.TemporaryDirectory() as td:
            full = Path(td) / "full"; full.mkdir()
            argv = ["b1", "--full_dir", str(full), "--dose_dir", str(full), "--data_dir", ".",
                    "--out_dir", str(full / "inside"), "--seed", "42"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(b1.e0, "load_modules", return_value=(legacy, score, pilot)):
                with self.assertRaises(RuntimeError):
                    b1.main()


if __name__ == "__main__":
    unittest.main()
