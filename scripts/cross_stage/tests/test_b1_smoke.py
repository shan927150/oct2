#!/usr/bin/env python3
"""Correctness invariants; no assertion that a scientific precision failure MUST occur."""
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
from torch.func import jvp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core
import stage1_trajectory_core as tr

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

b1 = load("b1_review_test", HERE / "16_b1_smoke.py")
fixture = load("b1_review_fixture", HERE / "tests/test_e0_residual_decomposition.py")
legacy, score, pilot = b1.e0.load_modules()
DEVICE = pilot.DEVICE
torch.set_num_threads(min(2, torch.get_num_threads()))
CFG = {"lr": 1e-3, "batch_size": 4, "wd": 1e-5, "deterministic": True}


def tiny(steps=4):
    X = np.random.default_rng(3).random((16, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), 4).astype(np.int64)
    orders = pilot.make_epoch_orders(np.arange(8), 2, 700042)
    prefixes = sorted({1, steps})
    _, _, native, masks, _ = tr.faithful_run(pilot, X, y, orders, 42, CFG, 0., (), steps,
                                             prefixes=prefixes, record_masks=steps)
    pilot.seed_everything(42, True)
    model = pilot.build_model("cnn", 1, 128, 4)
    theta0 = tr.flat(model)
    tr.swap_in_recorder(model)
    plan = tr.batch_plan(orders, 4, steps)
    return X, y, orders, prefixes, native, masks, model, theta0, plan


class B1Tests(unittest.TestCase):
    def test_mask_capture_preserves_rng_and_zero_activation_draws(self):
        model = nn.Sequential(nn.Dropout(.2)).to(DEVICE).train()
        x = torch.zeros((8, 128), device=DEVICE)
        x[:, ::3] = 1
        pilot.seed_everything(321, True)
        pre = tr.rng_states()
        expected = nn.functional.dropout(torch.ones_like(x), p=.2, training=True)
        post = tr.rng_states()
        torch.set_rng_state(pre["torch_cpu"])
        if x.is_cuda:
            torch.cuda.set_rng_state_all(pre["torch_cuda"])
        capture = tr.MaskCapture(model, 1)
        actual = model(x)
        capture.remove()
        self.assertTrue(core.exact_tree(post, tr.rng_states()))
        self.assertTrue(torch.equal(expected.cpu(), capture.store[0]))
        self.assertTrue(torch.equal(actual, x*expected))
        # Nonzero draws at zero inputs must survive; output!=0 cannot recover these.
        self.assertGreater(torch.count_nonzero(capture.store[0][x.cpu() == 0]).item(), 0)

    def test_native_loop_matches_original_trainer_and_replayed_masks(self):
        X, y, orders, prefixes, native, masks, model, th, plan = tiny()
        reference, _ = pilot.train_classifier_from_orders(X, y, orders, [8, 9, 10, 11], 42, 128,
            CFG["lr"], 4, CFG["wd"], True, deletion_mode="fixed_mask")
        self.assertTrue(torch.equal(native[4]["theta"], tr.flat(reference).cpu()))
        for alpha in (0., .125):
            _, _, a, _, _ = tr.faithful_run(pilot, X, y, orders, 42, CFG, alpha, [int(orders[0][0])],
                                           4, prefixes=prefixes)
            _, _, b, _, _ = tr.faithful_run(pilot, X, y, orders, 42, CFG, alpha, [int(orders[0][0])],
                                           4, prefixes=prefixes, masks=masks)
            for k in prefixes:
                self.assertTrue(b1.state_match(a[k], b[k])["bitwise_equal"])

    def test_adam_primal_against_independent_native_optimizer(self):
        # Nonzero carried moments, including coupled weight decay over several updates.
        p = nn.Parameter(torch.tensor([.1, -.2, .3], dtype=torch.float64, device=DEVICE))
        opt = torch.optim.Adam([p], lr=CFG["lr"], weight_decay=CFG["wd"], foreach=False)
        th, m, v = p.detach().clone(), torch.zeros_like(p), torch.zeros_like(p)
        for step in range(1, 5):
            raw = torch.tensor([.01*step, -.03, .00001], dtype=p.dtype, device=DEVICE)
            p.grad = raw.clone()
            opt.step()
            th, m, v = tr.adam_primal(th, m, v, raw.add(th, alpha=CFG["wd"]), step, CFG["lr"])
            for got, expected in ((th, p.detach()), (m, opt.state[p]["exp_avg"]), (v, opt.state[p]["exp_avg_sq"])):
                torch.testing.assert_close(got, expected, rtol=2e-15, atol=1e-18)

    def test_manual_adam_chain_rule_matches_independent_jvp(self):
        gen = torch.Generator().manual_seed(987)
        vals = [torch.randn(9, generator=gen, dtype=torch.float64, device="cpu").to(DEVICE) for _ in range(8)]
        th, m, vv, g, ut, um, uv, q = vals
        v = vv.square()+.1
        primal, tangent = jvp(lambda a,b,c,d: tr.adam_primal(a,b,c,d,7,CFG["lr"]),
                               (th,m,v,g), (ut,um,uv,q))
        got = tr.adam_tangent(primal[1], primal[2], g, q, ut, um, uv, 7, CFG["lr"])
        for a, b in zip(got, tangent):
            torch.testing.assert_close(a, b, rtol=5e-13, atol=2e-15)

    def test_first_step_sign_epsilon_and_exact_zero_composite(self):
        g = torch.tensor([0., 1e-8, -1e-8, 1e-3], dtype=torch.float64, device=DEVICE)
        q = torch.tensor([1., -.4, .7, 0.], dtype=g.dtype, device=DEVICE)
        z = torch.zeros_like(g)
        _, m, v = tr.adam_primal(z,z,z,g,1,CFG["lr"])
        u, um, uv = tr.adam_tangent(m,v,g,q,z,z,z,1,CFG["lr"])
        expected = -CFG["lr"]*tr.EPS/(g.abs()+tr.EPS).square()*q
        torch.testing.assert_close(u, expected, rtol=2e-14, atol=1e-12)
        self.assertTrue(torch.isfinite(u).all())
        report = tr.first_step_epsilon(g,q,u,CFG["lr"])
        self.assertLess(report["analytic_u1_check"]["relative_error"], 1e-14)
        self.assertAlmostEqual(report["update_gain_max"], CFG["lr"]/tr.EPS)
        # Large slope alone cannot imply large patient sensitivity: q can vanish.
        _, _, _ = tr.adam_tangent(m,v,g,z,z,z,z,1,CFG["lr"])
        self.assertEqual(tr.first_step_epsilon(g,z,z,CFG["lr"])["tangent_norm"], 0)

    def test_degenerate_zero_second_moment_is_not_silently_clipped(self):
        z = torch.zeros(2, dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "Degenerate"):
            tr.adam_tangent(torch.ones_like(z),z,z,z,z,z,z,1,CFG["lr"])

    def test_absent_direction_both_precisions_all_states_and_primal(self):
        X,y,orders,prefixes,native,masks,model,th,plan = tiny()
        for dtype in (torch.float32, torch.float64):
            base = tr.functional_trajectory(model,X,y,CFG,DEVICE,th.to(dtype),plan,masks,(),0.,prefixes)
            tan = tr.functional_trajectory(model,X,y,CFG,DEVICE,th.to(dtype),plan,masks,[12,13],0.,prefixes,True)
            for k in prefixes:
                self.assertTrue(b1.state_match(base[k],tan[k])["bitwise_equal"])
                for f in ("theta","m","v"):
                    self.assertFalse(tan[k]["u_"+f].requires_grad)
                    self.assertEqual(torch.count_nonzero(tan[k]["u_"+f]).item(),0)

    def test_nonzero_tangent_and_prediction_finite_difference_float64(self):
        X,y,orders,prefixes,native,masks,model,th,plan = tiny(2)
        present = [int(orders[0][0])]
        th = th.double()
        tan = tr.functional_trajectory(model,X,y,CFG,DEVICE,th,plan,masks,present,0.,prefixes,True)
        base = tr.functional_trajectory(model,X,y,CFG,DEVICE,th,plan,masks,present,0.,prefixes)
        interface = np.arange(8)
        for k in prefixes:
            p0, pu = tr.prediction_jvp(model,X,interface,base[k]["theta"],tan[k]["u_theta"],DEVICE,batch=3)
            self.assertEqual(p0.dtype,torch.float64)
            self.assertGreater(tr.norm(tan[k]["u_theta"]),0)
            errors = []
            for exponent in (20,24,28):
                alpha = tr.exact_alpha(exponent,torch.float64)
                pert = tr.functional_trajectory(model,X,y,CFG,DEVICE,th,plan,masks,present,alpha,prefixes)
                p1,_ = tr.prediction_jvp(model,X,interface,pert[k]["theta"],None,DEVICE,batch=3)
                a = tr.compare(tr.fd(pert[k]["theta"],base[k]["theta"],alpha),tan[k]["u_theta"])
                b = tr.compare(tr.fd(p1,p0,alpha),pu)
                errors.append((a["relative_error"],b["relative_error"]))
            self.assertLess(min(max(pair) for pair in errors), .02, errors)
            self.assertLess(errors[-1][0], errors[0][0])

    def test_dose_representability_is_precision_specific(self):
        tr.exact_alpha(24,torch.float32)
        tr.exact_alpha(53,torch.float64)
        for exp,dtype in ((25,torch.float32),(54,torch.float64),(0,torch.float64)):
            with self.assertRaises(ValueError): tr.exact_alpha(exp,dtype)

    def test_fd_converts_before_subtraction_and_uses_own_baseline(self):
        base=torch.tensor([1.,2.],dtype=torch.float32)
        other=base.clone(); other[0]+=1e-6
        self.assertEqual(tr.norm(tr.fd(base,base,2**-24)),0)
        self.assertGreater(tr.norm(tr.fd(base,other,2**-24)),1)
        self.assertEqual(tr.fd(other,base,.1).dtype,torch.float64)

    def test_band_needs_joint_adjacent_doses_and_handles_zero(self):
        rows=[]
        for exp,err in ((10,.5),(14,.001),(18,.5)):
            r={"exponent":exp,"alpha":2**-exp,"exposed":True}
            for f in ("param","pred","moment1","moment2"):
                r.update({f+"_relative_error":err,f+"_cosine":1.,f+"_absolute_error":err,f+"_reference_norm":1.})
            rows.append(r)
        self.assertFalse(b1.resolved_band(rows,.01,.9999)["resolved"])
        rows[-1]["param_relative_error"]=rows[-1]["pred_relative_error"]=.001
        self.assertTrue(b1.resolved_band(rows,.01,.9999)["resolved"])
        for r in rows:
            r["exposed"]=False
            for f in ("param","pred","moment1","moment2"):
                r[f+"_absolute_error"]=r[f+"_reference_norm"]=0.
                r[f+"_cosine"]=None
        self.assertEqual(b1.resolved_band(rows,.01,.9999)["status"],"ZERO_CONTROL")

    def test_checkpoint_comparison_checks_moments_and_rng(self):
        with tempfile.TemporaryDirectory() as td:
            X,y,groups,full,dose=fixture.build_fixture(Path(td))
            ctx=legacy.load_context
            ref=legacy.load_payload(full/"checkpoints/baseline_seed42_epochs/epoch002.pt",pilot)
            snap={"state_dict":ref["state_dict"],"optimizer_state":ref["metadata"]["optimizer_state"],
                  "rng_states":ref["metadata"]["rng_states"]}
            self.assertTrue(all(b1.checkpoint_match(snap,ref).values()))
            bad=tr.cpu_tree(snap)
            next(iter(bad["optimizer_state"]["state"].values()))["exp_avg"].view(-1)[0]+=1.
            self.assertFalse(b1.checkpoint_match(bad,ref)["optimizer_bitwise_equal"])
            bad=tr.cpu_tree(snap); bad["rng_states"]["torch_cpu"][0]^=1
            self.assertFalse(b1.checkpoint_match(bad,ref)["rng_bitwise_equal"])

    def test_end_to_end_one_step_still_replays_all_frozen_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            X,y,groups,full,dose=fixture.build_fixture(td)
            before={str(q):legacy.sha(q) for root in (full,dose) for q in root.rglob("*") if q.is_file()}
            flags=tr.tf32_settings()
            argv=["b1","--full_dir",str(full),"--dose_dir",str(dose),"--data_dir",str(td/"images"),
                  "--out_dir",str(td/"b1"),"--seed","42","--patients","101","102",
                  "--prefix_epochs","0","--exponents","18","24","--exponents64","22","24","28"]
            with mock.patch.object(sys,"argv",argv), mock.patch.object(b1.e0,"load_modules",return_value=(legacy,score,pilot)), \
                 mock.patch.object(pilot,"load_dataset",return_value=(X,y,groups)):
                b1.main()
            m=legacy.read_json(td/"b1/manifest.json")
            self.assertEqual(m["status"],"complete")
            self.assertTrue(m["input_files_unchanged"])
            self.assertEqual(flags,m["tf32"])
            self.assertEqual(flags,m["tf32_after"])
            summary=legacy.read_json(td/"b1/b1_summary.json")
            g1=summary["gates"]["G1"]
            self.assertEqual([r["epoch"] for r in g1["checked_against_original_epoch_checkpoints"]],[1,2])
            self.assertEqual(g1["status"],"PASS_FULL_REPLAY")
            payload=legacy.read_json(td/"b1/b1_rows.json")
            self.assertEqual(payload["prefixes"],[1])
            self.assertEqual(len(payload["rows"]),2*(2*2+3))
            self.assertFalse(summary["ready_for_B2"])
            self.assertEqual(before,{q:legacy.sha(Path(q)) for q in before})

    def test_prefix_outside_frozen_horizon_rejected(self):
        with self.assertRaises(ValueError): tr.batch_plan([[1,2]],2,2)

    def test_output_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            argv=["b1","--full_dir",td,"--dose_dir",td,"--data_dir",td,
                  "--out_dir",str(Path(td)/"bad"),"--seed","42"]
            with mock.patch.object(sys,"argv",argv), self.assertRaises(RuntimeError): b1.main()


if __name__ == "__main__":
    unittest.main()
