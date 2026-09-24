#!/usr/bin/env python3
"""Tests for the small-panel paired LOO continuation (24_A_smallpanel_loo.py).

Synthetic 2-to-4 epoch tests use the production pilot. These verify code
behavior on tiny tensors, not real OCT E100/E140 fidelity; Delta smoke is required.
"""
import importlib.util
import json
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


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sp = load("a24_test", "24_A_smallpanel_loo.py")
legacy = sp.legacy
score = legacy.module_from_path("score07_a24_test", HERE / "07_cross_stage_score_ladder.py")
pilot = score.import_pilot_module()
torch.set_num_threads(min(2, torch.get_num_threads()))
T0, TOTAL = 2, 4


def build_fixture(td):
    rng = np.random.default_rng(7)
    X = rng.random((16, 1, 32, 32), dtype=np.float32)
    y = np.tile(np.arange(4), 4).astype(np.int64)
    groups = np.arange(100, 116)
    split = {"target_train_idx": [8, 9, 10, 11], "target_test_idx": [12, 13, 14, 15],
             "shadow_models": [{"train_idx": [0, 1, 2, 3], "test_idx": [4, 5, 6, 7]}]}
    # one DME + one DRUSEN patient in the affected shadow train set
    panel = [{"patient_id": 101, "oct_class": 1, "class_name": "DME", "n_images": 1, "raw_indices": [1]},
             {"patient_id": 102, "oct_class": 2, "class_name": "DRUSEN", "n_images": 1, "raw_indices": [2]}]
    full = Path(td) / "full"
    with mock.patch.object(sys, "argv", ["pilot"]):
        p = pilot.parse_args()
    p.n_shadow, p.affected_shadow, p.seeds, p.classes, p.n_patients = 1, 0, [42], [1, 2], 2
    p.shadow_epochs, p.shadow_batch_size, p.deletion_mode = T0, 4, "fixed_mask"
    p.output_dir, p.n_total_samples = str(full), 16
    legacy.write_json(full / "splits/fresh_patient_split.json", split)
    h = legacy.sha(full / "splits/fresh_patient_split.json")[:16]
    legacy.write_json(full / "experiment_summary.json", {"status": "complete", "split_sha256": h})
    legacy.write_json(full / "experiment_config.json", {
        "training_numerics": pilot.TRAINING_NUMERICS, "args": vars(p),
        "oct_config": {"target_l2": 1e-5, "optimizer_type": "adam"}})
    legacy.write_json(full / "selected_patients.json", {"split_sha256": h, "patients": panel})
    # frozen E100 (=T0) truth endpoints: baseline + full/dose01 LOO for each patient
    e100_full, e100_dose = Path(td) / "e100_full", Path(td) / "e100_dose"
    orders = pilot.make_epoch_orders(np.asarray(split["shadow_models"][0]["train_idx"]), T0, 42 + 700000)
    base, bmetrics = pilot.train_classifier_from_orders(X, y, orders, split["shadow_models"][0]["test_idx"], 42, 128,
              p.shadow_lr, 4, 1e-5, True, deletion_mode="fixed_mask")
    for d in (e100_full, e100_dose):  # truth endpoints carry post-train RNG fingerprint, like the real pilot
        pilot.save_model(d / "checkpoints/shadow_0_baseline_seed42.pt", base, {"seed": 42, "metrics": bmetrics})
    certificates = {d: [] for d in (e100_full,e100_dose)}
    replay = load("truth_replay_fixture", "21_A_truth_replay.py")
    for pat in panel:
        for d, w in ((e100_full, 1.0), (e100_dose, 0.1)):
            m, mm = pilot.train_classifier_from_orders(X, y, orders, split["shadow_models"][0]["test_idx"], 42, 128,
                    p.shadow_lr, 4, 1e-5, True, excluded_indices=pat["raw_indices"],
                    deletion_mode="fixed_mask", deletion_weight=w,
                    epoch_checkpoint_dir=d/f"fixture_{pat['patient_id']}", epoch_checkpoints=[T0])
            pilot.save_model(d / f"checkpoints/shadow_0_seed42_patient{pat['patient_id']}.pt", m, {"seed": 42, "metrics": mm})
            cp = legacy.load_payload(d/f"fixture_{pat['patient_id']}/epoch{T0:03d}.pt", pilot)
            certificates[d].append(dict(role='loo',seed=42,training_instance=f"shadow_0_seed42_patient{pat['patient_id']}",
                passed=True,continued_optimizer_sha256_at_E_star=replay.tree_sha(cp['metadata']['optimizer_state'])))
    for d, checks in certificates.items():
        legacy.write_json(d/'route_a_epoch50_replay_checks.json',dict(status='complete',extended_epochs=T0,checks=checks))
        np.savez(d/'stage1_order_seed42.npz',raw_index_order=orders)
    return {"X": X, "y": y, "groups": groups, "full": full, "e100_full": e100_full, "e100_dose": e100_dose,
            "split": split, "panel": panel, "pargs": p}


def run_cell(fx, out_root, idx, schedule="1@1e-4,1@1e-5", extra=None):
    argv = ["--full_dir", str(fx["full"]), "--e100_full_dir", str(fx["e100_full"]),
            "--e100_dose_dir", str(fx["e100_dose"]), "--data_dir", "x", "--out_root", str(out_root),
            "--array_index", str(idx), "--panel", "101", "102", "--resume_epoch", str(T0),
            "--total_epochs", str(TOTAL), "--decay_schedule", schedule, "--observe_every", "1",
            "--hvp_batch", "4", *(extra or [])]
    real_loader = legacy.module_from_path
    with mock.patch.object(legacy, "module_from_path", side_effect=lambda name,path: score if Path(path).name == "07_cross_stage_score_ladder.py" else real_loader(name,path)), \
         mock.patch.object(score, "import_pilot_module", return_value=pilot), \
         mock.patch.object(pilot, "load_dataset", return_value=(fx["X"], fx["y"], fx["groups"])):
        sp.main(argv)


class SmallPanelTests(unittest.TestCase):
    def test_e100_replay_exact_and_decay_diverges(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            # cell order: seed x patient x dose x arm. idx0 = seed42,p101,full,control
            out = Path(td) / "sp"
            run_cell(fx, out, 0)  # full control
            man = legacy.read_json(out / "seed42_patient101_full_control/manifest.json")
            self.assertEqual(man["status"], "complete")
            self.assertTrue(man["replay_e100_passed"])
            # E100 replay check verified BOTH params and the post-train RNG fingerprint
            rc = legacy.read_json(out / "seed42_patient101_full_control/replay_checks.json")[0]
            self.assertTrue(rc["parameters_equal"] and rc["rng_fingerprint_equal"] and rc["passed"])
            # E100 checkpoint bitwise == frozen truth full LOO endpoint, and carries full continuation state
            e100 = legacy.load_payload(out / "seed42_patient101_full_control/checkpoints/epoch002.pt", pilot)
            ref = legacy.load_payload(fx["e100_full"] / "checkpoints/shadow_0_seed42_patient101.pt", pilot)
            self.assertTrue(core.exact_tree(e100["state_dict"], ref["state_dict"]))
            self.assertIn("optimizer_state", e100["metadata"])
            self.assertIn("rng_states", e100["metadata"])
            # the recorded gradient is the deletion-weighted training objective, not the unweighted one
            import csv as _csv
            with (out / "seed42_patient101_full_control/curve.csv").open() as f:
                crows = list(_csv.DictReader(f))
            self.assertTrue(all(r.get("weighted_eval_grad_norm") not in ("", None) for r in crows))
            # control E140 == fresh 4-epoch fixed_mask training from scratch (exact continuation)
            orders = pilot.make_epoch_orders(np.asarray(fx["split"]["shadow_models"][0]["train_idx"]), TOTAL, 42 + 700000)
            direct, _ = pilot.train_classifier_from_orders(fx["X"], fx["y"], orders,
                        fx["split"]["shadow_models"][0]["test_idx"], 42, 128, fx["pargs"].shadow_lr, 4, 1e-5, True,
                        excluded_indices=[1], deletion_mode="fixed_mask", deletion_weight=1.0)
            e140 = legacy.load_payload(out / "seed42_patient101_full_control/checkpoints/epoch004.pt", pilot)
            self.assertTrue(core.exact_tree(e140["state_dict"], direct.state_dict()))
            # decay arm shares E100 with control but diverges by E140
            run_cell(fx, out, 1)  # full decay
            dman = legacy.read_json(out / "seed42_patient101_full_decay/manifest.json")
            self.assertTrue(dman["replay_e100_passed"])
            e100d = legacy.load_payload(out / "seed42_patient101_full_decay/checkpoints/epoch002.pt", pilot)
            self.assertTrue(core.exact_tree(e100d["state_dict"], ref["state_dict"]))
            e140d = legacy.load_payload(out / "seed42_patient101_full_decay/checkpoints/epoch004.pt", pilot)
            self.assertFalse(core.exact_tree(e140d["state_dict"], e140["state_dict"]))
            self.assertEqual(e140d["metadata"]["learning_rate_at_epoch"], 1e-5)

    def test_guards(self):
        with tempfile.TemporaryDirectory() as td:
            fx = build_fixture(td)
            with self.assertRaises(RuntimeError):   # decay schedule sum != extra (idx 1 = decay arm)
                run_cell(fx, Path(td) / "bad", 1, schedule="1@1e-4")
            with self.assertRaises(RuntimeError):  # output overlaps a protected dir
                run_cell(fx, fx["full"] / "inside", 0)
            # a corrupted E100 truth endpoint must fail the replay check
            import torch as _t
            ref = fx["e100_full"] / "checkpoints/shadow_0_seed42_patient101.pt"
            payload = _t.load(ref, map_location="cpu", weights_only=False)
            for v in payload["state_dict"].values():
                v.add_(1.0)
            _t.save(payload, ref)
            with self.assertRaises(RuntimeError):
                run_cell(fx, Path(td) / "corrupt", 0)

    def test_optimizer_corruption_blocks_replay(self):
        with tempfile.TemporaryDirectory() as td:
            fx=build_fixture(td)
            cp=fx['e100_full']/'route_a_epoch50_replay_checks.json'
            cert=legacy.read_json(cp)
            cert['checks'][0]['continued_optimizer_sha256_at_E_star']='0'*64
            legacy.write_json(cp,cert)
            with self.assertRaises(RuntimeError): run_cell(fx,Path(td)/'out',0)

    def test_weighted_gradient_matches_autograd(self):
        torch.manual_seed(99)
        model=torch.nn.Linear(3,2).to(pilot.DEVICE)
        X=np.arange(18,dtype=np.float32).reshape(6,3)/18
        y=np.array([0,1,0,1,0,1])
        for dose in (1.,.1):
            excluded=torch.tensor([1,4],device=pilot.DEVICE)
            got=sp.weighted_objective_grad(model,X,y,np.arange(6),excluded,dose,2,1e-5)
            per=torch.nn.functional.cross_entropy(model(torch.tensor(X,device=pilot.DEVICE)),torch.tensor(y,device=pilot.DEVICE),reduction='none')
            weights=torch.ones(6,device=pilot.DEVICE);weights[excluded]=1-dose
            obj=(weights*per).mean()+1e-5/2*sum((v*v).sum() for v in model.parameters())
            g=torch.cat([v.reshape(-1) for v in torch.autograd.grad(obj,list(model.parameters()))])
            self.assertAlmostEqual(got,core.norm(g),places=6)

    def test_smoke_stops_at_resume(self):
        with tempfile.TemporaryDirectory() as td:
            fx=build_fixture(td);out=Path(td)/'out'
            run_cell(fx,out,0,extra=['--stop_at_resume'])
            d=out/'seed42_patient101_full_control'
            self.assertEqual(legacy.read_json(d/'manifest.json')['status'],'smoke_complete')
            self.assertFalse((d/'checkpoints/epoch004.pt').exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
