#!/usr/bin/env python3
"""Portable CPU/GPU integration check of 08 -> 05 -> 07 -> 09.

The deliberately short training is a bookkeeping/numerics test, not evidence
that the OCT attack or influence approximation is useful. It does not use OCT.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=16,
                    help="synthetic test only; 128 gives the slower full-resolution check")
    args = ap.parse_args()
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    (data/"SYNTHETIC_TEST_DATA.json").write_text(json.dumps({"synthetic": True, "seed": 0}))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    for ci, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        folder = data / "OCT2017" / "train" / cls; folder.mkdir(parents=True, exist_ok=True)
        for pid in range(40):
            off, freq = rng.normal(0, .3), rng.uniform(.8, 1.2)
            for k in range(4):
                base = np.sin((ci+1)*freq*xx/8 + off)*.5 + np.cos((ci+1)*yy/11)*.3
                img = base + rng.normal(0, .8, size=base.shape)
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                Image.fromarray(img).save(folder/f"{cls}-{pid+1000*ci}-{k}.jpeg")
    commands = []

    def run(script, *parts):
        if script.startswith(("05_", "07_", "08_")):
            prefix = [sys.executable, str(SCRIPTS/"tests"/"run_with_small_images.py"),
                      "--image_size", str(args.image_size)]
        else:
            prefix = [sys.executable]
        command = prefix + [str(SCRIPTS/script)] + [str(v) for p in parts for v in p]
        commands.append(command)
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        (root/"commands.json").write_text(json.dumps(commands, indent=2))

    split = ["--data_dir", data, "--n_total_samples", 640, "--target_data_size", 100,
             "--shadow_data_size", 100, "--n_shadow", 2]
    panel = root/"panel"/"eligibility_preflight.json"
    run("08_eligibility_preflight.py", split, ["--output_dir", root/"panel", "--classes", 1, 2,
        "--min_patient_images", 1, "--max_patient_images", 30, "--patients_per_class_per_shadow", 1,
        "--n_affected_shadows", 2])
    common = split + ["--n_patients", 2, "--classes", 1, 2, "--min_patient_images", 1,
                     "--max_patient_images", 30, "--shadow_epochs", 2, "--attack_epochs", 5,
                     "--noop_replays", 1, "--enforce_noop_gate", "--deletion_mode", "fixed_mask",
                     "--attack_seeds", 5101, 5102, "--affected_shadow", 1,
                     "--patient_panel_json", panel, "--require_complete_panel"]
    cases = {"full": ["--seeds", 42, 43],
             "dose": ["--seeds", 42, "--deletion_weight", .25],
             "late": ["--seeds", 42, "--removal_epochs", "1:2"]}
    for name, extra in cases.items():
        run("05_end_to_end_patient_loo_pilot.py", common, ["--output_dir", root/name], extra)
        run("07_cross_stage_score_ladder.py", ["--pilot_dir", root/name, "--data_dir", data,
            "--cg_iters", 4, "--hvp_batch", 32, "--lanczos_iters", 4,
            "--damping_shadow_grid", .3, "--classes", 1, 2])
        run("09_seed_variance.py", ["--pilot_dirs", root/name, "--out_dir", root/name/"seed_stats"])
        run("06_pilot_seed_stats.py", ["--pilot_dir", root/name])
    tables = {}
    for name in cases:
        folder = root/name
        with (folder/"score_ladder"/"ladder_rows.csv").open() as f:
            rows = list(csv.DictReader(f)); tables[name] = rows
        with (folder/"score_ladder"/"ladder_attack_seed_rows.csv").open() as f:
            reps = list(csv.DictReader(f))
        for row in rows:
            pair = [r for r in reps if r["seed"] == row["seed"] and r["patient_id"] == row["patient_id"]]
            assert [int(r["attack_seed"]) for r in pair] == [5101, 5102]
            assert float(row["J00_reproduction_max_abs_diff"]) <= 1e-6
            for key in ("actual_value", "L1_lin_value", "L2_lin_value", "L2_retrain_value",
                        "L3_lin_value", "L3_retrain_value", "L3_hybrid_full", "frozen_h", "frozen_self"):
                if row.get(key):
                    assert np.isclose(float(row[key]), np.mean([float(r[key]) for r in pair]), rtol=1e-5, atol=1e-9), key
            probs = np.load(folder/"runs"/f"seed{row['seed']}_patient{row['patient_id']}_attack_probs.npz")
            assert "target_patient_id" in probs and f"J00_class{row['oct_class']}" in probs
            query_y = probs["target_membership"][probs["target_classes"] == int(row["oct_class"])]
            truth = json.loads((folder/"runs"/f"seed{row['seed']}_patient{row['patient_id']}.json").read_text())
            for condition, payload in truth["conditions"].items():
                if payload is None:
                    continue
                logs = probs[f"{condition}_class{row['oct_class']}_log_probs"]
                losses = -logs[:, np.arange(len(query_y)), query_y].astype(np.float64).mean(axis=1)
                assert np.allclose(losses, payload["per_class"][row["oct_class"]]["per_rep_cross_entropy"], rtol=0, atol=1e-12)
        summary = json.loads((folder/"score_ladder"/"ladder_summary.json").read_text())["analysis"]
        assert summary["per_patient_seed"]["L2_retrain_value~actual_value"]["n"] == len(rows)
        if name == "late":
            assert all(r["L3_lin_value"] == "" and r["frozen_self"] == "" for r in rows)
    for dose in tables["dose"]:
        full = next(r for r in tables["full"] if r["seed"] == dose["seed"] and r["patient_id"] == dose["patient_id"])
        for key in ("L3_lin_value", "frozen_h", "frozen_self"):
            assert np.isclose(float(dose[key]), .25*float(full[key]), rtol=1e-5, atol=1e-9), key
    fits = json.loads((root/"full"/"seed_stats"/"seed_variance_components.json").read_text())["components"]
    assert fits and all(f["R"] == 2 and f["K"] == 2 and f["design"] == "crossed_fixed_panel" for f in fits)
    (root/"validation_result.json").write_text(json.dumps({"status": "passed", "cases": list(cases),
        "checks": ["frozen_panel", "fixed_mask_noop", "J00_seed_pairing", "per_seed_score_means",
                   "dose_scaling", "window_NA", "independent_layer_coverage", "crossed_CE_AUC_Brier", "stable_query_CE"],
        "image_size": args.image_size,
        "scope": "Synthetic CPU/GPU mechanism validation only; not score-quality evidence"}, indent=2))
    print("SYNTHETIC CHAIN PASSED", flush=True)


if __name__ == "__main__":
    main()
