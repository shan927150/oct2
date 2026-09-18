#!/usr/bin/env python3
"""Synthetic wiring check for Route A step 2 (submit_A_step2.sh -> 10 -> 05 -> 07 -> 12 -> 22).

The 05/07 command lines are taken verbatim from 10_run_calibration.py --dry_run
with the same options submit_A_step2.sh passes (plus panel-size options for
the tiny synthetic split). Only test-size overrides are appended (last flag
wins in argparse): attack gate thresholds and solver iteration counts.
It proves the plumbing and file formats, not anything about OCT.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
ADAPTER = SCRIPTS / "tests" / "run_with_small_images.py"


def make_data(data):
    data.mkdir(parents=True, exist_ok=True)
    (data / "SYNTHETIC_TEST_DATA.json").write_text(json.dumps({"synthetic": True, "seed": 0}))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    for ci, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        folder = data / "OCT2017" / "train" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for pid in range(40):
            off, freq = rng.normal(0, .3), rng.uniform(.8, 1.2)
            for k in range(4):
                base = np.sin((ci + 1) * freq * xx / 8 + off) * .5 + np.cos((ci + 1) * yy / 11) * .3
                img = base + rng.normal(0, .8, size=base.shape)
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                Image.fromarray(img).save(folder / f"{cls}-{pid + 1000 * ci}-{k}.jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.output_dir).resolve()
    data = root / "data"
    step2 = root / "runs" / "A" / f"l3_at_E{args.epochs}"
    make_data(data)
    small = [sys.executable, str(ADAPTER), "--image_size", str(args.image_size)]

    def run(cmd, capture=False):
        print("RUN", shlex.join(map(str, cmd)), flush=True)
        result = subprocess.run(list(map(str, cmd)), check=True, text=True,
                                capture_output=capture)
        return result.stdout if capture else None

    split = ["--data_dir", data, "--n_total_samples", 640, "--target_data_size", 100,
             "--shadow_data_size", 100, "--n_shadow", 2]
    run(small + [SCRIPTS / "08_eligibility_preflight.py", *split, "--output_dir", step2 / "panel",
                 "--classes", 1, 2, "--min_patient_images", 1, "--max_patient_images", 30,
                 "--patients_per_class_per_shadow", 1, "--n_affected_shadows", 2])
    shadow = json.loads((step2 / "panel/eligibility_preflight.json").read_text())["proposed_affected_shadows"][0]
    calibration = [sys.executable, SCRIPTS / "10_run_calibration.py", "--root", step2, "--data_dir", data,
                   "--shadow_epochs", args.epochs, "--stage1_seeds", 42, 43, "--attack_seeds", 5101, 5102,
                   "--patients_per_class", 1, "--n_total_samples", 640, "--target_data_size", 100,
                   "--shadow_data_size", 100, "--n_shadow", 2, "--min_patient_images", 1,
                   "--max_patient_images", 30, "--dry_run"]
    overrides_05 = ["--gate_min_queries_per_label", 1, "--gate_min_class_auc", 0,
                    "--gate_min_class_balanced_accuracy", 0, "--attack_epochs", 5]
    for condition in ("full", "dose01"):
        printed = run(calibration + ["--phase", "truth", "--condition", condition], capture=True)
        commands = [shlex.split(line) for line in printed.splitlines() if line.strip()]
        pilot_cmd = next(c for c in commands if c[1].endswith("05_end_to_end_patient_loo_pilot.py"))
        joined = " ".join(pilot_cmd)
        assert f"--shadow_epochs {args.epochs}" in joined, joined
        assert f"--save_epoch_checkpoints {args.epochs // 2} {args.epochs}" in joined, joined
        assert f"shadow{shadow}_{condition}" in joined, joined
        assert ("--deletion_weight 0.1" in joined) == (condition == "dose01"), joined
        run(small + pilot_cmd[1:] + overrides_05)
    full, dose = step2 / f"shadow{shadow}_full", step2 / f"shadow{shadow}_dose01"
    printed = run(calibration + ["--phase", "score", "--condition", "full", "--damping_attack", .2,
                                 "--damping_shadow", 1, "--damping_shadow_grid", .8, 1.2], capture=True)
    commands = [shlex.split(line) for line in printed.splitlines() if line.strip()]
    score_cmd = next(c for c in commands if c[1].endswith("07_cross_stage_score_ladder.py"))
    assert str(full / "score_ladder_A0.2_S1") in score_cmd, score_cmd
    run(small + score_cmd[1:] + ["--cg_iters", 4, "--hvp_batch", 32, "--lanczos_iters", 4])
    diag = [SCRIPTS / "12_stage1_diagnostics_v11.py", "--full_dir", full, "--dose_dir", dose, "--data_dir", data,
            "--seeds", 42, 43]
    run(small + diag + ["--mode", "preflight", "--out_dir", step2 / "diag_preflight"])
    run(small + diag + ["--mode", "damping", "--out_dir", step2 / "diag_damping", "--damping_grid", .01, 1,
                        "--krylov_steps", 6, 12, "--lanczos_iters", 4, "--probe_seeds", 17, "--hvp_batch", 32,
                        "--j00_tol", 1e-5])
    printed = run([sys.executable, SCRIPTS / "22_A_compare_E.py", "--new_root", step2, "--baseline_root", root,
                   "--ref_full", full, "--ref_preflight", step2 / "diag_preflight",
                   "--ref_damping", step2 / "diag_damping"], capture=True)
    comparison = json.loads((step2 / "compare_vs_E50/comparison.json").read_text())
    ref, new = comparison["reference_50"], comparison["E_star_run"]
    assert comparison["E_star"] == args.epochs
    assert ref["ladder"]["common_rows"] == new["ladder"]["common_rows"]
    assert new["ladder"]["dtheta_cosine"]["n"] == 4, new["ladder"]["dtheta_cosine"]
    assert new["ratios"]["available"] and new["ratios"]["n"] == 4
    assert new["damping"]["available"] and set(new["damping"]["by_gamma"]) == {
        "full_gamma0.01", "full_gamma1", "dose01_gamma0.01", "dose01_gamma1"}
    assert new["damping"]["h_norm"]["n"] > 0 and len(new["damping"]["spectrum"]) == 2
    assert (step2 / "compare_vs_E50/compare_50_vs_Estar.png").is_file()
    print(printed)
    print("A STEP2 SYNTHETIC CHAIN PASSED", flush=True)


if __name__ == "__main__":
    main()
