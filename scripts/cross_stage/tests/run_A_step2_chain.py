#!/usr/bin/env python3
"""Synthetic GPU smoke test for Route-A epoch-T0 replay and continuation.

The production 05 training path is used on generated 16x16 images.  A tiny
T0=2 reference truth is created, the affected baseline is independently
extended to E*=3, then 21_A_truth_replay.py must recover every reference
target/fixed/baseline/no-op/LOO endpoint at T0 and continue with native Adam
state.  This proves wiring and exactness, not anything about OCT results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
ADAPTER = SCRIPTS / "tests/run_with_small_images.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_data(data):
    data.mkdir(parents=True, exist_ok=True)
    (data / "SYNTHETIC_TEST_DATA.json").write_text(json.dumps({"synthetic": True, "seed": 0}))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    for class_index, class_name in enumerate(("CNV", "DME", "DRUSEN", "NORMAL")):
        folder = data / "OCT2017/train" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for patient in range(40):
            offset, frequency = rng.normal(0, .3), rng.uniform(.8, 1.2)
            for image_index in range(4):
                base = (np.sin((class_index + 1) * frequency * xx / 8 + offset) * .5 +
                        np.cos((class_index + 1) * yy / 11) * .3)
                image = base + rng.normal(0, .8, size=base.shape)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                Image.fromarray(image).save(folder / f"{class_name}-{patient + 1000 * class_index}-{image_index}.jpeg")


def run(command, capture=False):
    command = list(map(str, command))
    print("RUN", shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout if capture else None


def pilot_command(calibration):
    printed = run(calibration, capture=True)
    commands = [shlex.split(line) for line in printed.splitlines() if line.strip()]
    return next(command for command in commands
                if len(command) > 1 and command[1].endswith("05_end_to_end_patient_loo_pilot.py"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=16)
    args = ap.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise SystemExit(f"Smoke-test output must be new: {root}")
    root.mkdir(parents=True)
    data, reference, extended = root / "data", root / "reference", root / "extended"
    make_data(data)
    small = [sys.executable, ADAPTER, "--image_size", args.image_size]
    split = ["--data_dir", data, "--n_total_samples", 640, "--target_data_size", 100,
             "--shadow_data_size", 100, "--n_shadow", 2]
    run(small + [SCRIPTS / "08_eligibility_preflight.py", *split, "--output_dir", reference / "panel",
                 "--classes", 1, 2, "--min_patient_images", 1, "--max_patient_images", 30,
                 "--patients_per_class_per_shadow", 1, "--n_affected_shadows", 2])
    panel = json.loads((reference / "panel/eligibility_preflight.json").read_text())
    affected = int(panel["proposed_affected_shadows"][0])

    common = ["--data_dir", data, "--stage1_seeds", 42, "--attack_seeds", 5101,
              "--patients_per_class", 1, "--n_total_samples", 640, "--target_data_size", 100,
              "--shadow_data_size", 100, "--n_shadow", 2, "--min_patient_images", 1,
              "--max_patient_images", 30, "--dry_run"]
    overrides = ["--gate_min_queries_per_label", 1, "--gate_min_class_auc", 0,
                 "--gate_min_class_balanced_accuracy", 0, "--attack_epochs", 3]
    reference_calibration = [sys.executable, SCRIPTS / "10_run_calibration.py", "--root", reference,
                             "--shadow_epochs", 2, "--phase", "truth", "--condition", "full", *common]
    reference_pilot = pilot_command(reference_calibration)
    run(small + reference_pilot[1:] + overrides)
    reference_full = reference / f"shadow{affected}_full"
    if json.loads((reference_full / "experiment_summary.json").read_text())["status"] != "complete":
        raise RuntimeError("Synthetic reference truth did not complete")

    convergence = root / "runs/A/convergence"
    run(small + [SCRIPTS / "20_A_convergence_curve.py", "--full_dir", reference_full,
                 "--data_dir", data, "--out_root", convergence, "--array_index", 0,
                 "--epochs", 3, "--gradient_every", 1, "--dropout_mc_every", 1,
                 "--dropout_mc_reps", 2, "--hvp_batch", 32, "--eval_batch", 64])
    selected = convergence / f"shadow{affected}_seed42/checkpoints/epoch003.pt"
    selection = root / "frozen_E_star.json"
    selection.write_text(json.dumps({
        "schema": "pathway2_A_frozen_E_star_v1", "status": "frozen_human_choice",
        "E_star": 3, "original_epochs": 2,
        "checkpoint_sha256": {str(selected): sha(selected)},
    }, indent=2) + "\n")

    shutil.copytree(reference / "panel", extended / "panel")
    extended_calibration = [sys.executable, SCRIPTS / "10_run_calibration.py", "--root", extended,
                            "--shadow_epochs", 3, "--phase", "truth", "--condition", "full", *common]
    extended_pilot = pilot_command(extended_calibration)
    wrapper = [SCRIPTS / "21_A_truth_replay.py", "--reference_dir", reference_full,
               "--original_epochs", 2, "--selection", selection, "--", *extended_pilot[2:], *overrides]
    run(small + wrapper)

    new_full = extended / f"shadow{affected}_full"
    replay_path = new_full / "route_a_epoch50_replay_checks.json"
    replay = json.loads(replay_path.read_text())
    expected = {"target": 1, "fixed_shadow": 1, "affected_baseline_or_noop": 2, "loo": 2}
    if replay.get("status") != "complete" or replay.get("actual_counts") != expected:
        raise RuntimeError(f"Synthetic replay panel failed: {replay}")
    if not all(check.get("passed") is True for check in replay["checks"] + replay["E_star_anchor_checks"]):
        raise RuntimeError("Synthetic replay contains a failed endpoint check")
    loo = [check for check in replay["checks"] if check["role"] == "loo"]
    if len(loo) != 2 or not all(check.get("regenerated_optimizer_sha256") and
                                check.get("continued_optimizer_sha256_at_E_star") for check in loo):
        raise RuntimeError("LOO native Adam state was not regenerated, continued and recorded")
    certificate = {
        "schema": "pathway2_A_step2_gpu_smoke_v1", "status": "complete",
        "git_commit": os.environ.get("OCT_A_CODE"),
        "original_epochs": 2, "extended_epochs": 3, "affected_shadow": affected,
        "replay_report": str(replay_path), "replay_report_sha256": sha(replay_path),
        "actual_counts": replay["actual_counts"],
    }
    (root / "SMOKE_COMPLETE.json").write_text(json.dumps(certificate, indent=2) + "\n")
    print(json.dumps(certificate, indent=2), flush=True)


if __name__ == "__main__":
    main()
