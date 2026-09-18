#!/usr/bin/env python3
"""Generate the frozen 10/05/09 truth command and run it with Route-A replay checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--condition", choices=["full", "dose01"], required=True)
    ap.add_argument("--shadow_epochs", type=int, required=True)
    ap.add_argument("--original_epochs", type=int, default=50)
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args(argv)
    if args.shadow_epochs <= args.original_epochs:
        raise SystemExit("E* must extend the original endpoint")
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Route-A truth job has no usable CUDA device")

    dry = [sys.executable, str(HERE / "10_run_calibration.py"), "--phase", "truth",
           "--root", str(Path(args.root).resolve()), "--data_dir", str(Path(args.data_dir).resolve()),
           "--shadow_epochs", str(args.shadow_epochs), "--condition", args.condition, "--dry_run"]
    printed = subprocess.run(dry, check=True, text=True, capture_output=True).stdout
    commands = [shlex.split(line) for line in printed.splitlines() if line.strip()]
    pilot = next(c for c in commands if len(c) > 1 and c[1].endswith("05_end_to_end_patient_loo_pilot.py"))
    stats = next(c for c in commands if len(c) > 1 and c[1].endswith("09_seed_variance.py"))
    if "--overwrite" in pilot:
        raise RuntimeError("Route A truth must never overwrite an existing run")
    wrapper = [sys.executable, str(HERE / "21_A_truth_replay.py"),
               "--reference_dir", str(Path(args.reference_dir).resolve()),
               "--selection", str(Path(args.selection).resolve()),
               "--original_epochs", str(args.original_epochs), "--", *pilot[2:]]
    print("RUN", shlex.join(wrapper), flush=True)
    subprocess.run(wrapper, cwd=HERE.parents[1], check=True)
    print("RUN", shlex.join(stats), flush=True)
    subprocess.run(stats, cwd=HERE.parents[1], check=True)
    out = Path(args.root).resolve() / f"shadow3_{args.condition}"
    replay = json.loads((out / "route_a_epoch50_replay_checks.json").read_text())
    if replay.get("status") != "complete":
        raise RuntimeError("Truth finished without a complete epoch-50 replay certificate")


if __name__ == "__main__":
    main()
