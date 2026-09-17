#!/usr/bin/env python3
"""One explicit calibration phase per invocation; suitable for Slurm.

Defaults: 4 DME + 4 DRUSEN patients, one eligible shadow, 5 Stage-1 seeds x
5 shared attack seeds. Run preflight -> baseline -> truth -> score.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Dose ladder: the condition name is the directory suffix, the value is --deletion_weight.
# dose01/025/05 are the original v4.1 conditions and keep their exact meaning; the small
# doses below were added for the Pathway2 B0 local-derivative probe and change nothing else.
DOSE_CONDITIONS = {
    "dose05": 0.5, "dose025": 0.25, "dose01": 0.1,
    "dose003": 0.03, "dose001": 0.01, "dose0003": 0.003, "dose0001": 0.001,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["tests", "smoke", "preflight", "baseline", "truth", "score", "stats"], required=True)
    ap.add_argument("--root", default="results/cross_stage_calibration_v4_1")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--condition", choices=list(DOSE_CONDITIONS) + ["full", "early", "late"], default="full")
    ap.add_argument("--patients", type=int, nargs="*", default=None,
                    help=("restrict the LOO loop to these patient ids from the frozen panel "
                          "(B0 small-dose cells); the panel itself is unchanged, and baseline/"
                          "no-op still cover the whole split"))
    ap.add_argument("--stage1_seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--attack_seeds", type=int, nargs="+", default=[5101, 5102, 5103, 5104, 5105])
    ap.add_argument("--n_total_samples", type=int, default=40000)
    ap.add_argument("--target_data_size", type=int, default=2000)
    ap.add_argument("--shadow_data_size", type=int, default=2000)
    ap.add_argument("--n_shadow", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=42005)
    ap.add_argument("--selection_seed", type=int, default=42006)
    ap.add_argument("--n_affected_shadows", type=int, default=1)
    ap.add_argument("--panel_index", type=int, default=0, help="index into preflight's proposed_affected_shadows")
    ap.add_argument("--patients_per_class", type=int, default=4)
    ap.add_argument("--min_patient_images", type=int, default=5)
    ap.add_argument("--max_patient_images", type=int, default=15)
    ap.add_argument("--shadow_epochs", type=int, default=50)
    ap.add_argument("--attack_epochs", type=int, default=50)
    ap.add_argument("--damping_shadow", type=float, default=.01)
    ap.add_argument("--damping_attack", type=float, default=.001)
    ap.add_argument("--damping_shadow_grid", type=float, nargs="*", default=[.003, .03, .1])
    ap.add_argument("--cg_iters", type=int, default=100)
    ap.add_argument("--hvp_batch", type=int, default=64)
    ap.add_argument("--lanczos_iters", type=int, default=30)
    ap.add_argument("--require_cuda", action="store_true",
                    help="fail if CUDA is unavailable; the Delta GPU launcher sets this")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.require_cuda and not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Delta GPU job has no usable CUDA device; check the loaded PyTorch module")
        print(f"CUDA required: torch={torch.__version__}, CUDA={torch.version.cuda}, "
              f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    root = Path(args.root).resolve(); data = Path(args.data_dir).resolve()
    commands = []

    def run(name, opts=()):
        cmd = [sys.executable, str(HERE/name), *map(str, opts)]
        print(shlex.join(cmd), flush=True)
        commands.append(cmd)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO, check=True)

    if args.phase == "tests":
        run("tests/test_score_ladder_math.py")
        run("tests/test_cuda_reproducibility.py")
        run("tests/test_formal_analysis.py")
        return
    if args.phase == "smoke":
        run("tests/run_synthetic_chain.py", ["--output_dir", root/"synthetic_smoke"])
        return
    for seeds in (args.stage1_seeds, args.attack_seeds):
        if len(seeds) != len(set(seeds)):
            raise ValueError("Seed panels must contain unique values")
    if args.shadow_epochs < 2 or args.patients_per_class < 1:
        raise ValueError("Need >=2 epochs and >=1 patient per class")
    if not 1 <= args.n_affected_shadows <= args.n_shadow:
        raise ValueError("n_affected_shadows must be between 1 and n_shadow")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        import numpy, scipy, sklearn, torch
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata = {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                    "job_id": os.environ.get("SLURM_JOB_ID"), "args": vars(args),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())}
        log_dir = root/"invocations"; log_dir.mkdir(exist_ok=True)
        (log_dir/f"{stamp}_{args.phase}.json").write_text(json.dumps(metadata, indent=2))
    split = ["--data_dir", data, "--n_total_samples", args.n_total_samples,
             "--target_data_size", args.target_data_size, "--shadow_data_size", args.shadow_data_size,
             "--n_shadow", args.n_shadow, "--split_seed", args.split_seed]
    selection = ["--classes", 1, 2, "--min_patient_images", args.min_patient_images,
                 "--max_patient_images", args.max_patient_images, "--selection_seed", args.selection_seed]
    panel = root/"panel"/"eligibility_preflight.json"
    if args.phase == "preflight":
        if panel.exists() and not args.dry_run:
            print(f"Frozen panel already exists: {panel}. Use a new --root for a new selection.")
            return
        run("08_eligibility_preflight.py", [*split, *selection, "--output_dir", root/"panel",
            "--patients_per_class_per_shadow", args.patients_per_class,
            "--n_affected_shadows", args.n_affected_shadows])
        return
    if not panel.is_file():
        raise RuntimeError(f"Run --phase preflight first; missing {panel}")
    proposal = json.loads(panel.read_text())
    if not proposal.get("panel_complete"):
        raise RuntimeError(f"Panel incomplete: {proposal.get('shortfall')}. Review eligibility counts before changing the design.")
    if not 0 <= args.panel_index < len(proposal["proposed_affected_shadows"]):
        raise ValueError("panel_index outside proposed shadow list")
    if proposal["args"]["patients_per_class_per_shadow"] != args.patients_per_class:
        raise ValueError("Patient quota differs from frozen panel; pass the original parameters")
    shadow = proposal["proposed_affected_shadows"][args.panel_index]
    out = root/f"shadow{shadow}_{args.condition}"
    common = [*split, *selection, "--output_dir", out, "--affected_shadow", shadow,
              "--n_patients", 2*args.patients_per_class, "--patient_panel_json", panel,
              "--require_complete_panel", "--require_all_classes", "--enforce_attack_gate",
              "--enforce_noop_gate", "--noop_replays", 1, "--deletion_mode", "fixed_mask",
              "--window_membership", "value_only", "--shadow_epochs", args.shadow_epochs,
              "--attack_epochs", args.attack_epochs, "--seeds", *args.stage1_seeds,
              "--attack_seeds", *args.attack_seeds, "--save_epoch_checkpoints", args.shadow_epochs//2, args.shadow_epochs]
    if args.condition in DOSE_CONDITIONS:
        common += ["--deletion_weight", DOSE_CONDITIONS[args.condition]]
    elif args.condition in ("early", "late"):
        mid = args.shadow_epochs//2
        common += ["--removal_epochs", f"0:{mid}" if args.condition == "early" else f"{mid}:{args.shadow_epochs}"]
    if args.patients:
        common += ["--loo_patients", *args.patients]
    if args.phase in ("baseline", "truth"):
        run("05_end_to_end_patient_loo_pilot.py", common + (["--baseline_only"] if args.phase == "baseline" else []))
    if args.phase == "score":
        summary_path = out/"experiment_summary.json"
        if not summary_path.is_file() or json.loads(summary_path.read_text()).get("status") != "complete":
            raise RuntimeError("Score phase requires a completed truth run")
        score_dir = out/f"score_ladder_A{args.damping_attack:g}_S{args.damping_shadow:g}"
        run("07_cross_stage_score_ladder.py", ["--pilot_dir", out, "--data_dir", data, "--out_dir", score_dir,
            "--damping_shadow", args.damping_shadow, "--damping_attack", args.damping_attack,
            "--damping_shadow_grid", *args.damping_shadow_grid, "--cg_iters", args.cg_iters,
            "--hvp_batch", args.hvp_batch, "--lanczos_iters", args.lanczos_iters,
            "--cg_tol", 1e-4, "--cg_fail_tol", 1e-3])
    if args.phase in ("truth", "score", "stats"):
        run("09_seed_variance.py", ["--pilot_dirs", out, "--out_dir", out/"seed_stats"])
    if not args.dry_run:
        (log_dir/f"{stamp}_{args.phase}_commands.json").write_text(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
