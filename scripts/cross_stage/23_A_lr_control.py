#!/usr/bin/env python3
"""Manual prepare/test/run/status gates for the baseline-only LR probe.

Standard library only. Never submits a downstream job automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def git(*args):
    return command(["git", "-C", str(REPO), *args])


def preflight(cfg):
    return json.loads(command([
        sys.executable, str(REPO / "experiments/pathway2/preflight.py"), "verify",
        "--baseline-root", cfg["baseline_root"], "--output-root", cfg["runs_root"]]))


def prepare(note):
    if not note.strip():
        raise ValueError("Record the actual authorization for the decay probe; do not invent advisor approval")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked files are modified")
    cfg = {"schema": "pathway2_A_lr_submission_v2", "repo": str(REPO),
           "git_commit": git("rev-parse", "HEAD"), "approval_note": note,
           "seeds": SEEDS, "resume_epoch": 100, "extra_epochs": 40,
           "decay_schedule": "20@1e-4,20@1e-5", "account": os.environ.get("OCT_ACCOUNT", "bgjy-delta-gpu")}
    for field, env in (("baseline_root", "OCT_BASELINE_ROOT"), ("runs_root", "OCT_RUNS_ROOT"),
                       ("data_dir", "OCT_DATA_DIR"), ("step1_root", "OCT_A_STEP1_ROOT")):
        cfg[field] = str(Path(os.environ[env]).expanduser().resolve())
    proof = preflight(cfg)
    if proof.get("status") != "PASS_FILE_INTEGRITY_ONLY":
        raise ValueError("Route A preflight failed")
    full = Path(cfg["baseline_root"]) / "results/cross_stage_calibration_v4_1/shadow3_full"
    args = read(full / "experiment_config.json")["args"]
    if args["seeds"] != SEEDS or args["affected_shadow"] != 3:
        raise ValueError("This submission panel requires affected shadow3 and seeds 42..46 in that order")
    inputs = [full / "experiment_config.json", full / "splits/fresh_patient_split.json"]
    for seed in SEEDS:
        run = Path(cfg["step1_root"]) / f"shadow3_seed{seed}"
        manifest = read(run / "manifest.json")
        if (manifest.get("status") != "complete" or manifest.get("replay_exact") is not True
                or manifest.get("seed") != seed or manifest.get("extended_epochs", 0) < 100):
            raise ValueError(f"Incomplete/non-exact/wrong Step-1 run: {run}")
        inputs += [run / "manifest.json", run / "checkpoints/epoch100.pt",
                   run / "input_sha256.json", run / "dataset_fingerprint.json"]
    cfg["input_sha256"] = {str(p): sha(p) for p in inputs}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(cfg["runs_root"]) / f"lr_probe_R100_X40_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    cfg["root"] = str(root)
    write_new(root / "probe_config.json", cfg)
    write_new(root / "preflight.json", proof)
    print(root)
    return root


def load_config(root):
    root = Path(root).resolve()
    cfg = read(root / "probe_config.json")
    if cfg["root"] != str(root) or cfg["repo"] != str(REPO):
        raise ValueError("Use the immutable attempt in the independent clone that prepared it")
    if git("rev-parse", "HEAD") != cfg["git_commit"] or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Prepared source commit changed or tracked files are modified")
    for path, digest in cfg["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Prepared input changed: {path}")
    return root, cfg


def job_success(job_id):
    output = command(["sacct", "-X", "--noheader", "--parsable2", "-j", str(job_id),
                      "--format=JobIDRaw,State,ExitCode"])
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    return any(len(r) >= 3 and r[0].strip() == str(job_id)
               and r[1].strip() == "COMPLETED" and r[2].strip() == "0:0" for r in rows)


def submit(root, action, max_parallel=2):
    root, cfg = load_config(root)
    if not 1 <= max_parallel <= 4:
        raise ValueError("max_parallel must be 1..4")
    if (root / f"{action}_submission.json").exists() or (root / f".{action}_submission.lock").exists():
        raise ValueError(f"{action} already submitted or a submission attempt needs inspection; no duplicate submitted")
    if action == "run":
        test = read(root / "test_submission.json")
        marker = read(root / "gpu_tests_passed.json")
        if (marker.get("job_id") != test["job_id"] or marker.get("git_commit") != cfg["git_commit"]
                or marker.get("config_sha256") != sha(root / "probe_config.json")):
            raise ValueError("GPU test marker does not match this attempt")
        if not job_success(test["job_id"]):
            raise ValueError("GPU test must be COMPLETED 0:0 in sacct before array submission")
    if preflight(cfg).get("status") != "PASS_FILE_INTEGRITY_ONLY":
        raise ValueError("Live Route A preflight failed")
    env = dict(os.environ)
    env.update(PROJECT_DIR=str(REPO), OCT_A_CODE=cfg["git_commit"], OCT_BASELINE_ROOT=cfg["baseline_root"],
               OCT_RUNS_ROOT=cfg["runs_root"], OCT_DATA_DIR=cfg["data_dir"], OCT_A_STEP1_ROOT=cfg["step1_root"],
               OCT_A_LR_OUT=str(root), OCT_A_RESUME=str(cfg["resume_epoch"]), OCT_A_EXTRA=str(cfg["extra_epochs"]),
               OCT_A_DECAY=cfg["decay_schedule"])
    (REPO / "logs").mkdir(exist_ok=True)
    args = ["sbatch", "--parsable", f"--account={cfg['account']}"]
    if action == "run":
        args += [f"--array=0-9%{max_parallel}"]
    args += ["scripts/cross_stage/23_A_lr_tests.slurm" if action == "test"
             else "scripts/cross_stage/23_A_lr_probe.slurm"]
    # Kept even after an error: a lost sbatch reply is not permission to submit twice.
    write_new(root / f".{action}_submission.lock", {"action": action, "command": args})
    job_id = command(args, cwd=REPO, env=env).split(";")[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch response {job_id!r}; inspect queue before retrying")
    write_new(root / f"{action}_submission.json", {"job_id": job_id, "git_commit": cfg["git_commit"],
              "config_sha256": sha(root / "probe_config.json"), "max_parallel": max_parallel if action == "run" else 1})
    print(f"{action} submitted as job {job_id}\nOUT={root}\nNo downstream job was submitted.")
    return job_id


def mark_test(root):
    root, cfg = load_config(root)
    test = read(root / "test_submission.json")
    if os.environ.get("SLURM_JOB_ID") != test["job_id"]:
        raise ValueError("GPU test completion must come from this attempt's recorded Slurm job")
    write_new(root / "gpu_tests_passed.json", {"job_id": test["job_id"], "git_commit": cfg["git_commit"],
              "config_sha256": sha(root / "probe_config.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("note")
    for action in ("test", "run", "status", "mark-test", "check"):
        p = sub.add_parser(action)
        p.add_argument("root", type=Path)
        if action == "run":
            p.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.note)
    elif args.action in ("test", "run"):
        submit(args.root, args.action, getattr(args, "max_parallel", 2))
    elif args.action == "mark-test":
        mark_test(args.root)
    elif args.action == "check":
        root, cfg = load_config(args.root)
        print(json.dumps({"checked_root": str(root), "git_commit": cfg["git_commit"]}))
    else:
        cfg = read(args.root / "probe_config.json")
        print(json.dumps({"root": cfg["root"], "git_commit": cfg["git_commit"],
                         "gpu_tests_passed": (args.root / "gpu_tests_passed.json").is_file()}, indent=2))
        ids = [read(p)["job_id"] for p in (args.root / "test_submission.json", args.root / "run_submission.json")
               if p.exists()]
        if ids:
            subprocess.run(["sacct", "-X", "-j", ",".join(ids),
                            "--format=JobID%18,JobName%16,State,ExitCode,Elapsed,NodeList"], check=True)


if __name__ == "__main__":
    main()
