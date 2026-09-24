#!/usr/bin/env python3
"""Immutable prepare/test/run/mc/status/report gates; no automatic downstream jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import route_a_smallpanel as common

REPO = common.REPO
read, write_new, sha, load_config = common.read, common.write_new, common.sha, common.load_config


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def preflight(cfg):
    proof = json.loads(command([sys.executable, str(REPO / "experiments/pathway2/preflight.py"),
                      "verify", "--baseline-root", cfg["baseline_root"], "--output-root", cfg["runs_root"]]))
    if proof.get("status") != "PASS_FILE_INTEGRITY_ONLY":
        raise ValueError("Original Route A integrity preflight failed")
    return proof


def prepare(note):
    if not note.strip() or "<" in note or ">" in note:
        raise ValueError("Record actual authorization, not a placeholder or invented advisor approval")
    if common.git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Commit source changes before preparing")
    cfg = dict(schema="pathway2_A_smallpanel_config_v3", repo=str(REPO),
               git_commit=common.git("rev-parse", "HEAD"), approval_note=note, cells=common.expected_cells(),
               resume_epoch=100, total_epochs=140, decay_schedule=common.SCHEDULE,
               account=os.environ.get("OCT_ACCOUNT", "bgjy-delta-gpu"))
    for k, env in (("baseline_root", "OCT_BASELINE_ROOT"), ("runs_root", "OCT_RUNS_ROOT"),
                   ("data_dir", "OCT_DATA_DIR"), ("e100_root", "OCT_A_E100_ROOT"),
                   ("probe_root", "OCT_A_PROBE_ROOT")):
        cfg[k] = str(Path(os.environ[env]).expanduser().resolve())
    cfg["full_dir"] = str(Path(cfg["baseline_root"]) / "results/cross_stage_calibration_v4_1/shadow3_full")
    proof = preflight(cfg)
    inputs, dataset, facts = common.collect_inputs(cfg)
    cfg.update(input_sha256=inputs, dataset_fingerprint=dataset, probe_facts=facts)
    root = Path(cfg["runs_root"]) / ("smallpanel_loo_R100_X40_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    common.disjoint(root, REPO, cfg["baseline_root"], cfg["e100_root"], cfg["probe_root"], cfg["data_dir"])
    root.mkdir(parents=True, exist_ok=False)
    cfg["root"] = str(root)
    write_new(root / common.CONFIG, cfg)
    (root / 'config.sha256').write_text(sha(root / common.CONFIG) + '\n')
    write_new(root / "preflight.json", proof)
    print(root)
    return root


def job_success(job):
    output = command(["sacct", "-X", "--noheader", "--parsable2", "-j", str(job),
                      "--format=JobIDRaw,State,ExitCode"])
    return any(len(r) >= 3 and r[0].strip() == str(job) and r[1].strip() == "COMPLETED" and r[2].strip() == "0:0"
               for r in (line.split("|") for line in output.splitlines()))


def smoke_checks(root, cfg):
    paths = []
    for pid in common.PATIENTS:
        for dose in common.DOSES:
            d = root / "smoke" / common.cell_name(42, pid, dose, "control")
            m = read(d / "manifest.json")
            if (m.get("status") != "smoke_complete" or m.get("replay_e100_passed") is not True or
                    m.get("config_sha256") != sha(root / common.CONFIG) or m.get("git_commit") != cfg["git_commit"]):
                raise ValueError("Real E100 smoke manifest missing or mismatched")
            checks = read(d / "replay_checks.json")
            if len(checks) != 1 or not all(checks[0].get(k) is True for k in
                ("passed", "parameters_equal", "rng_fingerprint_equal", "optimizer_hash_equal", "recorded_orders_equal")):
                raise ValueError("Real E100 model/Adam/RNG/order replay failed")
            paths += [d / "manifest.json", d / "replay_checks.json"]
    return {str(p): sha(p) for p in paths}


def require_test(root, cfg):
    sub = read(root / "test_submission.json")
    marker = read(root / "gpu_tests_passed.json")
    if (marker.get("job_id") != sub["job_id"] or marker.get("git_commit") != cfg["git_commit"] or
            marker.get("config_sha256") != sha(root / common.CONFIG) or
            marker.get("smoke_sha256") != smoke_checks(root, cfg)):
        raise ValueError("GPU/real-data smoke marker does not match this attempt")
    if not job_success(sub["job_id"]):
        raise ValueError("Test must be COMPLETED 0:0 before submission")


def submit(root, action, max_parallel=2):
    root, cfg = load_config(root)
    if not 1 <= max_parallel <= 4:
        raise ValueError("max_parallel must be 1..4; also respect the total GPU budget across A/B/C")
    lock = root / f".{action}_submission.lock"
    if lock.exists() or (root / f"{action}_submission.json").exists():
        raise ValueError(f"{action} already submitted or attempted; inspect Slurm, no duplicate submission")
    if action in ("run", "mc"):
        require_test(root, cfg)
    preflight(cfg)
    env = dict(os.environ, PROJECT_DIR=str(REPO), OCT_A_CODE=cfg["git_commit"], OCT_A_SP_OUT=str(root))
    (REPO / "logs").mkdir(exist_ok=True)
    cmd = ["sbatch", "--parsable", f"--account={cfg['account']}", "--export=ALL"]
    if action == "run": cmd += [f"--array=0-39%{max_parallel}"]
    scripts = dict(test="24_A_smallpanel_tests.slurm", run="24_A_smallpanel.slurm", mc="25_A_endpoint_mc.slurm")
    cmd += ["scripts/cross_stage/" + scripts[action]]
    write_new(lock, dict(command=cmd, config_sha256=sha(root / common.CONFIG)))
    job = command(cmd, cwd=REPO, env=env).split(";")[0]
    if not job.isdigit(): raise ValueError("Unexpected sbatch response; inspect queue before recovery")
    write_new(root / f"{action}_submission.json", dict(job_id=job, git_commit=cfg["git_commit"],
              config_sha256=sha(root / common.CONFIG), max_parallel=max_parallel if action == "run" else 1))
    print(f"{action} submitted as job {job}\nOUT={root}\nNo downstream job was submitted.")
    return job


def mark_test(root):
    root, cfg = load_config(root)
    job = read(root / "test_submission.json")["job_id"]
    if os.environ.get("SLURM_JOB_ID") != job:
        raise ValueError("Test marker must be written by its recorded GPU job")
    write_new(root / "gpu_tests_passed.json", dict(job_id=job, git_commit=cfg["git_commit"],
              config_sha256=sha(root / common.CONFIG), smoke_sha256=smoke_checks(root, cfg)))


def execute(root, action, index=None):
    root, cfg = load_config(root)
    if action == "mc-exec":
        subprocess.run([sys.executable, str(REPO / "scripts/cross_stage/25_A_endpoint_mc.py"),
                        "--config", str(root / common.CONFIG), "--out", str(root / "endpoint_mc"),
                        "--mc_reps", "128", "--require_cuda"], check=True)
        return
    indices = (0, 2, 4, 6) if action == "smoke" else (index,)
    for i in indices:
        if i is None or not 0 <= i < 40: raise ValueError("array index must be 0..39")
        cmd = [sys.executable, str(REPO / "scripts/cross_stage/24_A_smallpanel_loo.py"),
               "--config", str(root / common.CONFIG), "--full_dir", cfg["full_dir"],
               "--e100_full_dir", str(Path(cfg["e100_root"]) / "shadow3_full"),
               "--e100_dose_dir", str(Path(cfg["e100_root"]) / "shadow3_dose01"),
               "--data_dir", cfg["data_dir"], "--out_root", str(root / "smoke" if action == "smoke" else root),
               "--array_index", str(i), "--require_cuda"]
        if action == "smoke": cmd += ["--stop_at_resume"]
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("note")
    for action in ("test", "run", "mc", "status", "check", "mark-test", "smoke", "cell", "mc-exec", "report"):
        p = sub.add_parser(action); p.add_argument("root", type=Path)
        if action == "run": p.add_argument("--max-parallel", type=int, default=2)
        if action == "cell": p.add_argument("index", type=int)
    a = ap.parse_args()
    if a.action == "prepare": prepare(a.note)
    elif a.action in ("test", "run", "mc"): submit(a.root, a.action, getattr(a, "max_parallel", 2))
    elif a.action == "mark-test": mark_test(a.root)
    elif a.action in ("smoke", "cell", "mc-exec"): execute(a.root, a.action, getattr(a, "index", None))
    elif a.action == "check": load_config(a.root); print("Prepared inputs and source verified")
    elif a.action == "report":
        load_config(a.root)
        subprocess.run([sys.executable, str(REPO / "scripts/cross_stage/24_A_smallpanel_report.py"),
                        "--root", str(a.root)], check=True)
    else:
        cfg = read(a.root / common.CONFIG)
        print(json.dumps(dict(root=str(a.root), git_commit=cfg["git_commit"],
              completed=sum(read(p).get("status") == "complete" for p in a.root.glob("seed*/manifest.json")),
              expected_cells=40, gpu_test_marker=(a.root / "gpu_tests_passed.json").exists()), indent=2))
        ids = [read(p)["job_id"] for p in a.root.glob("*_submission.json")]
        if ids: subprocess.run(["sacct", "-X", "-j", ",".join(ids), "--format=JobID%18,JobName%16,State,ExitCode,Elapsed,NodeList"], check=True)


if __name__ == "__main__": main()
