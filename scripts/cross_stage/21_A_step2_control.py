#!/usr/bin/env python3
"""Prepare and validate the manually gated Route-A Step-2 stages."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def state(root):
    root = Path(root).resolve()
    path = root / "A_STEP2_STATE.json"
    if not path.is_file():
        raise SystemExit(f"Not a prepared Route-A Step-2 root: {root}")
    return root, path, read(path)


def events(root):
    path = Path(root) / "submission_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def file_json_status(path):
    path = Path(path)
    if not path.is_file():
        return "absent"
    try:
        return read(path).get("status", "present")
    except (OSError, ValueError, TypeError):
        return "unreadable"


def run_preflight(baseline, runs):
    result = subprocess.run(
        [sys.executable, str(REPO / "experiments/pathway2/preflight.py"), "verify",
         "--baseline-root", str(Path(baseline).resolve()),
         "--output-root", str(Path(runs).resolve())],
        check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    if payload.get("status") != "PASS_FILE_INTEGRITY_ONLY":
        raise SystemExit(f"Route-A preflight did not pass: {payload}")
    return payload


def complete_truth(root, condition, state_payload=None):
    folder = root / f"shadow3_{condition}"
    summary = folder / "experiment_summary.json"
    replay = folder / "route_a_epoch50_replay_checks.json"
    if not summary.is_file() or not replay.is_file():
        return False
    try:
        replay_payload = read(replay)
        expected = replay_payload.get("expected_counts")
        checks = replay_payload.get("checks", [])
        anchors = replay_payload.get("E_star_anchor_checks", [])
        valid = (
            read(summary).get("status") == "complete" and
            replay_payload.get("schema") == "pathway2_A_epoch50_replay_v2" and
            replay_payload.get("status") == "complete" and
            isinstance(expected, dict) and expected and
            replay_payload.get("actual_counts") == expected and
            len(checks) == sum(int(v) for v in expected.values()) and
            len(anchors) == int(replay_payload.get("expected_E_star_anchor_checks", -1)) and
            all(row.get("passed") is True and row.get("regenerated_optimizer_sha256") and
                row.get("continued_optimizer_sha256_at_E_star")
                for row in checks) and
            all(row.get("passed") is True for row in anchors)
        )
        if state_payload is not None:
            valid = valid and (
                replay_payload.get("original_epochs") == state_payload["original_epochs"] and
                replay_payload.get("extended_epochs") == state_payload["E_star"] and
                replay_payload.get("selection_sha256") == state_payload["selection_sha256"] and
                expected == {"target": 1, "fixed_shadow": 4,
                             "affected_baseline_or_noop": 10, "loo": 40}
            )
        return valid
    except (KeyError, OSError, TypeError, ValueError):
        return False


def score_complete(root):
    folder = root / "shadow3_full/score_ladder_A0.2_S1"
    required = ("score_config.json", "ladder_rows.csv", "ladder_summary.json")
    if not all((folder / name).is_file() for name in required):
        return False
    try:
        config = read(folder / "score_config.json")
        read(folder / "ladder_summary.json")
        return (float(config.get("damping_attack", -1)) == .2 and
                float(config.get("damping_shadow", -1)) == 1. and
                config.get("damping_shadow_grid") == [] and
                csv_rows(folder / "ladder_rows.csv") > 0)
    except (OSError, TypeError, ValueError):
        return False


def manifest_complete(path, mode):
    try:
        payload = read(path)
        return payload.get("status") == "complete" and payload.get("args", {}).get("mode") == mode
    except (OSError, TypeError, ValueError):
        return False


def primary_compare_complete(root, payload):
    path = root / "compare_primary_vs_E50/comparison.json"
    if not path.is_file():
        return False
    try:
        value = read(path)
        return (value.get("schema") == "pathway2_A_matched_E_comparison_v2" and
                value.get("mode") == "primary" and
                int(value.get("E_star", -1)) == int(payload["E_star"]) and
                value.get("provenance", {}).get("status") == "PASS")
    except (OSError, TypeError, ValueError):
        return False


def gpu_smoke_complete(root, payload):
    path = root / "step2_gpu_smoke/SMOKE_COMPLETE.json"
    if not path.is_file():
        return False
    try:
        value = read(path)
        return (
            value.get("schema") == "pathway2_A_step2_gpu_smoke_v1" and
            value.get("status") == "complete" and
            value.get("git_commit") == payload["git_commit"] and
            value.get("original_epochs") == 2 and
            value.get("extended_epochs") == 3 and
            value.get("actual_counts") == {
                "target": 1, "fixed_shadow": 1,
                "affected_baseline_or_noop": 2, "loo": 2}
        )
    except (OSError, TypeError, ValueError):
        return False


def validate_common(root, payload):
    current = git("rev-parse", "HEAD")
    if current != payload["git_commit"]:
        raise SystemExit(f"Prepared code {payload['git_commit']} differs from current {current}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked files are modified")
    selection = root / "frozen_E_star.json"
    if sha(selection) != payload["selection_sha256"]:
        raise SystemExit("Frozen E* record changed after prepare")
    if str(root) != payload.get("root"):
        raise SystemExit("Prepared root path differs from the immutable state")
    panel_src = Path(payload["baseline_root"]) / "results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json"
    panel_dst = root / "panel/eligibility_preflight.json"
    if not panel_src.is_file() or not panel_dst.is_file() or \
            sha(panel_src) != payload["panel_sha256"] or sha(panel_dst) != payload["panel_sha256"]:
        raise SystemExit("Frozen patient panel changed after prepare")
    stored_preflight = root / "route_a_preflight.json"
    if not stored_preflight.is_file() or sha(stored_preflight) != payload["preflight_sha256"]:
        raise SystemExit("Prepared preflight record changed after prepare")
    live = run_preflight(payload["baseline_root"], payload["runs_root"])
    if (live.get("commit") != payload["git_commit"] or
            Path(live.get("baseline_root", "")).resolve() != Path(payload["baseline_root"]).resolve()):
        raise SystemExit("Live source/baseline preflight differs from the prepared state")
    return selection


def prepare(args):
    selection_path = Path(args.selection).resolve()
    selection = read(selection_path)
    if selection.get("schema") != "pathway2_A_frozen_E_star_v1" or selection.get("status") != "frozen_human_choice":
        raise SystemExit("Step 2 requires a valid human-frozen E* record")
    code = git("rev-parse", "HEAD")
    if selection.get("git_commit") != code:
        raise SystemExit("Selection was produced by a different code commit")
    if selection.get("seeds") != [42, 43, 44, 45, 46] or int(selection.get("original_epochs", -1)) != 50:
        raise SystemExit("Selection is not the frozen five-seed, epoch-50 Route-A panel")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked files are modified")
    for name, digest in {selection["report"]: selection["report_sha256"],
                         **selection["curve_sha256"], **selection["checkpoint_sha256"]}.items():
        path = Path(name)
        if not path.is_file() or sha(path) != digest:
            raise SystemExit(f"Frozen selection input changed or disappeared: {path}")

    baseline = Path(args.baseline_root).resolve()
    runs = Path(args.runs_root).resolve()
    if runs.name != "A":
        raise SystemExit("Route-A runs root must end in /A")
    token = sha(selection_path)[:8]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = runs / f"l3_at_E{int(selection['E_star'])}_{token}_{stamp}"
    for protected in (baseline, REPO.resolve()):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise SystemExit(f"Step-2 output overlaps protected path {protected}")
    if root.exists():
        raise SystemExit(f"Step-2 root already exists: {root}")
    panel_src = baseline / "results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json"
    if not panel_src.is_file():
        raise SystemExit(f"Missing frozen patient panel: {panel_src}")
    preflight_payload = run_preflight(baseline, runs)
    if preflight_payload.get("commit") != code:
        raise SystemExit("Route-A preflight commit differs from the selected code")

    root.mkdir(parents=True)
    copied = root / "frozen_E_star.json"
    shutil.copy2(selection_path, copied)
    panel_dst = root / "panel/eligibility_preflight.json"
    panel_dst.parent.mkdir(parents=True)
    shutil.copy2(panel_src, panel_dst)
    if sha(panel_src) != sha(panel_dst):
        raise SystemExit("Frozen panel copy mismatch")
    preflight_path = root / "route_a_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "schema": "pathway2_A_step2_state_v2", "status": "prepared_manual_gates",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root), "git_commit": code, "E_star": int(selection["E_star"]),
        "original_epochs": int(selection["original_epochs"]),
        "selection_source": str(selection_path), "selection_sha256": sha(copied),
        "baseline_root": str(baseline), "runs_root": str(runs),
        "panel_sha256": sha(panel_dst),
        "preflight_sha256": sha(preflight_path),
        "automatic_downstream_submission": False,
    }
    write_new(root / "A_STEP2_STATE.json", payload)
    print(root)


def check(args):
    root, _, payload = state(args.root)
    validate_common(root, payload)
    action = args.action
    submitted = {event["action"] for event in events(root)}
    if (not getattr(args, "job_execution", False) and
            action in {"test", "truth-full", "truth-dose", "score", "preflight", "damping"} and
            action in submitted):
        raise SystemExit(f"{action} was already submitted for this immutable attempt")
    if action == "test" and (root / "step2_gpu_smoke").exists():
        raise SystemExit("Step-2 GPU smoke output already exists")
    if action in ("truth-full", "truth-dose") and not gpu_smoke_complete(root, payload):
        raise SystemExit(f"{action} requires the completed Step-2 GPU replay smoke test")
    if action == "truth-full" and (root / "shadow3_full").exists():
        raise SystemExit("full truth output already exists")
    if action == "truth-dose" and (root / "shadow3_dose01").exists():
        raise SystemExit("dose01 truth output already exists")
    if action == "truth-dose" and not primary_compare_complete(root, payload):
        raise SystemExit("truth-dose is intentionally gated until the A1 primary comparison is complete")
    if action == "score":
        if not complete_truth(root, "full", payload):
            raise SystemExit("score requires completed full truth and replay certificate")
        if (root / "shadow3_full/score_ladder_A0.2_S1").exists():
            raise SystemExit("score output already exists")
    if action in ("compare-primary", "approve-a2"):
        if not complete_truth(root, "full", payload) or not score_complete(root):
            raise SystemExit("primary comparison requires the completed original-damping score")
    if action == "compare-primary" and (root / "compare_primary_vs_E50").exists():
        raise SystemExit("primary comparison output already exists")
    if action in ("preflight", "approve-a2", "damping", "compare-full"):
        if not all(complete_truth(root, c, payload) for c in ("full", "dose01")):
            raise SystemExit(f"{action} requires completed full and dose01 truth with replay certificates")
    if action == "preflight" and (root / "diag_preflight").exists():
        raise SystemExit("diag_preflight already exists")
    if action in ("approve-a2", "damping", "compare-full"):
        manifest = root / "diag_preflight/manifest.json"
        if not manifest_complete(manifest, "preflight"):
            raise SystemExit(f"{action} requires completed truth preflight")
    if action == "approve-a2" and (root / "A2_DAMPING_APPROVAL.json").exists():
        raise SystemExit("A2 approval already exists")
    if action in ("damping", "compare-full"):
        approval = root / "A2_DAMPING_APPROVAL.json"
        if not approval.is_file():
            raise SystemExit("A2 damping requires an explicit post-A1 approval record")
        a = read(approval)
        for name, digest in a["reviewed_sha256"].items():
            if not Path(name).is_file() or sha(name) != digest:
                raise SystemExit(f"Reviewed A1 artifact changed: {name}")
    if action == "damping" and (root / "diag_damping").exists():
        raise SystemExit("diag_damping already exists")
    if action == "compare-full":
        manifest = root / "diag_damping/manifest.json"
        if not manifest_complete(manifest, "damping"):
            raise SystemExit("full comparison requires completed A2 damping")
        if (root / "compare_full_vs_E50").exists():
            raise SystemExit("full comparison output already exists")
    if not getattr(args, "quiet", False):
        print(json.dumps(payload))


def job_check(args):
    args.job_execution = True
    args.quiet = True
    check(args)


def approve(args):
    root, _, payload = state(args.root)
    check(argparse.Namespace(root=str(root), action="approve-a2", quiet=True))
    files = [root / "compare_primary_vs_E50/comparison.json",
             root / "compare_primary_vs_E50/COMPARE.md",
             root / "diag_preflight/manifest.json",
             root / "diag_preflight/checkpoint_ratios.csv"]
    if not all(path.is_file() for path in files):
        raise SystemExit("Review primary comparison and truth preflight before approving A2")
    record = {
        "schema": "pathway2_A2_damping_approval_v1", "status": "approved_after_A1_review",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "note": args.note,
        "git_commit": payload["git_commit"], "E_star": payload["E_star"],
        "reviewed_sha256": {str(path): sha(path) for path in files},
    }
    write_new(root / "A2_DAMPING_APPROVAL.json", record)
    print(root / "A2_DAMPING_APPROVAL.json")


def field(args):
    root, _, payload = state(args.root)
    value = payload.get(args.name)
    if value is None:
        raise SystemExit(f"Unknown state field {args.name}")
    print(value)


def record(args):
    root, _, payload = state(args.root)
    if git("rev-parse", "HEAD") != payload["git_commit"]:
        raise SystemExit("Code commit changed between validation and Slurm submission")
    path = root / "submission_events.jsonl"
    if any(event.get("action") == args.action for event in events(root)):
        raise SystemExit(f"Submission event already recorded for {args.action}")
    if not args.job_id.isdigit():
        raise SystemExit("Slurm job id must contain digits only")
    event = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "action": args.action, "job_id": args.job_id, "git_commit": payload["git_commit"]}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def status(args):
    root, _, payload = state(args.root)
    full = root / "shadow3_full"
    dose = root / "shadow3_dose01"
    score = full / "score_ladder_A0.2_S1"
    preflight = root / "diag_preflight"
    damping = root / "diag_damping"
    result = {
        "root": str(root), "E_star": payload["E_star"], "git_commit": payload["git_commit"],
        "jobs": events(root),
        "stages": {
            "A0_step2_gpu_smoke": file_json_status(root / "step2_gpu_smoke/SMOKE_COMPLETE.json"),
            "A1_truth_full": {
                "summary": file_json_status(full / "experiment_summary.json"),
                "replay": file_json_status(full / "route_a_epoch50_replay_checks.json"),
                "patient_seed_results_written": len(list((full / "runs").glob("seed*_patient*.json"))) if (full / "runs").is_dir() else 0,
            },
            "A1_score_gamma1": {
                "status": "complete" if score_complete(root) else ("started" if score.exists() else "absent"),
                "ladder_rows_written": csv_rows(score / "ladder_rows.csv"),
            },
            "A1_primary_comparison": (
                "complete" if primary_compare_complete(root, payload) else
                ("invalid_or_partial" if (root / "compare_primary_vs_E50").exists() else "absent")
            ),
            "A1_truth_dose01": {
                "summary": file_json_status(dose / "experiment_summary.json"),
                "replay": file_json_status(dose / "route_a_epoch50_replay_checks.json"),
                "patient_seed_results_written": len(list((dose / "runs").glob("seed*_patient*.json"))) if (dose / "runs").is_dir() else 0,
            },
            "A1_truth_preflight": file_json_status(preflight / "manifest.json"),
            "A2_approval": "present" if (root / "A2_DAMPING_APPROVAL.json").is_file() else "absent",
            "A2_damping": file_json_status(damping / "manifest.json"),
            "A2_full_comparison": "complete" if (root / "compare_full_vs_E50/comparison.json").is_file() else "absent",
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", required=True); p.add_argument("--baseline_root", required=True); p.add_argument("--runs_root", required=True)
    p = sub.add_parser("check"); p.add_argument("--root", required=True); p.add_argument("--action", required=True,
        choices=["test", "truth-full", "truth-dose", "score", "compare-primary", "preflight", "approve-a2", "damping", "compare-full"])
    p = sub.add_parser("job-check"); p.add_argument("--root", required=True); p.add_argument("--action", required=True,
        choices=["test", "truth-full", "truth-dose", "score", "preflight", "damping"])
    p = sub.add_parser("approve-a2"); p.add_argument("--root", required=True); p.add_argument("--note", required=True)
    p = sub.add_parser("field"); p.add_argument("--root", required=True); p.add_argument("--name", required=True)
    p = sub.add_parser("record-job"); p.add_argument("--root", required=True); p.add_argument("--action", required=True); p.add_argument("--job_id", required=True)
    p = sub.add_parser("status"); p.add_argument("--root", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "check": check, "job-check": job_check, "approve-a2": approve,
     "field": field, "record-job": record, "status": status}[args.command](args)


if __name__ == "__main__":
    main()
