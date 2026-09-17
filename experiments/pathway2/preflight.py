#!/usr/bin/env python3
"""Read-only protocol inspection and frozen-file checks; submits no GPU jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def output_is_separate(output: Path, baseline: Path, code: Path) -> None:
    for protected in (baseline, code):
        if overlaps(output, protected):
            raise ValueError(f"Output overlaps protected directory: {protected}")


def validate_hashes(root: Path, expected: dict[str, str]) -> list[dict]:
    failures = []
    for relative, digest in expected.items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError(f"Invalid relative input path: {relative}")
        path = root / part
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif sha256(path) != digest:
            failures.append({"path": relative, "reason": "sha256_mismatch"})
    return failures


def verify(baseline: Path, output: Path, contract: dict, plan: dict) -> dict:
    baseline = baseline.resolve(strict=True)
    output = output.resolve()
    output_is_separate(output, baseline, SOURCE_ROOT)
    if output.name != plan["role"]:
        raise ValueError(f"Output root must end in /{plan['role']} to separate A and B")
    for relative in contract["baseline_inputs_sha256"]:
        original = baseline / relative
        if original.exists():
            if overlaps(output, original.resolve().parent):
                raise ValueError(f"Output overlaps a resolved input directory: {relative}")
    head = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "branch", "--show-current"], text=True
    ).strip()
    if branch != plan["branch"]:
        raise ValueError(f"Wrong worktree branch: expected {plan['branch']}, got {branch}")
    ancestor = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "merge-base", "--is-ancestor",
         contract["base_commit"], head], check=False
    )
    if ancestor.returncode != 0:
        raise ValueError("Current commit does not descend from the frozen source commit")
    branch_failures = validate_hashes(SOURCE_ROOT, contract["source_sha256"])
    original_source_failures = validate_hashes(baseline, contract["source_sha256"])
    input_failures = validate_hashes(baseline, contract["baseline_inputs_sha256"])
    passed = not (branch_failures or original_source_failures or input_failures)
    return {
        "status": "PASS_FILE_INTEGRITY_ONLY" if passed else "FAIL",
        "branch": branch,
        "commit": head,
        "frozen_base_commit": contract["base_commit"],
        "baseline_root": str(baseline),
        "proposed_output_root": str(output),
        "contract_sha256": sha256(HERE / "frozen_baseline.json"),
        "plan_sha256": sha256(HERE / "plan.json"),
        "source_files_per_root": len(contract["source_sha256"]),
        "baseline_files": len(contract["baseline_inputs_sha256"]),
        "branch_source_failures": branch_failures,
        "original_source_failures": original_source_failures,
        "baseline_input_failures": input_failures,
        "dataset_bytes_checked": False,
        "new_runner_replay_checked": False,
        "gpu_jobs_submitted": 0,
        "note": "Read-only verification. Record and verify dataset/cache bytes on Delta; "
                "new runner replay/derivative gates are separate. Repeat checks after each job.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Print the planned, not-yet-implemented experiment matrix")
    check = sub.add_parser("verify", help="Check original files without modifying them")
    check.add_argument("--baseline-root", type=Path, required=True)
    check.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((HERE / "frozen_baseline.json").read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        report = verify(args.baseline_root, args.output_root, contract, plan)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS_FILE_INTEGRITY_ONLY" else 2


if __name__ == "__main__":
    sys.exit(main())
