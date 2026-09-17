#!/usr/bin/env python3
"""Check completed E0/B0 artifacts and collect a small review archive (no weights)."""
import argparse
import csv
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--include-checkpoints", action="store_true")
    args = ap.parse_args()
    submission = args.submission.resolve(strict=True)
    root = submission.parent
    ids = dict(line.split("=", 1) for line in submission.read_text().splitlines() if "=" in line)
    for key in ("GPU_TEST", "E0", "B0_ARRAY"):
        if not ids[key].isdigit():
            raise ValueError(f"Invalid job id: {key}")
    checks = [(ids["GPU_TEST"], "single"), (ids["E0"], "single"),
              (ids["B0_ARRAY"], "0"), (ids["B0_ARRAY"], "1")]
    for job, task in checks:
        p = root / f"postcheck_{job}_{task}.json"
        if read(p)["status"] != "PASS_FILE_INTEGRITY_ONLY":
            raise RuntimeError(f"Post-job frozen-input check failed: {p}")
    e0 = root / f"e0_{ids['E0']}"
    folders = [e0]
    manifest = read(e0 / "manifest.json")
    if manifest["status"] != "complete" or not manifest["input_files_unchanged"]:
        raise RuntimeError("E0 execution/input integrity failed")
    if manifest["git_commit"] != ids["CODE"]:
        raise RuntimeError("E0 code commit differs from submission")
    loaded_data = manifest["loaded_data_sha256"]
    payload = read(e0 / "e0_rows.json")
    expected = {(seed, pid, alpha) for seed in (42, 43) for pid in (807, 2085) for alpha in (1., .1)}
    actual = {(r["seed"], r["patient_id"], r["alpha"]) for r in payload["rows"]}
    if actual != expected or len(payload["rows"]) != 8 or len(payload["monitor"]) != 16:
        raise RuntimeError("E0 cell/monitor coverage is incomplete")
    import math
    for r in payload["rows"]:
        if r["algebra_closure_rel"] is None or not math.isfinite(r["algebra_closure_rel"]) or r["algebra_closure_rel"] > 1e-6:
            raise RuntimeError("E0 algebra closure gate failed")
    verdicts = {}
    for seed in (42, 43):
        folder = root / f"b0_{ids['B0_ARRAY']}_seed{seed}"
        folders.append(folder)
        if read(folder / "manifest.json")["loaded_data_sha256"] != loaded_data:
            raise RuntimeError("E0/B0 loaded data arrays differ")
        for path in (folder / "manifest.json", folder / "analysis/manifest.json"):
            m = read(path)
            if m["status"] != "complete" or not m["input_files_unchanged"]:
                raise RuntimeError(f"B0 execution/input integrity failed: {path}")
            if m.get("git_commit") != ids["CODE"]:
                raise RuntimeError(f"B0 code commit mismatch: {path}")
        noop = read(folder / "noop_replay.json")
        if not noop["parameters_exact"] or not noop["interface_exact"] or noop["displacement_norm"] != 0:
            raise RuntimeError("B0 replay gate failed")
        with (folder / "analysis/b0_rows.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        expected = {(seed, p, a) for p in (807, 2085) for a in (1., .1, .03, .01, .003, .001)}
        actual = {(int(r["seed"]), int(r["patient_id"]), float(r["alpha"])) for r in rows}
        if actual != expected or len(rows) != 12:
            raise RuntimeError("B0 ladder cell coverage is incomplete")
        verdicts[str(seed)] = read(folder / "analysis/b0_summary.json")["per_cell_verdict"]
    output = args.output.expanduser().resolve()
    if output.exists() or any(output.is_relative_to(p) for p in folders):
        raise ValueError("Archive must be new and outside the result directories")
    report = {"execution_and_integrity": "PASS", "jobs": ids, "local_derivative_verdicts": verdicts,
              "note": "Scientific unresolved_band is a valid result, not an execution failure. No alpha=1 validity claim."}
    with tarfile.open(output, "x:gz") as archive:
        archive.add(submission, arcname="submission.txt")
        for job, task in checks:
            archive.add(root / f"postcheck_{job}_{task}.json", arcname=f"postcheck_{job}_{task}.json")
        for folder in folders:
            for p in sorted(folder.rglob("*")):
                if p.is_symlink():
                    raise RuntimeError(f"Unexpected result symlink: {p}")
                if p.is_file() and (args.include_checkpoints or p.suffix not in (".pt", ".npz")):
                    archive.add(p, arcname=str(p.relative_to(root)))
        logs = Path(__file__).resolve().parents[2] / "logs"
        for pattern in (f"oct_e0b0_test-{ids['GPU_TEST']}.*", f"oct_e0-{ids['E0']}.*", f"oct_b0-{ids['B0_ARRAY']}_*.*"):
            for p in logs.glob(pattern):
                archive.add(p, arcname=f"logs/{p.name}")
    print(json.dumps({**report, "archive": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
