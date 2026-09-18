#!/usr/bin/env python3
"""Freeze one human-reviewed Route A checkpoint after the Step-1 report exists.

This command never chooses an epoch.  It validates the five completed affected-
shadow runs and records the exact user/advisor choice, report, curves and
checkpoints in ``frozen_E_star.json``.  Step 2 refuses to prepare without this
record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_report_module():
    spec = importlib.util.spec_from_file_location("a20_report_freeze", HERE / "20_A_convergence_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="completed convergence_E... directory")
    ap.add_argument("--epoch", required=True, type=int, help="the already reviewed checkpoint; this script does not choose it")
    ap.add_argument("--confirmed_by", required=True, help="short provenance label, e.g. 'advisor meeting 2026-09-19'")
    ap.add_argument("--note", required=True, help="brief reason based on the reviewed curves")
    ap.add_argument("--out", default=None, help="default: <root>/frozen_E_star.json")
    args = ap.parse_args(argv)
    if not args.confirmed_by.strip() or not args.note.strip():
        raise SystemExit("--confirmed_by and --note must record a non-empty human review")

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / "frozen_E_star.json"
    if out.exists():
        raise SystemExit(f"Selection already exists and is immutable: {out}")
    report_path = root / "report" / "plateau_report.json"
    if not report_path.is_file():
        raise SystemExit(f"Run 20_A_convergence_report.py first; missing {report_path}")

    report = load_report_module()
    runs, problems, missing = report.load_runs(root)
    if problems or missing:
        raise SystemExit("Cannot freeze an incomplete panel: " + "; ".join(problems + missing))
    report.validate_panel(runs)
    affected = {name: run for name, run in runs.items()
                if run["manifest"].get("role") == "affected_shadow"}
    if len(affected) != 5:
        raise SystemExit(f"Expected exactly five affected-shadow runs, found {sorted(affected)}")
    manifests = [run["manifest"] for run in affected.values()]
    seeds = sorted(int(m["seed"]) for m in manifests)
    if seeds != [42, 43, 44, 45, 46]:
        raise SystemExit(f"Unexpected affected-shadow seed panel: {seeds}")
    t0 = int(manifests[0]["original_epochs"])
    emax = int(manifests[0]["extended_epochs"])
    if not t0 < args.epoch <= emax:
        raise SystemExit(f"E* must be an observed extension checkpoint in ({t0}, {emax}], got {args.epoch}")

    checkpoints, curves = {}, {}
    for name, run in sorted(affected.items()):
        folder = Path(run["folder"])
        if args.epoch not in {int(row["epoch"]) for row in run["curve"]}:
            raise SystemExit(f"{name}: epoch {args.epoch} is absent from the reviewed curve")
        checkpoint = folder / "checkpoints" / f"epoch{args.epoch:03d}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"{name}: epoch {args.epoch} was not saved; choose a saved checkpoint")
        checkpoints[str(checkpoint)] = sha(checkpoint)
        curve = folder / "curve.csv"
        curves[str(curve)] = sha(curve)

    summary = json.loads(report_path.read_text(encoding="utf-8"))
    if summary.get("selected_E_star") is not None or summary.get("selection_status") != "not_frozen":
        raise SystemExit("The descriptive report unexpectedly contains an automatic selection")
    record = {
        "schema": "pathway2_A_frozen_E_star_v1",
        "status": "frozen_human_choice",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "convergence_root": str(root),
        "E_star": args.epoch,
        "original_epochs": t0,
        "extended_epochs": emax,
        "confirmed_by": args.confirmed_by,
        "note": args.note,
        "git_commit": manifests[0].get("git_commit"),
        "seeds": seeds,
        "report": str(report_path),
        "report_sha256": sha(report_path),
        "curve_sha256": curves,
        "checkpoint_sha256": checkpoints,
        "selection_was_not_automatic": True,
    }
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_E_star": args.epoch, "selection": str(out)}, indent=2))


if __name__ == "__main__":
    main()
