#!/usr/bin/env python3
"""Read-only eligibility preflight for the formal patient-LOO experiment.

Builds (or reuses) the patient-level split for a given configuration and
counts, per shadow model, OCT class and image-count stratum, how many
patients would be eligible for deletion.  No model is trained and nothing
in an existing pilot directory is modified; the split JSON is written into
``--output_dir`` so that a later 05 run with the same ``--output_dir`` and
split arguments reuses exactly this split.

It also proposes a set of affected shadows and disjoint patient panels
(different patient IDs across shadows, class-balanced, chosen by
``--selection_seed`` only) so that the formal patient list is frozen
before any outcome is seen.

Usage:
    python scripts/cross_stage/08_eligibility_preflight.py \
        --output_dir results/cross_stage_patient_loo_formal_40k --data_dir ./data \
        --n_total_samples 40000 --target_data_size 2000 --shadow_data_size 2000 \
        --n_shadow 5 --classes 1 2 --min_patient_images 5 --max_patient_images 15 \
        --patients_per_class_per_shadow 10 --n_affected_shadows 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_total_samples", type=int, default=40000)
    ap.add_argument("--target_data_size", type=int, default=2000)
    ap.add_argument("--shadow_data_size", type=int, default=2000)
    ap.add_argument("--n_shadow", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=42005)
    ap.add_argument("--selection_seed", type=int, default=42006)
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--min_patient_images", type=int, default=5)
    ap.add_argument("--max_patient_images", type=int, default=15)
    ap.add_argument("--patients_per_class_per_shadow", type=int, default=10)
    ap.add_argument("--n_affected_shadows", type=int, default=3)
    ap.add_argument("--strata", type=int, nargs="*", default=[5, 8, 11, 16, 21, 31],
                    help="image-count bin edges for the eligibility table")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("pilot05", HERE / "05_end_to_end_patient_loo_pilot.py")
    pilot = importlib.util.module_from_spec(spec); sys.modules["pilot05"] = pilot; spec.loader.exec_module(pilot)

    pargs = argparse.Namespace(
        output_dir=args.output_dir, data_dir=args.data_dir, n_total_samples=args.n_total_samples,
        target_data_size=args.target_data_size, shadow_data_size=args.shadow_data_size,
        n_shadow=args.n_shadow, split_seed=args.split_seed, shadow_epochs=50, attack_epochs=50,
        shadow_batch_size=128, attack_batch_size=128, shadow_lr=1e-3, attack_lr=1e-2)
    cfg = pilot.build_config(pargs)
    X, y, groups = pilot.load_dataset(cfg)
    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    split_path, splits = pilot.prepare_split(cfg, X, y, groups, out_dir, overwrite=False)
    validation = pilot.validate_patient_split(splits, groups)

    edges = list(args.strata)
    table: Dict[str, object] = {}
    eligible: Dict[int, Dict[int, List[dict]]] = {}
    for sid, split in enumerate(splits["shadow_models"]):
        idx = np.asarray(split["train_idx"], dtype=np.int64)
        per_class: Dict[int, List[dict]] = {c: [] for c in args.classes}
        for pid in np.unique(groups[idx]):
            pidx = idx[groups[idx] == pid]
            vals, counts = np.unique(y[pidx], return_counts=True)
            cls = int(vals[counts.argmax()])
            if cls in per_class:
                per_class[cls].append({"patient_id": int(pid), "n_images": int(len(pidx)),
                                       "raw_indices": sorted(pidx.tolist())})
        eligible[sid] = {}
        table[f"shadow_{sid}"] = {}
        for cls, rows in per_class.items():
            counts_by_bin = {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                counts_by_bin[f"[{lo},{hi})"] = int(sum(lo <= r["n_images"] < hi for r in rows))
            elig = [r for r in rows if args.min_patient_images <= r["n_images"] <= args.max_patient_images]
            eligible[sid][cls] = elig
            table[f"shadow_{sid}"][pilot.CLASS_NAMES[cls]] = {
                "n_patients_in_train": len(rows),
                "n_eligible": len(elig),
                "by_image_count_bin": counts_by_bin,
                "median_images": float(np.median([r["n_images"] for r in rows])) if rows else None,
            }

    # choose affected shadows: those with the most eligible patients in the scarcest class
    scarcity = {sid: min(len(eligible[sid][c]) for c in args.classes) for sid in eligible}
    ranked = sorted(scarcity, key=lambda s: (-scarcity[s], s))
    chosen = ranked[:args.n_affected_shadows]
    rng = np.random.default_rng(args.selection_seed)
    used: set = set()
    panels = {}
    shortfall = {}
    for sid in chosen:
        panels[sid] = {}
        for cls in args.classes:
            pool = [r for r in eligible[sid][cls] if r["patient_id"] not in used]
            rng.shuffle(pool)
            take = pool[:args.patients_per_class_per_shadow]
            if len(take) < args.patients_per_class_per_shadow:
                shortfall[f"shadow_{sid}_class_{cls}"] = args.patients_per_class_per_shadow - len(take)
            used.update(r["patient_id"] for r in take)
            panels[sid][pilot.CLASS_NAMES[cls]] = take

    split_hash = pilot.sha256_file(Path(split_path))
    report = {
        "split_path": str(split_path), "split_sha256": split_hash, "split_validation": validation,
        "args": vars(args), "eligibility_table": table,
        "min_eligible_per_class_by_shadow": {str(s): v for s, v in scarcity.items()},
        "proposed_affected_shadows": chosen,
        "proposed_patient_panels_disjoint_across_shadows": {str(s): v for s, v in panels.items()},
        "shortfall": shortfall,
        "panel_complete": not shortfall,
        "consumed_by": "05 --patient_panel_json <this file> --affected_shadow <sid> [--require_complete_panel]",
        "note": ("selection uses split metadata and selection_seed only; freeze this file before "
                 "training. Patients are unique across shadows to avoid pseudoreplication."),
    }
    pilot.json_dump(out_dir / "eligibility_preflight.json", report)
    if shortfall:
        print("WARNING: panel incomplete:", json.dumps(shortfall))
    print(json.dumps({k: report[k] for k in ("eligibility_table", "proposed_affected_shadows", "shortfall")}, indent=2))
    print("wrote", out_dir / "eligibility_preflight.json")


if __name__ == "__main__":
    main()
