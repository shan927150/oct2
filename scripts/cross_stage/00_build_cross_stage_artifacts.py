#!/usr/bin/env python3
"""
00_build_cross_stage_artifacts.py

Reconstruct the raw-sample -> attack-row provenance for the whole MIA pipeline,
so that every attack-model training vector can be traced back to (shadow_id,
in/out, raw OCT image index, patient_id, OCT class). This is the prerequisite
for any cross-stage (Direction B) attribution: without it, the 804 degenerate
pairs cannot be mapped back to raw images / patients and every downstream score
risks a silent index mismatch.

No GPU. No model re-run. Provenance comes purely from the split JSON + the fixed
concatenation order in attack.py::_train_shadows.

Outputs (under --out_dir):
    provenance.csv            one row per attack-train row (full map)
    class_local_index.csv     (oct_class, per_class_local) -> raw provenance
    manifest.json             hashes, counts, validation report

Typical use on Delta (from repo root, after `module load pytorch-conda/2.8`):
    python scripts/cross_stage/00_build_cross_stage_artifacts.py \
        --attack_data ./results/oct_mia_tracin/tracin/attack_data.npz \
        --output_dir  ./results/oct_mia_tracin \
        --out_dir     ./results/cross_stage/artifacts \
        --load_dataset            # attaches patient_id + cross-checks class

Omit --load_dataset to run structurally only (still gives shadow/raw mapping,
but patient_id = -1 and the class cross-check is skipped).
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# make repo `src/` importable regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import common  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack_data", required=True,
                    help="path to attack_data.npz produced by run_tracin.py")
    ap.add_argument("--output_dir", required=True,
                    help="results dir whose splits/ holds the split JSON")
    ap.add_argument("--split_path", default=None,
                    help="explicit split JSON (else auto-resolved from output_dir/splits)")
    ap.add_argument("--out_dir", default="./results/cross_stage/artifacts")
    ap.add_argument("--load_dataset", action="store_true",
                    help="load OCT via repo loader to attach patient_id + verify class")
    ap.add_argument("--preset", default="oct")
    ap.add_argument(
        "--config_overrides_json", default=None,
        help=("optional JSON object applied to the preset, e.g. "
              "{\"n_total_samples\":40000,\"target_data_size\":2000,"
              "\"shadow_data_size\":2000,\"n_shadow\":5}"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. locate + load inputs
    split_path = common.resolve_split_path(args.output_dir, args.split_path)
    split = common.load_split(split_path)
    ad = common.load_attack_data(args.attack_data)

    y_raw = groups_raw = None
    if args.load_dataset:
        overrides = json.loads(args.config_overrides_json) if args.config_overrides_json else None
        _, y_raw, groups_raw = common.try_load_dataset(args.preset, overrides=overrides)
        if y_raw is None or groups_raw is None:
            raise RuntimeError(
                "--load_dataset was requested but the real dataset could not be loaded. "
                "Refusing a structural-only PASS. Check the preset/config overrides and data path.")

    # 2. reconstruct
    rows, report = common.reconstruct_provenance(
        split=split,
        attack_train_y=ad["attack_train_y"],
        train_classes=ad["train_classes"],
        y_raw=y_raw,
        groups_raw=groups_raw,
    )

    # 3. validation gate — loud, because everything downstream depends on it
    ok = report["structural_match"] and report["membership_mismatch"] == 0
    if report["class_checked"]:
        ok = ok and report["class_mismatch"] == 0
    logger.info("=" * 66)
    logger.info("PROVENANCE VALIDATION")
    logger.info(f"  shadows                 : {report['n_shadows']}")
    logger.info(f"  rows (split vs attack)  : {report['n_rows_from_split']} vs "
                f"{report['n_rows_in_attack_data']}  "
                f"{'OK' if report['structural_match'] else 'MISMATCH'}")
    logger.info(f"  membership mismatches   : {report['membership_mismatch']}")
    if report["class_checked"]:
        logger.info(f"  class mismatches        : {report['class_mismatch']} "
                    f"(y[raw_index] vs train_classes)")
    else:
        logger.info("  class cross-check       : SKIPPED (no --load_dataset)")
    logger.info(f"  patient_id attached     : {report['patient_attached']}")
    logger.info(f"  VERDICT                 : {'PASS' if ok else 'FAIL'}")
    logger.info("=" * 66)

    # 4. write provenance.csv
    prov_path = out / "provenance.csv"
    fields = ["global_row", "shadow_id", "role", "membership", "raw_index",
              "oct_class_attack", "oct_class_raw", "patient_id", "per_class_local"]
    with open(prov_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    logger.info(f"wrote {prov_path}  ({len(rows)} rows)")

    # 5. write class_local_index.csv (what the 804-pair csv indexes into)
    cl_path = out / "class_local_index.csv"
    with open(cl_path, "w", newline="") as f:
        # ``fields`` already contains per_class_local; do not duplicate the CSV header.
        w = csv.DictWriter(f, fieldnames=["oct_class"] + fields)
        w.writeheader()
        for r in rows:
            row = {"oct_class": r["oct_class_attack"]}
            row.update({k: r[k] for k in fields})
            w.writerow(row)
    logger.info(f"wrote {cl_path}")

    # 6. small human-readable summaries
    arr = common.rows_to_arrays(rows)
    per_shadow = {int(s): int((arr["shadow_id"] == s).sum())
                  for s in np.unique(arr["shadow_id"])}
    per_class = {common.CLASS_NAMES.get(int(c), int(c)): int((arr["oct_class_attack"] == c).sum())
                 for c in np.unique(arr["oct_class_attack"])}
    n_members = int((arr["membership"] == 1).sum())
    n_patients = int(len(np.unique(arr["patient_id"][arr["patient_id"] >= 0]))) \
        if report["patient_attached"] else -1

    manifest = {
        "split_path": split_path,
        "attack_data": os.path.abspath(args.attack_data),
        "attack_data_sha1": common.sha1_of_file(args.attack_data),
        "split_sha1": common.sha1_of_file(split_path),
        "validation": report,
        "verdict_pass": bool(ok),
        "rows_per_shadow": per_shadow,
        "rows_per_class": per_class,
        "n_members": n_members,
        "n_nonmembers": len(rows) - n_members,
        "n_unique_patients_in_attack_train": n_patients,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote {out / 'manifest.json'}")

    if not ok:
        logger.error("VALIDATION FAILED — do not proceed to 01/02 until resolved. "
                     "Most likely --split_path points at the wrong split JSON.")
        sys.exit(2)
    logger.info("Provenance OK. Next: 01_graph_sanity.py, then 02_pair_separation_test.py")


if __name__ == "__main__":
    main()
