#!/usr/bin/env python3
"""Paired CE/AUC/Brier seed effects from completed 05 runs (no model training).

Never take the AUC of seed-averaged probabilities: each seed's AUC is evaluated
first, then matched condition differences are formed. Incomplete grids are
reported explicitly and receive no variance decomposition.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from formal_analysis import clean_json, seed_variance_components


EFFECTS = {"value": {"J10": 1, "J00": -1},
           "full": {"J11": 1, "J00": -1},
           "relabel": {"J01": 1, "J00": -1},
           "relabel_given_P1": {"J11": 1, "J10": -1},
           "interaction": {"J11": 1, "J10": -1, "J01": -1, "J00": 1}}


def read_effects(pilot_dir):
    root = Path(pilot_dir).resolve()
    cfg = json.loads((root / "experiment_config.json").read_text())["args"]
    expected_r = sorted(map(int, cfg["seeds"]))
    groups, records = {}, []
    for path in sorted((root / "runs").glob("seed*_patient*.json")):
        run = json.loads(path.read_text())
        cls = str(run["patient"]["oct_class"])
        r = int(run["seed"])
        if r not in expected_r:
            raise ValueError(f"Unexpected Stage-1 seed {r}: {path}")
        base = run["conditions"]["J00"]["per_class"][cls]
        ce_definition = base.get("ce_definition", "legacy_probability_CE")
        seeds = list(map(int, base.get("attack_seeds", [r*100+int(cls)])))
        design = base.get("attack_seed_design", "nested_derived_from_stage1_seed")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate attack seeds: {path}")
        for effect, terms in EFFECTS.items():
            conditions = [run["conditions"].get(c) for c in terms]
            if any(c is None for c in conditions):
                continue
            for metric in ("cross_entropy", "auc", "brier"):
                arrays = []
                for c in terms:
                    entry = run["conditions"][c]["per_class"][cls]
                    if entry.get("ce_definition", "legacy_probability_CE") != ce_definition:
                        raise ValueError(f"Mixed CE definitions in {path}: {c}")
                    if list(map(int, entry.get("attack_seeds", seeds))) != seeds:
                        raise ValueError(f"Unpaired condition seed order in {path}: {c}")
                    values = entry.get("per_rep_" + metric)
                    if values is None and len(seeds) == 1 and metric in entry:
                        values = [entry[metric]]
                    if values is None:
                        break
                    if len(values) != len(seeds):
                        raise ValueError(f"Missing per-seed {metric}: {path}, {c}")
                    arrays.append(terms[c] * np.asarray(values, dtype=float))
                if len(arrays) != len(terms):
                    continue
                delta = np.sum(arrays, axis=0)
                pid = int(run["patient"]["patient_id"])
                key = (int(run["affected_shadow"]), int(cls), pid, metric, effect)
                info = groups.setdefault(key, {"cells": {}, "design": design,
                                              "ce_definition": ce_definition,
                                              "split_sha256": run["split_sha256"]})
                if r in info["cells"] or info["design"] != design or info["split_sha256"] != run["split_sha256"] or info["ce_definition"] != ce_definition:
                    raise ValueError(f"Duplicate/mixed experiment cells: {path}")
                info["cells"][r] = (seeds, delta)
                for k, value in zip(seeds, delta):
                    records.append({"pilot_dir": str(root), "affected_shadow": key[0],
                                    "oct_class": key[1], "patient_id": pid,
                                    "stage1_seed": r, "attack_seed": k, "design": design,
                                    "ce_definition": ce_definition,
                                    "metric": metric, "effect": effect,
                                    "paired_delta": float(value), "split_sha256": run["split_sha256"]})
    components = []
    for key, info in sorted(groups.items()):
        cells, design = info["cells"], info["design"]
        item = dict(zip(("affected_shadow", "oct_class", "patient_id", "metric", "effect"), key))
        item.update(pilot_dir=str(root), split_sha256=info["split_sha256"],
                    ce_definition=info["ce_definition"],
                    expected_stage1_seeds=expected_r, observed_stage1_seeds=sorted(cells),
                    attack_seeds_by_stage1={str(r): seeds for r, (seeds, _) in cells.items()})
        sizes = {len(s) for s, _ in cells.values()}
        if set(cells) != set(expected_r) or len(sizes) != 1:
            item.update(status="incomplete_grid", variance_grand_mean=None, se_grand_mean=None)
        else:
            seed_lists = [cells[r][0] for r in expected_r]
            if design == "crossed_fixed_panel" and any(s != seed_lists[0] for s in seed_lists):
                raise ValueError(f"Crossed design has mismatched attack seed panels: {key}")
            if design == "nested_derived_from_stage1_seed":
                flattened = sum(seed_lists, [])
                if len(flattened) != len(set(flattened)):
                    raise ValueError(f"Nested design reuses attack seeds across Stage-1 seeds: {key}")
            matrix = np.stack([cells[r][1] for r in expected_r])
            item.update(seed_variance_components(matrix, design))
        components.append(item)
    return records, components


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        for r in clean_json(rows):
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    raw, components = [], []
    for root in args.pilot_dirs:
        a, b = read_effects(root)
        raw.extend(a); components.extend(b)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_seed_effects.csv", raw)
    write_csv(out / "seed_variance_components.csv", components)
    payload = {"components": components, "n_paired_records": len(raw),
               "note": "Each component fit is conditional on one patient/split/shadow. "
                       "Different output directories/interventions are never pooled. "
                       "Calibration estimates; no automatic sample-size or damping selection."}
    (out / "seed_variance_components.json").write_text(
        json.dumps(clean_json(payload), indent=2, allow_nan=False))
    print(f"Wrote {len(raw)} paired effects and {len(components)} variance fits to {out}")


if __name__ == "__main__":
    main()
