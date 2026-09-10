#!/usr/bin/env python3
"""Per-patient seed statistics for the 05 patient-LOO pilot (offline, CPU only).

Reads ``patient_seed_summary.csv`` (and, when present, ``runs/*.json``) from a
05 output directory and writes

    patient_seed_stats.csv / patient_seed_stats.json

with, for every patient and every endpoint:

    mean, sample SD (ddof=1), sample variance, min, max, positive/negative seed
    fraction, majority direction, sign concordance.

It also reports the two *path* decompositions of the full effect that do not
need an interaction term:

    value-first:   J11 - J00 = (J10 - J00) + (J11 - J10)
                                 value        relabel | P1
    relabel-first: J11 - J00 = (J01 - J00) + (J11 - J01)
                                 relabel      value | M1

and the Shapley split (average of the two orderings). J11-J10 is the relabel
effect conditional on P1. The continuous score's target is J10-J00.

This script describes Stage-1 row means (already averaged over attack seeds).
For crossed/nested seed variance components use 09_seed_variance.py; this
script's mean_over_se is not a crossed-design uncertainty estimate.

Usage:
    python scripts/cross_stage/06_pilot_seed_stats.py \
        --pilot_dir results/cross_stage_patient_loo_05_pilot_v3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# csv column -> short endpoint name
ENDPOINTS = {
    "primary_delta_full_matched_cross_entropy": "full_ce",
    "delta_value_matched_cross_entropy": "value_ce",
    "delta_relabel_matched_cross_entropy": "relabel_ce",
    "delta_full_matched_auc": "full_auc",
    "delta_full_matched_balanced_accuracy": "full_bacc",
    "delta_full_matched_accuracy": "full_acc",
    "deleted_mean_js": "deleted_js",
    "all_affected_mean_js": "all_affected_js",
}
DERIVED = {
    # J11 - J10  = full - value        (relabel effect at LOO vectors P1)
    "relabel_given_P1_ce": ("full_ce", "value_ce"),
    # J11 - J01  = full - relabel      (value effect at LOO labels M1)
    "value_given_M1_ce": ("full_ce", "relabel_ce"),
}


def describe(values: np.ndarray) -> Dict[str, object]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "sd": None, "var": None, "min": None, "max": None,
                "range": None, "pos_frac": None, "neg_frac": None, "majority_direction": None,
                "sign_concordance": None, "mean_over_se": None, "values": []}
    out = {
        "n_seeds": int(n),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if n > 1 else None,
        "var": float(v.var(ddof=1)) if n > 1 else None,
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
        "pos_frac": float((v > 0).mean()),
        "neg_frac": float((v < 0).mean()),
        "majority_direction": ("positive" if (v > 0).sum() > (v < 0).sum()
                               else "negative" if (v < 0).sum() > (v > 0).sum()
                               else "tie"),
        "sign_concordance": float(max((v > 0).mean(), (v < 0).mean())),
        # paired t-like ratio; descriptive only (n is tiny)
        "mean_over_se": (float(v.mean() / (v.std(ddof=1) / np.sqrt(n)))
                         if n > 1 and v.std(ddof=1) > 0 else None),
        "values": [float(x) for x in v],
    }
    return out


def _no_nan(obj):
    """Recursively replace float NaN/inf by None so the JSON is standard."""
    if isinstance(obj, dict):
        return {k: _no_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_no_nan(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def load_rows(pilot_dir: Path) -> List[dict]:
    path = pilot_dir / "patient_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def load_condition_levels(pilot_dir: Path) -> Dict[tuple, dict]:
    """Optional: raw J00/J10/J01/J11 matched-class CE from runs/*.json."""
    out = {}
    for p in sorted((pilot_dir / "runs").glob("seed*_patient*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        cls = str(r["patient"]["oct_class"])
        key = (int(r["seed"]), int(r["patient"]["patient_id"]))
        def get(cond, key):
            entry = r["conditions"].get(cond)
            return None if entry is None else float(entry["per_class"][cls][key])
        out[key] = {cond: get(cond, "cross_entropy") for cond in ("J00", "J10", "J01", "J11")}
        out[key].update({f"{cond}_auc": get(cond, "auc") for cond in ("J00", "J10", "J01", "J11")})
        out[key]["membership_conditions_evaluated"] = r.get("membership_conditions_evaluated", "J00,J10,J01,J11")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", required=True)
    ap.add_argument("--out_prefix", default="patient_seed_stats")
    args = ap.parse_args()
    pilot_dir = Path(args.pilot_dir)
    rows = load_rows(pilot_dir)
    levels = load_condition_levels(pilot_dir) if (pilot_dir / "runs").exists() else {}

    by_patient: Dict[int, List[dict]] = {}
    for r in rows:
        by_patient.setdefault(int(r["patient_id"]), []).append(r)

    patients = []
    for pid, prs in sorted(by_patient.items()):
        prs = sorted(prs, key=lambda r: int(r["seed"]))
        rec = {
            "patient_id": pid,
            "oct_class": int(prs[0]["oct_class"]),
            "class_name": prs[0]["class_name"],
            "n_images": int(prs[0]["n_images"]),
            "seeds": [int(r["seed"]) for r in prs],
            "endpoints": {},
        }
        arrays = {}
        for col, name in ENDPOINTS.items():
            if col in prs[0]:
                arrays[name] = np.asarray([float(r[col]) for r in prs])
        for name, (a, b) in DERIVED.items():
            if a in arrays and b in arrays:
                arrays[name] = arrays[a] - arrays[b]
        if "value_ce" in arrays and "relabel_ce" in arrays and "full_ce" in arrays:
            arrays["interaction_ce"] = arrays["full_ce"] - arrays["value_ce"] - arrays["relabel_ce"]
            # Shapley split of the full effect into two pathway shares
            arrays["shapley_value_share_ce"] = 0.5 * (arrays["value_ce"] + arrays["value_given_M1_ce"])
            arrays["shapley_relabel_share_ce"] = 0.5 * (arrays["relabel_ce"] + arrays["relabel_given_P1_ce"])
        for name, v in arrays.items():
            rec["endpoints"][name] = describe(v)
        if levels:
            rec["condition_levels"] = {
                str(int(r["seed"])): levels.get((int(r["seed"]), pid)) for r in prs
            }
        patients.append(rec)

    # class-level pooled summary (patient means)
    classes = {}
    for rec in patients:
        classes.setdefault(rec["class_name"], []).append(rec)
    class_summary = {}
    for cname, recs in classes.items():
        class_summary[cname] = {}
        for name in recs[0]["endpoints"]:
            means = np.asarray([np.nan if r["endpoints"][name]["mean"] is None else r["endpoints"][name]["mean"]
                                for r in recs], dtype=float)
            finite = means[np.isfinite(means)]
            sds = [r["endpoints"][name]["sd"] for r in recs if r["endpoints"][name]["sd"] is not None]
            vars_ = [r["endpoints"][name]["var"] for r in recs if r["endpoints"][name]["var"] is not None]
            class_summary[cname][name] = {
                "n_patients": int(len(recs)),
                "n_patients_with_values": int(len(finite)),
                "mean_of_patient_means": float(finite.mean()) if len(finite) else None,
                "sd_between_patients": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                "mean_within_patient_seed_sd": float(np.mean(sds)) if sds else None,
                "rms_within_patient_seed_sd": float(np.sqrt(np.mean(vars_))) if vars_ else None,
                "seed_variance_note": ("SD across Stage-1 row means, after the configured attack-seed "
                                       "averaging. This is not an isolated Stage-1 variance component. "
                                       "Use 09 for crossed/nested decomposition and grand-mean SE."),
                "patients_positive": int((finite > 0).sum()),
                "patients_negative": int((finite < 0).sum()),
            }

    # flat CSV
    flat_fields = ["patient_id", "oct_class", "class_name", "n_images", "n_seeds"]
    stat_keys = ["mean", "sd", "var", "min", "max", "pos_frac", "neg_frac",
                 "majority_direction", "sign_concordance", "mean_over_se"]
    endpoint_names = list(patients[0]["endpoints"].keys())
    for name in endpoint_names:
        for k in stat_keys:
            flat_fields.append(f"{name}__{k}")
    out_csv = pilot_dir / f"{args.out_prefix}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for rec in patients:
            row = {k: rec[k] for k in ("patient_id", "oct_class", "class_name", "n_images")}
            row["n_seeds"] = len(rec["seeds"])
            for name in endpoint_names:
                for k in stat_keys:
                    row[f"{name}__{k}"] = rec["endpoints"][name][k]
            w.writerow(row)
    out_json = pilot_dir / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(_no_nan({
        "source": str(pilot_dir / "patient_seed_summary.csv"),
        "notes": {
            "relabel_given_P1_ce": "J11 - J10: relabel effect evaluated at LOO vectors",
            "value_given_M1_ce": "J11 - J01: value effect evaluated at LOO labels",
            "interaction_ce": "J11 - J10 - J01 + J00",
            "shapley_*": "average of the two path orderings; value+relabel shares sum to full",
            "sign": "positive CE change = attack worse after deletion",
            "mean_over_se": "Descriptive row-mean ratio; not a crossed-design t statistic or CI. Use 09.",
        },
        "patients": patients,
        "class_summary": class_summary,
    }), indent=2, allow_nan=False), encoding="utf-8")

    # console table
    print(f"{'pid':>6} {'class':>7} {'img':>4} | {'full_ce mean':>12} {'sd':>9} {'min':>9} {'max':>9} "
          f"{'maj':>9} | {'value':>9} {'J11-J10':>9} | {'full_auc':>9}")
    def fmt(v, w=9, plus=True):
        if v is None:
            return f"{'n/a':>{w}}"
        return f"{v:>+{w}.6f}" if plus else f"{v:>{w}.6f}"
    for rec in patients:
        e = rec["endpoints"]
        fc = e["full_ce"]
        print(f"{rec['patient_id']:>6} {rec['class_name']:>7} {rec['n_images']:>4} | "
              f"{fmt(fc['mean'], 12)} {fmt(fc['sd'], plus=False)} {fmt(fc['min'])} {fmt(fc['max'])} "
              f"{str(fc['majority_direction']):>9} | {fmt(e['value_ce']['mean'])} "
              f"{fmt(e['relabel_given_P1_ce']['mean'])} | {fmt(e['full_auc']['mean'])}")
    print("\nclass summary (mean of patient means / mean within-patient seed SD):")
    for cname, s in class_summary.items():
        for name in ("full_ce", "value_ce", "relabel_ce", "interaction_ce",
                     "relabel_given_P1_ce", "value_given_M1_ce"):
            if name in s and s[name]['mean_of_patient_means'] is not None:
                bp = s[name]['sd_between_patients']; wp = s[name]['rms_within_patient_seed_sd']
                print(f"  {cname:>7} {name:>22}: {s[name]['mean_of_patient_means']:+.6f} "
                      f"(between-patient SD {'n/a' if bp is None else f'{bp:.6f}'}, "
                      f"RMS within-patient seed SD {'n/a' if wp is None else f'{wp:.6f}'})")
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    main()
