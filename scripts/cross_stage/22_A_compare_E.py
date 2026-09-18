#!/usr/bin/env python3
"""Matched-cell comparison of the frozen 50-epoch endpoint and one frozen E*.

``primary`` is the A1 review: original gamma_A=0.2, gamma_S=1 only.  ``full``
adds dose/preflight and the separately approved A2 damping sweep.  Metrics at
50 and E* are always computed on the same seed/patient/class cells; independent
qualified cohorts are never presented as an epoch comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from formal_analysis import GATES, comparison  # noqa: E402

METHODS = ("L1_lin_value", "L2_lin_value", "L2_retrain_value", "L3_lin_value", "L3_retrain_value")
SATURATION_RHS = 1e-6
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLORS = {"E50": "#2a78d6", "Estar": "#eb6834"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", "None", None):
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    values = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def key3(row):
    return int(row["seed"]), int(row["patient_id"]), int(row["oct_class"])


def unique(rows, key_fn, label):
    out = {}
    for row in rows:
        key = key_fn(row)
        if key in out:
            raise RuntimeError(f"Duplicate {label} row for {key}")
        out[key] = row
    return out


def passes(row, method):
    return all(row.get(gate) is True for gate in GATES[method])


def comparable_configs(reference, new, condition):
    a, b = reference["args"], new["args"]
    keys = (
        "n_total_samples", "target_data_size", "shadow_data_size", "n_shadow", "affected_shadow",
        "n_patients", "min_patient_images", "max_patient_images", "classes", "seeds", "split_seed",
        "selection_seed", "target_seed", "fixed_shadow_seed", "shadow_batch_size", "attack_epochs",
        "attack_batch_size", "shadow_lr", "attack_lr", "noop_replays", "gate_min_queries_per_label",
        "gate_min_class_auc", "gate_min_class_balanced_accuracy", "enforce_attack_gate",
        "require_all_classes", "enforce_noop_gate", "noop_tolerance", "deterministic",
        "removal_epochs", "attack_seed_reps", "trajectory_replays", "attack_seeds", "deletion_mode",
        "deletion_weight", "window_membership", "panel_shadow", "require_complete_panel",
    )
    mismatches = {key: {"E50": a.get(key), "E_star": b.get(key)} for key in keys if a.get(key) != b.get(key)}
    if mismatches:
        raise RuntimeError(f"{condition} E50/E* configurations are not comparable: {mismatches}")
    for key, value in reference.get("oct_config", {}).items():
        if key in {"output_dir", "data_dir", "target_epochs", "shadow_epochs"}:
            continue
        if new.get("oct_config", {}).get(key) != value:
            raise RuntimeError(f"{condition} oct_config differs in {key}")


def validate_truth_pair(reference_dir, new_dir, state, condition):
    reference_dir, new_dir = Path(reference_dir), Path(new_dir)
    for folder in (reference_dir, new_dir):
        if read_json(folder / "experiment_summary.json").get("status") != "complete":
            raise RuntimeError(f"Incomplete truth run: {folder}")
    replay = read_json(new_dir / "route_a_epoch50_replay_checks.json")
    if replay.get("schema") != "pathway2_A_epoch50_replay_v2" or replay.get("status") != "complete":
        raise RuntimeError(f"Missing complete epoch-50 replay certificate: {new_dir}")
    if int(replay.get("original_epochs", -1)) != 50 or int(replay.get("extended_epochs", -1)) != state["E_star"]:
        raise RuntimeError("Replay certificate epochs differ from the prepared state")
    if replay.get("selection_sha256") != state["selection_sha256"]:
        raise RuntimeError("Replay certificate does not use the prepared frozen E* selection")
    if Path(replay.get("reference_dir", "")).resolve() != reference_dir.resolve():
        raise RuntimeError(f"{condition} replay certificate names a different reference truth")
    if replay.get("reference_config_sha256") != sha(reference_dir / "experiment_config.json"):
        raise RuntimeError(f"{condition} reference truth config changed after replay")
    if replay.get("actual_counts") != replay.get("expected_counts"):
        raise RuntimeError("Replay certificate has incomplete Stage-1 path counts")
    if len(replay.get("E_star_anchor_checks", [])) != replay.get("expected_E_star_anchor_checks"):
        raise RuntimeError("Replay certificate has incomplete E* baseline/no-op anchors")
    if not all(row.get("passed") is True for row in replay["checks"] + replay["E_star_anchor_checks"]):
        raise RuntimeError("At least one replay check failed")
    if not all(row.get("regenerated_optimizer_sha256") and
               row.get("continued_optimizer_sha256_at_E_star") for row in replay["checks"]):
        raise RuntimeError("Replay certificate is missing a regenerated/continued native-Adam fingerprint")
    ref_config, new_config = read_json(reference_dir / "experiment_config.json"), read_json(new_dir / "experiment_config.json")
    if int(ref_config["args"]["shadow_epochs"]) != 50 or int(new_config["args"]["shadow_epochs"]) != state["E_star"]:
        raise RuntimeError("Truth endpoints do not match T0=50 and frozen E*")
    comparable_configs(ref_config, new_config, condition)
    if read_json(reference_dir / "splits/fresh_patient_split.json") != read_json(new_dir / "splits/fresh_patient_split.json"):
        raise RuntimeError(f"{condition} split changed between E50 and E*")
    ref_panel = read_json(reference_dir / "selected_patients.json")
    new_panel = read_json(new_dir / "selected_patients.json")
    if ref_panel.get("split_sha256") != new_panel.get("split_sha256") or ref_panel.get("patients") != new_panel.get("patients"):
        raise RuntimeError(f"{condition} selected-patient panel changed between E50 and E*")
    expected = {
        "target": 1,
        "fixed_shadow": int(ref_config["args"]["n_shadow"]) - 1,
        "affected_baseline_or_noop": len(ref_config["args"]["seeds"]) *
                                       (1 + int(ref_config["args"]["noop_replays"])),
        "loo": len(ref_config["args"]["seeds"]) * len(ref_panel["patients"]),
    }
    if replay.get("expected_counts") != expected:
        raise RuntimeError(f"{condition} replay panel counts differ from the frozen design")
    return {"reference_config_sha256": sha(reference_dir / "experiment_config.json"),
            "new_config_sha256": sha(new_dir / "experiment_config.json"),
            "new_replay_sha256": sha(new_dir / "route_a_epoch50_replay_checks.json")}


def ladder_rows(folder, require_original_damping=False):
    score = Path(folder) / "score_ladder_A0.2_S1"
    config = read_json(score / "score_config.json")
    if float(config.get("damping_attack", -1)) != .2 or float(config.get("damping_shadow", -1)) != 1.:
        raise RuntimeError(f"Score directory is not gamma_A=.2, gamma_S=1: {score}")
    if require_original_damping and config.get("damping_shadow_grid") not in ([], None):
        raise RuntimeError("A1 E* score must contain only the original gamma_S=1; damping sweep belongs to A2")
    read_json(score / "ladder_summary.json")
    return score, read_csv(score / "ladder_rows.csv")


def matched_ladder(reference_rows, new_rows):
    ref, new = unique(reference_rows, key3, "E50 ladder"), unique(new_rows, key3, "E* ladder")
    if set(ref) != set(new):
        raise RuntimeError(f"Ladder cell keys differ: only_E50={sorted(set(ref)-set(new))}, only_Estar={sorted(set(new)-set(ref))}")
    keys = sorted(ref)
    common = [key for key in keys if all(
        finite(ref[key].get(field)) and finite(new[key].get(field)) and
        passes(ref[key], field) and passes(new[key], field) for field in METHODS) and
        finite(ref[key].get("actual_value")) and finite(new[key].get("actual_value"))]
    by_method = {}
    for method in METHODS:
        use = [key for key in keys if finite(ref[key].get(method)) and finite(new[key].get(method)) and
               finite(ref[key].get("actual_value")) and finite(new[key].get("actual_value")) and
               passes(ref[key], method) and passes(new[key], method)]
        by_method[method] = {
            "n_matched": len(use), "keys": [list(k) for k in use],
            "E50": comparison([ref[k][method] for k in use], [ref[k]["actual_value"] for k in use]),
            "E_star": comparison([new[k][method] for k in use], [new[k]["actual_value"] for k in use]),
        }
    comparisons = {method: {
        "E50": comparison([ref[k][method] for k in common], [ref[k]["actual_value"] for k in common]),
        "E_star": comparison([new[k][method] for k in common], [new[k]["actual_value"] for k in common]),
    } for method in METHODS}
    dtheta = [key for key in keys if all(finite(row.get("dtheta_cosine")) and
              row.get("cg_dtheta_reliable") is True for row in (ref[key], new[key]))]
    signal = [key for key in keys if finite(ref[key].get("actual_value")) and finite(new[key].get("actual_value"))]
    rows_for_plot = [{"key": list(key), "actual_E50": ref[key]["actual_value"],
                      "pred_E50": ref[key]["L3_lin_value"], "actual_E_star": new[key]["actual_value"],
                      "pred_E_star": new[key]["L3_lin_value"],
                      "dtheta_cosine_E50": ref[key]["dtheta_cosine"],
                      "dtheta_cosine_E_star": new[key]["dtheta_cosine"]} for key in common]
    return {
        "n_total_cells_each": len(keys), "key_sets_exactly_equal": True,
        "all_method_common": {"n": len(common), "keys": [list(k) for k in common], "comparisons": comparisons},
        "method_specific_matched": by_method,
        "dtheta_cosine": {"n_matched": len(dtheta),
                           "E50": stats([ref[k]["dtheta_cosine"] for k in dtheta]),
                           "E_star": stats([new[k]["dtheta_cosine"] for k in dtheta]),
                           "paired_delta_Estar_minus_E50": stats([new[k]["dtheta_cosine"]-ref[k]["dtheta_cosine"] for k in dtheta])},
        "truth_signal": {"n_matched": len(signal),
                         "actual_value_E50": stats([ref[k]["actual_value"] for k in signal]),
                         "actual_value_E_star": stats([new[k]["actual_value"] for k in signal]),
                         "zero_predictor_mae_E50": float(np.mean([abs(ref[k]["actual_value"]) for k in signal])) if signal else None,
                         "zero_predictor_mae_E_star": float(np.mean([abs(new[k]["actual_value"]) for k in signal])) if signal else None},
        "plot_rows": rows_for_plot,
    }


def matched_ratios(reference_dir, new_dir):
    ref_rows, new_rows = read_csv(Path(reference_dir) / "checkpoint_ratios.csv"), read_csv(Path(new_dir) / "checkpoint_ratios.csv")
    ref, new = unique(ref_rows, key3, "E50 ratio"), unique(new_rows, key3, "E* ratio")
    if set(ref) != set(new):
        raise RuntimeError("Checkpoint-ratio cell keys differ between E50 and E*")
    keys = sorted(ref)
    fields = ("full_over_theta", "full_over_median_seed_distance", "dose01_over_point1_full", "cosine_dose01_full")
    return {"n_matched": len(keys), "fields": {field: {
        "E50": stats([ref[k].get(field) for k in keys]), "E_star": stats([new[k].get(field) for k in keys]),
        "paired_delta_Estar_minus_E50": stats([new[k][field] - ref[k][field] for k in keys
                                                if finite(ref[k].get(field)) and finite(new[k].get(field))])
    } for field in fields}}


def damping_rows(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("args", {}).get("mode") != "damping":
        raise RuntimeError(f"Incomplete damping output: {folder}")
    return read_csv(folder / "all_rows.csv"), manifest


def damping_key(row):
    return str(row["condition"]), float(row["gamma"]), *key3(row)


def projection_summary(rows):
    qualified = [r for r in rows if r.get("qualified") is True]
    return {"n_cells": len(rows), "n_qualified": len(qualified),
            "cosine": stats([r.get("cosine") for r in qualified]),
            "norm_ratio": stats([r.get("norm_ratio") for r in qualified]),
            "h_projection": comparison([r.get("pred_h_projection") for r in qualified],
                                       [r.get("true_h_projection") for r in qualified])}


def matched_damping(reference_dir, new_dir):
    ref_rows, ref_manifest = damping_rows(reference_dir)
    new_rows, new_manifest = damping_rows(new_dir)
    expected = [.01, .03, .1, .3, 1., 2.]
    if [float(v) for v in new_manifest["args"]["damping_grid"]] != expected:
        raise RuntimeError(f"Unexpected A2 E* damping grid: {new_manifest['args']['damping_grid']}")
    ref = unique(ref_rows, damping_key, "E50 damping")
    new = unique(new_rows, damping_key, "E* damping")
    ref_pairs = sorted({(k[0], k[1]) for k in ref})
    new_pairs = sorted({(k[0], k[1]) for k in new})
    overlap, matched = sorted(set(ref_pairs) & set(new_pairs)), {}
    for condition, gamma in overlap:
        rkeys = {k[2:] for k in ref if k[:2] == (condition, gamma)}
        nkeys = {k[2:] for k in new if k[:2] == (condition, gamma)}
        if rkeys != nkeys:
            raise RuntimeError(f"Damping cell keys differ for {condition}, gamma={gamma:g}")
        use = [key for key in sorted(rkeys)
               if ref[(condition, gamma) + key].get("qualified") is True and
               new[(condition, gamma) + key].get("qualified") is True]
        rr = [ref[(condition, gamma) + key] for key in use]
        nn = [new[(condition, gamma) + key] for key in use]
        matched[f"{condition}_gamma{gamma:g}"] = {
            "condition": condition, "gamma": gamma, "n_total_each": len(rkeys), "n_matched_qualified": len(use),
            "n_qualified_E50": sum(ref[(condition, gamma) + key].get("qualified") is True for key in rkeys),
            "n_qualified_E_star": sum(new[(condition, gamma) + key].get("qualified") is True for key in rkeys),
            "E50": projection_summary(rr), "E_star": projection_summary(nn),
        }
    only_new = {}
    for condition, gamma in sorted(set(new_pairs) - set(ref_pairs)):
        only_new[f"{condition}_gamma{gamma:g}"] = projection_summary(
            [row for key, row in new.items() if key[:2] == (condition, gamma)])
    only_ref = [f"{condition}_gamma{gamma:g}" for condition, gamma in sorted(set(ref_pairs) - set(new_pairs))]

    def cell_rhs(rows):
        values = {}
        for row in rows:
            if row["condition"] != "full" or not finite(row.get("rhs_norm")):
                continue
            key = key3(row)
            values.setdefault(key, []).append(row["rhs_norm"])
        for key, vals in values.items():
            if max(vals) - min(vals) > 1e-10 * max(1., abs(vals[0])):
                raise RuntimeError(f"rhs_norm changes across gamma for {key}")
        return {key: vals[0] for key, vals in values.items()}

    ref_rhs, new_rhs = cell_rhs(ref_rows), cell_rhs(new_rows)
    rhs_keys = sorted(set(ref_rhs) & set(new_rhs))
    return {"matched_overlap": matched, "new_only_no_E50_comparator": only_new,
            "reference_only": only_ref,
            "rhs_norm_b_p": {"n_matched": len(rhs_keys), "E50": stats([ref_rhs[k] for k in rhs_keys]),
                              "E_star": stats([new_rhs[k] for k in rhs_keys]),
                              "n_below_1e-6_E50": sum(ref_rhs[k] < SATURATION_RHS for k in rhs_keys),
                              "n_below_1e-6_E_star": sum(new_rhs[k] < SATURATION_RHS for k in rhs_keys)}}


def h_and_spectrum(folder):
    folder = Path(folder)
    h_norms, spectra = [], []
    for path in sorted(folder.glob("h_checks_seed*.json")):
        for cls in read_json(path).get("classes", {}).values():
            h_norms.extend(cls.get("h_norms", []))
    for path in sorted(folder.glob("spectrum_seed*.json")):
        row = read_json(path)
        spectra.append({"seed_file": path.name, "min_ritz": row.get("undamped_min_ritz_estimate"),
                        "max_ritz": row.get("undamped_max_ritz_estimate"),
                        "eval_grad_norm": row.get("baseline_eval_gradient_norm")})
    return {"h_norm": stats(h_norms), "eval_grad_norm": stats([r["eval_grad_norm"] for r in spectra]),
            "min_ritz": stats([r["min_ritz"] for r in spectra]), "spectrum": spectra}


def validate_state(root, baseline, mode):
    state = read_json(root / "A_STEP2_STATE.json")
    if state.get("schema") != "pathway2_A_step2_state_v2" or state.get("status") != "prepared_manual_gates":
        raise RuntimeError("Invalid Step-2 state")
    selection = root / "frozen_E_star.json"
    if sha(selection) != state["selection_sha256"]:
        raise RuntimeError("Frozen E* record changed")
    frozen = read_json(selection)
    if (frozen.get("status") != "frozen_human_choice" or
            int(frozen.get("E_star", -1)) != int(state["E_star"]) or
            frozen.get("git_commit") != state["git_commit"]):
        raise RuntimeError("Frozen E* selection differs from the prepared state")
    if Path(state["baseline_root"]).resolve() != baseline.resolve():
        raise RuntimeError("Comparison baseline differs from the prepared baseline")
    panel_src = baseline / "results/cross_stage_calibration_v4_1/panel/eligibility_preflight.json"
    panel_dst = root / "panel/eligibility_preflight.json"
    if (not panel_src.is_file() or not panel_dst.is_file() or
            sha(panel_src) != state["panel_sha256"] or sha(panel_dst) != state["panel_sha256"]):
        raise RuntimeError("Frozen patient panel changed after Step-2 prepare")
    preflight_path = root / "route_a_preflight.json"
    if sha(preflight_path) != state["preflight_sha256"]:
        raise RuntimeError("Prepared preflight record changed")
    preflight = read_json(preflight_path)
    if (preflight.get("status") != "PASS_FILE_INTEGRITY_ONLY" or
            preflight.get("commit") != state["git_commit"] or
            Path(preflight.get("baseline_root", "")).resolve() != baseline.resolve()):
        raise RuntimeError("Prepared source/baseline preflight is not valid for this commit")
    if mode == "full":
        approval = read_json(root / "A2_DAMPING_APPROVAL.json")
        if approval.get("status") != "approved_after_A1_review" or approval.get("git_commit") != state["git_commit"]:
            raise RuntimeError("A2 damping was not approved after A1 review")
        for name, digest in approval["reviewed_sha256"].items():
            if not Path(name).is_file() or sha(name) != digest:
                raise RuntimeError(f"Reviewed A1 artifact changed: {name}")
    return state


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def make_markdown(result):
    e = result["E_star"]
    ladder = result["ladder"]
    common = ladder["all_method_common"]
    lines = [f"# Route A · 50 epochs vs E* = {e}", "",
             f"Mode: **{result['mode']}**. Provenance and matched-cell checks: **PASS**.", "",
             "## A1: original damping (γ_A = 0.2, γ_S = 1)", "",
             f"Every entry below uses the same {common['n']} seed/patient/class cells at both endpoints.", "",
             "| method | Spearman 50 | Spearman E* | MAE 50 | MAE E* | sign 50 | sign E* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        a, b = common["comparisons"][method]["E50"], common["comparisons"][method]["E_star"]
        lines.append(f"| {method} | {fmt(a.get('spearman'))} | {fmt(b.get('spearman'))} | "
                     f"{fmt(a.get('mae'))} | {fmt(b.get('mae'))} | {fmt(a.get('sign_agreement'))} | "
                     f"{fmt(b.get('sign_agreement'))} |")
    d, s = ladder["dtheta_cosine"], ladder["truth_signal"]
    lines += ["", "| diagnostic | 50 epochs | E* |", "|---|---:|---:|",
              f"| median cosine(predicted Δθ, true Δθ), matched n={d['n_matched']} | {fmt(d['E50']['median'])} | {fmt(d['E_star']['median'])} |",
              f"| mean truth J10−J00, matched n={s['n_matched']} | {fmt(s['actual_value_E50']['mean'])} | {fmt(s['actual_value_E_star']['mean'])} |",
              f"| zero-predictor MAE | {fmt(s['zero_predictor_mae_E50'])} | {fmt(s['zero_predictor_mae_E_star'])} |"]
    if result["mode"] == "primary":
        lines += ["", "A2 damping is intentionally absent from this report. Review this A1 result and the truth "
                  "preflight before creating the explicit A2 approval record."]
    else:
        ratios, damping = result["ratios"], result["damping"]
        lines += ["", "## Truth scale / dose checks (same matched cells)", "",
                  "| quantity | median 50 | median E* |", "|---|---:|---:|"]
        labels = {"full_over_theta": "‖Δθ‖/‖θ‖", "full_over_median_seed_distance": "‖Δθ‖/seed distance",
                  "dose01_over_point1_full": "Rα = ‖d0.1‖/(0.1‖d1‖)",
                  "cosine_dose01_full": "cos(d0.1,d1)"}
        for field, label in labels.items():
            row = ratios["fields"][field]
            lines.append(f"| {label} | {fmt(row['E50']['median'])} | {fmt(row['E_star']['median'])} |")
        lines += ["", "## A2 damping (qualified intersection at both epochs)", "",
                  "| condition | γ | matched qualified | median cos 50 | median cos E* | h-proj ρ 50 | h-proj ρ E* |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in damping["matched_overlap"].values():
            a, b = row["E50"], row["E_star"]
            lines.append(f"| {row['condition']} | {row['gamma']:g} | {row['n_matched_qualified']} | "
                         f"{fmt(a['cosine']['median'])} | {fmt(b['cosine']['median'])} | "
                         f"{fmt(a['h_projection'].get('spearman'))} | {fmt(b['h_projection'].get('spearman'))} |")
        if damping["new_only_no_E50_comparator"]:
            lines += ["", "γ=0.01 is a new E*-only sensitivity point; it is reported without pretending that an "
                      "E50 comparator exists: `" + "`, `".join(damping["new_only_no_E50_comparator"]) + "`."]
    lines += ["", "Interpretation guardrail: lower error is meaningful only with coverage, a non-collapsed truth signal, "
              "and parameter-change geometry. This E* sensitivity run does not replace the frozen 50-epoch truth.", ""]
    return "\n".join(lines)


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["ladder"]["plot_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, label in (("E50", "50 epochs"), ("Estar", f"E*={result['E_star']}")):
        actual = [r["actual_E50" if tag == "E50" else "actual_E_star"] for r in rows]
        pred = [r["pred_E50" if tag == "E50" else "pred_E_star"] for r in rows]
        axes[0].scatter(actual, pred, s=16, alpha=.75, color=COLORS[tag], label=label,
                        edgecolors="white", linewidths=.4)
    for index, row in enumerate(rows):
        axes[1].plot([0, 1], [row["dtheta_cosine_E50"], row["dtheta_cosine_E_star"]],
                     color="#b8b8b8", lw=.5, alpha=.5)
        axes[1].scatter([0, 1], [row["dtheta_cosine_E50"], row["dtheta_cosine_E_star"]],
                        color=[COLORS["E50"], COLORS["Estar"]], s=10)
    if result["mode"] == "full":
        pairs = [r for r in result["damping"]["matched_overlap"].values() if r["condition"] == "full"]
        axes[2].plot([r["gamma"] for r in pairs], [r["E50"]["h_projection"].get("spearman") for r in pairs],
                     marker="o", color=COLORS["E50"], label="50 epochs")
        axes[2].plot([r["gamma"] for r in pairs], [r["E_star"]["h_projection"].get("spearman") for r in pairs],
                     marker="o", color=COLORS["Estar"], label=f"E*={result['E_star']}")
        axes[2].set_xscale("log")
    else:
        signal = result["ladder"]["truth_signal"]
        axes[2].bar([0, 1], [signal["zero_predictor_mae_E50"], signal["zero_predictor_mae_E_star"]],
                    color=[COLORS["E50"], COLORS["Estar"]])
    axes[0].axhline(0, color=MUTED, lw=.6); axes[0].axvline(0, color=MUTED, lw=.6)
    axes[1].axhline(0, color=MUTED, lw=.6); axes[1].set_xticks([0, 1], ["50", "E*"])
    titles = ("L3 prediction vs truth (matched cells)", "Δθ cosine, paired cells",
              "h-projection Spearman vs γ" if result["mode"] == "full" else "zero-predictor MAE")
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(color=GRID, lw=.6)
        ax.tick_params(labelsize=7, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["primary", "full"], required=True)
    ap.add_argument("--new_root", required=True)
    ap.add_argument("--baseline_root", required=True)
    ap.add_argument("--ref_full", default=None)
    ap.add_argument("--ref_dose", default=None)
    ap.add_argument("--ref_preflight", default=None)
    ap.add_argument("--ref_damping", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, baseline, out = Path(args.new_root).resolve(), Path(args.baseline_root).resolve(), Path(args.out).resolve()
    expected_out = root / ("compare_primary_vs_E50" if args.mode == "primary" else "compare_full_vs_E50")
    if out != expected_out:
        raise SystemExit(f"Comparison output must be the immutable attempt path: {expected_out}")
    if out.exists():
        raise SystemExit(f"Comparison output already exists: {out}")
    state = validate_state(root, baseline, args.mode)
    ref_full = Path(args.ref_full).resolve() if args.ref_full else baseline / "results/cross_stage_calibration_v4_1/shadow3_full"
    ref_dose = Path(args.ref_dose).resolve() if args.ref_dose else baseline / "results/cross_stage_calibration_v4_1/shadow3_dose01"
    ref_pre = Path(args.ref_preflight).resolve() if args.ref_preflight else baseline / "results/stage1_diagnostics_v11/preflight_20260916T173829Z"
    ref_damp = Path(args.ref_damping).resolve() if args.ref_damping else baseline / "results/stage1_diagnostics_v11/damping_22122145"
    new_full = root / "shadow3_full"
    provenance = {"full": validate_truth_pair(ref_full, new_full, state, "full")}
    ref_score, ref_rows = ladder_rows(ref_full)
    new_score, new_rows = ladder_rows(new_full, require_original_damping=True)
    result = {
        "schema": "pathway2_A_matched_E_comparison_v2", "mode": args.mode,
        "E_star": state["E_star"], "reference_epochs": 50,
        "provenance": {"status": "PASS", "state_sha256": sha(root / "A_STEP2_STATE.json"),
                       "selection_sha256": state["selection_sha256"], "git_commit": state["git_commit"],
                       "truth": provenance, "reference_score": str(ref_score), "new_score": str(new_score)},
        "ladder": matched_ladder(ref_rows, new_rows),
    }
    if args.mode == "full":
        provenance["dose01"] = validate_truth_pair(ref_dose, root / "shadow3_dose01", state, "dose01")
        for folder in (ref_pre, root / "diag_preflight"):
            manifest = read_json(folder / "manifest.json")
            if manifest.get("status") != "complete" or manifest.get("args", {}).get("mode") != "preflight":
                raise RuntimeError(f"Incomplete checkpoint preflight: {folder}")
        result["ratios"] = matched_ratios(ref_pre, root / "diag_preflight")
        result["damping"] = matched_damping(ref_damp, root / "diag_damping")
        result["damping_context"] = {"E50": h_and_spectrum(ref_damp), "E_star": h_and_spectrum(root / "diag_damping")}
        result["provenance"]["truth"] = provenance
    report = make_markdown(result)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.tmp-", dir=root) as temporary:
        staged = Path(temporary)
        (staged / "comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (staged / "COMPARE.md").write_text(report, encoding="utf-8")
        plot(result, staged / "compare_50_vs_Estar")
        staged.rename(out)
    print(report)


if __name__ == "__main__":
    main()
