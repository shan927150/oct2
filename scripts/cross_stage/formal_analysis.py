"""Pure NumPy/SciPy analysis shared by the calibration scripts.

Numeric solver gates follow the computational dependencies of each layer.
Seed variance is conditional on one fixed patient, split, and affected shadow.
"""
from __future__ import annotations

import numpy as np


GATES = {
    "L1_lin_value": ("attack_solve_reliable",),
    "L2_lin_value": ("attack_solve_reliable",),
    "L2_retrain_value": (),
    "L2_hybrid_full": (),
    "L3_lin_value": ("attack_solve_reliable", "cg_score_reliable"),
    "L3_retrain_value": ("cg_dtheta_reliable",),
    "L3_hybrid_full": ("cg_dtheta_reliable",),
    "frozen_h": ("attack_solve_reliable",),
    "frozen_self": (),
    "actual_relabel": (),
}


def finite(value):
    return value is not None and bool(np.isfinite(value))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def comparison(xs, ys):
    from scipy.stats import pearsonr, spearmanr
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    out = {"n": len(pairs), "spearman": None, "pearson": None,
           "mae": None, "sign_agreement": None}
    if not pairs:
        return out
    x, y = np.asarray(pairs, dtype=float).T
    out.update(mae=float(np.mean(np.abs(x - y))),
               sign_agreement=float(np.mean(np.sign(x) == np.sign(y))),
               pred_mean=float(x.mean()), actual_mean=float(y.mean()),
               pred_sd=float(x.std(ddof=1)) if len(x) > 1 else None,
               actual_sd=float(y.std(ddof=1)) if len(y) > 1 else None)
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        out.update(spearman=float(spearmanr(x, y).statistic),
                   pearson=float(pearsonr(x, y).statistic),
                   slope_actual_on_pred=float(slope), intercept=float(intercept))
    return clean_json(out)


def passes(row, key):
    # Missing gate metadata must not silently certify an inverse-based score.
    return all(row.get(gate) is True for gate in GATES[key])


def analyze_ladder(rows, expected_seeds=None):
    expected = set(map(int, expected_seeds or sorted({r["seed"] for r in rows})))
    pairs = [(k, "actual_value") for k in (
        "L1_lin_value", "L2_lin_value", "L2_retrain_value", "L3_lin_value",
        "L3_retrain_value", "frozen_h", "frozen_self")]
    pairs += [(k, "actual_full") for k in (
        "actual_relabel", "L2_hybrid_full", "L3_hybrid_full")]
    out = {"n_rows": len(rows), "expected_stage1_seeds": sorted(expected),
           "per_patient_seed": {}, "per_patient_mean": {}, "per_class": {},
           "note": "Descriptive calibration. CE J10-J00 is the primary estimand. "
                   "Patient means require the complete specified Stage-1 seed panel; "
                   "pooled rows are dependent, so rowwise p-values are not reported. "
                   "Positive Lanczos Ritz values are diagnostics, not an SPD proof."}

    def patient_key(r):
        return (r.get("affected_shadow", 0), r["oct_class"], r["patient_id"])

    def summarize(input_rows, pk, ak):
        numeric = [r for r in input_rows if finite(r.get(pk)) and finite(r.get(ak))]
        use = [r for r in numeric if passes(r, pk)]
        coverage = {"n_total": len(input_rows), "n_finite": len(numeric),
                    "n_gate_passed": len(use), "required_gates": list(GATES[pk])}
        groups = {}
        for r in use:
            groups.setdefault(patient_key(r), []).append(r)
        complete = [rs for rs in groups.values()
                    if len(rs) == len(expected) and {int(r["seed"]) for r in rs} == expected]
        means = comparison([np.mean([r[pk] for r in rs]) for rs in complete],
                           [np.mean([r[ak] for r in rs]) for rs in complete])
        means.update(n_patients_total=len({patient_key(r) for r in input_rows}),
                     n_patients_complete=len(complete), expected_seeds=sorted(expected))
        return {**comparison([r[pk] for r in use], [r[ak] for r in use]), **coverage}, means

    for pk, ak in pairs:
        label = f"{pk}~{ak}"
        out["per_patient_seed"][label], out["per_patient_mean"][label] = summarize(rows, pk, ak)
        for cls in sorted({r["oct_class"] for r in rows}):
            cr = [r for r in rows if r["oct_class"] == cls]
            a, b = summarize(cr, pk, ak)
            out["per_class"].setdefault(str(cls), {})[label] = {
                "per_patient_seed": a, "per_patient_mean": b}

    # Fair comparisons use the SAME rows; relabel-only is the full-effect baseline.
    out["hybrid_increment_over_relabel"] = {}
    for pk in ("L2_hybrid_full", "L3_hybrid_full"):
        use = [r for r in rows if passes(r, pk) and all(
            finite(r.get(k)) for k in (pk, "actual_relabel", "actual_full"))]
        base = comparison([r["actual_relabel"] for r in use], [r["actual_full"] for r in use])
        hybrid = comparison([r[pk] for r in use], [r["actual_full"] for r in use])
        out["hybrid_increment_over_relabel"][pk] = {
            "n_common": len(use), "relabel_only": base, "hybrid": hybrid,
            "mae_reduction": base["mae"] - hybrid["mae"] if use else None}
    methods = ["L1_lin_value", "L2_lin_value", "L2_retrain_value", "L3_lin_value", "L3_retrain_value"]
    common = [r for r in rows if finite(r.get("actual_value")) and all(
        finite(r.get(k)) and passes(r, k) for k in methods)]
    out["value_ladder_common_rows"] = {
        "n_common": len(common), "comparisons": {
            k: comparison([r[k] for r in common], [r["actual_value"] for r in common]) for k in methods}}
    out["damping_sensitivity"] = {}
    for key in sorted({k for r in rows for k in r if k.startswith("L3_lin_value_gamma") and not k.endswith("_reliable")}):
        use = [r for r in rows if passes(r, "L3_lin_value") and r.get(key + "_reliable") is True
               and finite(r.get(key)) and finite(r.get("L3_lin_value"))]
        out["damping_sensitivity"][key] = comparison(
            [r[key] for r in use], [r["L3_lin_value"] for r in use])
    for gate in ("attack_solve_reliable", "cg_score_reliable", "cg_dtheta_reliable"):
        out["n_rows_" + gate] = sum(r.get(gate) is True for r in rows)
    return clean_json(out)


def seed_variance_components(matrix, design):
    """Balanced crossed or nested ANOVA, one observation per seed cell.

    Interaction and other cell residuals cannot be separated without additional
    within-cell replication. Negative method-of-moments estimates are preserved;
    nonnegative components are descriptive truncations, not REML estimates.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or min(x.shape) < 1:
        raise ValueError("variance analysis requires a complete finite R x K matrix")
    R, K = x.shape
    out = {"R": R, "K": K, "mean_paired_delta": float(x.mean()), "design": design,
           "status": "insufficient_replication", "variance_grand_mean": None,
           "se_grand_mean": None, "raw_components": None, "components_nonnegative": None}
    if R < 2:
        out["conditional_attack_sd"] = float(x.std(ddof=1)) if K > 1 else None
        return out
    between = float(np.var(x.mean(axis=1), ddof=1))
    out["stage1_rowmean_sd"] = float(np.sqrt(between))
    if K < 2:
        out.update(status="combined_only_K1", variance_grand_mean=between / R,
                   se_grand_mean=float(np.sqrt(between / R)),
                   note="K=1 cannot identify Stage-1 and attack components; SD is combined.")
        return out
    if design == "crossed_fixed_panel":
        row, col, grand = x.mean(1), x.mean(0), float(x.mean())
        ms_row = K * between
        ms_col = R * float(np.var(col, ddof=1))
        ms_error = float(np.square(x - row[:, None] - col[None, :] + grand).sum() / ((R-1)*(K-1)))
        raw = {"stage1": (ms_row-ms_error)/K, "attack": (ms_col-ms_error)/R,
               "interaction_and_cell_residual": ms_error}
        comp = {k: max(0., v) for k, v in raw.items()}
        terms = {"stage1": comp["stage1"]/R, "attack": comp["attack"]/K,
                 "interaction_and_cell_residual": comp["interaction_and_cell_residual"]/(R*K)}
        out["mean_squares"] = {"stage1": ms_row, "attack": ms_col, "residual": ms_error}
        out["variance_formula"] = "sigma_S^2/R + sigma_A^2/K + sigma_SA^2/(R*K)"
    elif design == "nested_derived_from_stage1_seed":
        within = float(np.var(x, axis=1, ddof=1).mean())
        raw = {"stage1": between-within/K, "attack_within_stage1": within}
        comp = {k: max(0., v) for k, v in raw.items()}
        terms = {"stage1": comp["stage1"]/R, "attack_within_stage1": within/(R*K)}
        out["variance_formula"] = "sigma_S^2/R + sigma_A_within_S^2/(R*K)"
    else:
        raise ValueError(f"unknown seed design: {design}")
    variance = sum(terms.values())
    out.update(status="descriptive_anova", raw_components=raw, components_nonnegative=comp,
               negative_component_estimates=[k for k, v in raw.items() if v < 0],
               variance_contributions_to_grand_mean=terms, variance_grand_mean=variance,
               se_grand_mean=float(np.sqrt(variance)),
               note="Conditional on this patient/split/shadow and fixed target queries. "
                    "Nonnegative truncation; small seed panels give uncertain component estimates. "
                    "This is seed uncertainty, not patient/query population uncertainty.")
    return clean_json(out)
