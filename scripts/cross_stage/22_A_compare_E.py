#!/usr/bin/env python3
"""Route A step 2 readout: original 50-epoch results versus the E*-epoch rerun.

Question from the meeting: once every Stage 1 model is trained to the plateau
epoch E*, does the Level 3 static prediction (Eq. 56, H^-1 b) get closer to the
real parameter change, and does a smaller damping become usable?

Reads only finished outputs (no torch):
  * 07 score ladder   <full>/score_ladder_A0.2_S1/{ladder_summary.json, ladder_rows.csv}
  * 12 preflight      checkpoint_ratios.csv          (scale and alpha-linearity of the truth)
  * 12 damping        all_rows.csv, spectrum_seed*.json, h_checks_seed*.json

Also reports the signal-size quantities Route A requires (||b_p||, ||h||,
near-zero-gradient cells, |J10-J00| versus the zero predictor), so that a
better-conditioned formula is never mistaken for a better prediction of a
signal that disappeared.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

METHODS = ("L1_lin_value", "L2_lin_value", "L2_retrain_value", "L3_lin_value", "L3_retrain_value")
SATURATION_RHS = 1e-6   # descriptive label only: ||b_p|| below this is reported as near-zero gradient
COLORS = {"E50": "#2a78d6", "Estar": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def stats(values):
    values = [v for v in values if finite(v)]
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def spearman(pairs):
    pairs = [(x, y) for x, y in pairs if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, dtype=float).T
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def ladder(full_dir):
    score = Path(full_dir) / "score_ladder_A0.2_S1"
    out = {"score_dir": str(score)}
    summary_path = score / "ladder_summary.json"
    if summary_path.is_file():
        analysis = json.loads(summary_path.read_text(encoding="utf-8"))["analysis"]
        common = analysis.get("value_ladder_common_rows", {})
        out["n_common_rows"] = common.get("n_common")
        out["common_rows"] = {m: common.get("comparisons", {}).get(m) for m in METHODS}
        out["per_patient_mean"] = {m: analysis.get("per_patient_mean", {}).get(f"{m}~actual_value") for m in METHODS}
        out["gate_counts"] = {k: analysis.get(k) for k in
                              ("n_rows_attack_solve_reliable", "n_rows_cg_score_reliable", "n_rows_cg_dtheta_reliable")}
    rows = read_csv(score / "ladder_rows.csv")
    if rows:
        out["dtheta_cosine"] = stats([r.get("dtheta_cosine") for r in rows])
        out["dtheta_cosine_positive"] = sum(1 for r in rows if finite(r.get("dtheta_cosine")) and r["dtheta_cosine"] > 0)
        out["dtheta_true_norm"] = stats([r.get("dtheta_true_norm") for r in rows])
        out["pred_over_true_norm"] = stats([r["dtheta_hat_norm"] / r["dtheta_true_norm"] for r in rows
                                            if finite(r.get("dtheta_hat_norm")) and finite(r.get("dtheta_true_norm"))
                                            and r["dtheta_true_norm"] > 0])
        actual = [r.get("actual_value") for r in rows if finite(r.get("actual_value"))]
        out["actual_value"] = stats(actual)
        out["zero_prediction_mae"] = float(np.mean(np.abs(actual))) if actual else None
        out["rows"] = [{k: r.get(k) for k in ("seed", "patient_id", "oct_class", "actual_value", "L3_lin_value",
                                              "L3_retrain_value", "dtheta_cosine")} for r in rows]
    return out


def ratios(preflight_dir):
    rows = read_csv(Path(preflight_dir) / "checkpoint_ratios.csv") if preflight_dir else None
    if not rows:
        return {"available": False}
    return {"available": True, "n": len(rows),
            "full_over_theta": stats([r.get("full_over_theta") for r in rows]),
            "full_over_median_seed_distance": stats([r.get("full_over_median_seed_distance") for r in rows]),
            "R_alpha_dose01_over_point1_full": stats([r.get("dose01_over_point1_full") for r in rows]),
            "cosine_dose01_full": stats([r.get("cosine_dose01_full") for r in rows])}


def damping(damping_dir):
    folder = Path(damping_dir) if damping_dir else None
    rows = read_csv(folder / "all_rows.csv") if folder else None
    if not rows:
        return {"available": False}
    out = {"available": True, "by_gamma": {}}
    for condition in ("full", "dose01"):
        for gamma in sorted({r["gamma"] for r in rows}):
            use = [r for r in rows if r["condition"] == condition and r["gamma"] == gamma]
            q = [r for r in use if r.get("qualified") is True]
            out["by_gamma"][f"{condition}_gamma{gamma:g}"] = {
                "condition": condition, "gamma": gamma, "n_cells": len(use), "n_qualified": len(q),
                "cosine": stats([r.get("cosine") for r in q]),
                "norm_ratio": stats([r.get("norm_ratio") for r in q]),
                "h_projection_spearman": spearman([(r.get("pred_h_projection"), r.get("true_h_projection")) for r in q])}
    cells = {(r["seed"], r["patient_id"]): r.get("rhs_norm") for r in rows if r["condition"] == "full"}
    out["rhs_norm_b_p"] = stats(list(cells.values()))
    out["n_cells_rhs_below_1e-6"] = sum(1 for v in cells.values() if finite(v) and v < SATURATION_RHS)
    out["n_cells"] = len(cells)
    spectra, h_norms = [], []
    for path in sorted(folder.glob("spectrum_seed*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        spectra.append({"file": path.name, "min_ritz": s.get("undamped_min_ritz_estimate"),
                        "max_ritz": s.get("undamped_max_ritz_estimate"),
                        "eval_grad_norm": s.get("baseline_eval_gradient_norm")})
    for path in sorted(folder.glob("h_checks_seed*.json")):
        for cls in json.loads(path.read_text(encoding="utf-8")).get("classes", {}).values():
            h_norms.extend(cls.get("h_norms", []))
    out["spectrum"] = spectra
    out["min_ritz"] = stats([s["min_ritz"] for s in spectra])
    out["eval_grad_norm"] = stats([s["eval_grad_norm"] for s in spectra])
    out["h_norm"] = stats(h_norms)
    return out


def collect(full_dir, preflight_dir, damping_dir):
    return {"full_dir": str(full_dir), "ladder": ladder(full_dir), "ratios": ratios(preflight_dir),
            "damping": damping(damping_dir)}


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}g}"


def markdown(ref, new, e_star):
    t0 = ref.get("shadow_epochs", 50)
    L = [f"# Route A step 2 · {t0} epochs vs E* = {e_star}", "",
         "Headline question (meeting 2026-09-18): after training to the plateau, does Level 3 move closer to the truth?", "",
         "## 1 Score ladder on common gate-passing rows (γ_A = 0.2, γ_S = 1)", "",
         f"| method | Spearman {t0} | Spearman E* | MAE {t0} | MAE E* | sign agree {t0} | sign agree E* |",
         "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        a, b = g(ref, "ladder", "common_rows", m) or {}, g(new, "ladder", "common_rows", m) or {}
        L.append(f"| {m} | {fmt(a.get('spearman'))} | {fmt(b.get('spearman'))} | {fmt(a.get('mae'))} | "
                 f"{fmt(b.get('mae'))} | {fmt(a.get('sign_agreement'))} | {fmt(b.get('sign_agreement'))} |")
    L += ["", f"Common rows: {t0} → {fmt(g(ref, 'ladder', 'n_common_rows'))}, E* → {fmt(g(new, 'ladder', 'n_common_rows'))}. "
          f"Zero-predictor MAE: {t0} → {fmt(g(ref, 'ladder', 'zero_prediction_mae'))}, "
          f"E* → {fmt(g(new, 'ladder', 'zero_prediction_mae'))}.", "",
          "## 2 Stage 1 parameter change: predicted (H̄+I)⁻¹b vs real Δθ", "",
          f"| quantity | {t0} epochs | E* |", "|---|---|---|"]
    rows = [("cosine(pred, true) median", ("ladder", "dtheta_cosine", "median")),
            ("cosine(pred, true) mean", ("ladder", "dtheta_cosine", "mean")),
            ("cells with positive cosine", ("ladder", "dtheta_cosine_positive")),
            ("‖pred‖/‖true‖ median", ("ladder", "pred_over_true_norm", "median")),
            ("‖Δθ_true‖ median", ("ladder", "dtheta_true_norm", "median")),
            ("‖Δθ‖/‖θ0‖ median", ("ratios", "full_over_theta", "median")),
            ("‖Δθ‖ / seed-to-seed distance median", ("ratios", "full_over_median_seed_distance", "median")),
            ("R_α = ‖d_0.1‖/(0.1‖d_1‖) median (1 = linear)", ("ratios", "R_alpha_dose01_over_point1_full", "median")),
            ("cos(d_0.1, d_1) median", ("ratios", "cosine_dose01_full", "median")),
            ("‖∇L_eval(θ0)‖ median over seeds", ("damping", "eval_grad_norm", "median")),
            ("min Ritz of H̄ (min over seeds)", ("damping", "min_ritz", "min")),
            ("‖b_p‖ median", ("damping", "rhs_norm_b_p", "median")),
            ("cells with ‖b_p‖ < 1e-6", ("damping", "n_cells_rhs_below_1e-6")),
            ("‖h‖ median", ("damping", "h_norm", "median")),
            ("mean J10−J00 (truth signal)", ("ladder", "actual_value", "mean"))]
    for label, path in rows:
        L.append(f"| {label} | {fmt(g(ref, *path))} | {fmt(g(new, *path))} |")
    L += ["", "## 3 Damping sweep (v1.1 Lanczos-MINRES; each γ on its own qualified cells)", "",
          f"| condition | γ | qualified {t0} | qualified E* | median cos {t0} | median cos E* | h-proj Spearman {t0} | h-proj Spearman E* |",
          "|---|---|---|---|---|---|---|---|"]
    keys = sorted(set((g(ref, "damping", "by_gamma") or {}).keys()) | set((g(new, "damping", "by_gamma") or {}).keys()),
                  key=lambda k: (k.split("_gamma")[0], float(k.split("_gamma")[1])))
    for k in keys:
        a, b = g(ref, "damping", "by_gamma", k) or {}, g(new, "damping", "by_gamma", k) or {}
        cond, gamma = k.split("_gamma")
        L.append(f"| {cond} | {gamma} | {fmt(a.get('n_qualified'))}/{fmt(a.get('n_cells'))} | "
                 f"{fmt(b.get('n_qualified'))}/{fmt(b.get('n_cells'))} | {fmt(g(a, 'cosine', 'median'))} | "
                 f"{fmt(g(b, 'cosine', 'median'))} | {fmt(a.get('h_projection_spearman'))} | {fmt(b.get('h_projection_spearman'))} |")
    L += ["", "Reading guide: qualified cohorts differ between γ values and between 50 and E*; compare Spearman values "
          "only together with their coverage. A lower MAE than the zero predictor, a clearly positive Δθ cosine and "
          "R_α closer to 1 are the three signs that the static premise now holds. If ‖b_p‖ or |J10−J00| collapses at "
          "E*, report the signal as unresolved rather than the formula as improved.", ""]
    return "\n".join(L)


def plot(ref, new, e_star, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, data in (("E50", ref), ("Estar", new)):
        rows = g(data, "ladder", "rows") or []
        name = f"{ref.get('shadow_epochs', 50)} epochs" if label == "E50" else f"E* = {e_star}"
        pts = [(r["actual_value"], r["L3_lin_value"]) for r in rows if finite(r.get("actual_value")) and finite(r.get("L3_lin_value"))]
        if pts:
            x, yv = np.asarray(pts).T
            axes[0].scatter(x, yv, s=16, color=COLORS[label], alpha=.8, label=name, edgecolors="white", linewidths=.5)
        cos = [r["dtheta_cosine"] for r in rows if finite(r.get("dtheta_cosine"))]
        if cos:
            pos = 0 if label == "E50" else 1
            axes[1].scatter(np.full(len(cos), pos) + np.random.default_rng(pos).uniform(-.12, .12, len(cos)), cos,
                            s=14, color=COLORS[label], alpha=.8, label=name)
        by = g(data, "damping", "by_gamma") or {}
        pts = sorted((v["gamma"], v["h_projection_spearman"]) for v in by.values()
                     if v["condition"] == "full" and v.get("h_projection_spearman") is not None)
        if pts:
            axes[2].plot(*zip(*pts), marker="o", ms=5, lw=1.6, color=COLORS[label], label=name)
    axes[0].axhline(0, color=MUTED, lw=.7)
    axes[0].axvline(0, color=MUTED, lw=.7)
    axes[1].axhline(0, color=MUTED, lw=.7)
    axes[1].set_xticks([0, 1], [f"{ref.get('shadow_epochs', 50)} epochs", f"E* = {e_star}"])
    if not axes[2].lines:
        axes[2].text(.5, .5, "no γ with ≥3 qualified cells", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=8, color=MUTED)
    axes[2].set_xscale("log")
    for ax, title, xlabel in ((axes[0], "L3 linear prediction vs real J10−J00", "real J10 − J00"),
                              (axes[1], "cosine(predicted Δθ, real Δθ) per cell", ""),
                              (axes[2], "h-projection Spearman vs damping γ (full)", "γ")):
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color=GRID, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("L3 linear score", fontsize=8, color=MUTED)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new_root", required=True, help="l3_at_E<E> directory written by submit_A_step2.sh")
    ap.add_argument("--baseline_root", required=True, help="frozen oct2-calibration-v4 tree")
    ap.add_argument("--ref_full", default=None)
    ap.add_argument("--ref_preflight", default=None)
    ap.add_argument("--ref_damping", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    base, root = Path(args.baseline_root), Path(args.new_root)
    ref_full = Path(args.ref_full) if args.ref_full else base / "results/cross_stage_calibration_v4_1/shadow3_full"
    ref_pre = args.ref_preflight or next(iter(sorted((base / "results/stage1_diagnostics_v11").glob("preflight_*"))), None)
    ref_damp = args.ref_damping or base / "results/stage1_diagnostics_v11/damping_22122145"
    fulls = sorted(root.glob("shadow*_full"))
    if len(fulls) != 1:
        raise SystemExit(f"Expected exactly one shadow*_full directory under {root}, found {fulls}")
    config = json.loads((fulls[0] / "experiment_config.json").read_text(encoding="utf-8"))
    e_star = config["args"]["shadow_epochs"]
    ref_config = json.loads((ref_full / "experiment_config.json").read_text(encoding="utf-8"))
    for key in ("seeds", "attack_seeds", "affected_shadow", "split_seed", "deletion_mode", "shadow_lr",
                "shadow_batch_size", "attack_epochs", "n_total_samples", "n_shadow"):
        if ref_config["args"].get(key) != config["args"].get(key):
            raise SystemExit(f"Reference and E* runs differ in {key}; they are not comparable")
    ref = collect(ref_full, ref_pre, ref_damp)
    new = collect(fulls[0], root / "diag_preflight", root / "diag_damping")
    ref["shadow_epochs"], new["shadow_epochs"] = ref_config["args"]["shadow_epochs"], e_star
    out = Path(args.out) if args.out else root / "compare_vs_E50"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({"E_star": e_star, "reference_50": ref, "E_star_run": new},
                                                    indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "COMPARE.md").write_text(markdown(ref, new, e_star), encoding="utf-8")
    plot(ref, new, e_star, out / "compare_50_vs_Estar")
    print((out / "COMPARE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
