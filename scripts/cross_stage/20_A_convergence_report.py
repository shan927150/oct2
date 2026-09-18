#!/usr/bin/env python3
"""Collect 20_A_convergence_curve.py outputs: curves, 10-epoch windows, plateau rule, spectrum.

Pure NumPy + matplotlib (no torch), so it can run on a login node.

Plateau rule (fixed before any extended result was seen, documented in
docs/PATHWAY2_A_CONVERGENCE_RUNBOOK_CN.md):

  * 10-epoch windows (k-10, k], k = 10, 20, ..., E_max.
  * Window mean m_k of the per-epoch online training CE (the curve discussed
    in the meeting) and its standard error se_k (SD / sqrt(n) inside the window).
    The pair (k, k+10) is "flat" when
        |m_k - m_{k+10}| <= max(rel_tol * m_k, abs_tol, z * sqrt(se_k^2 + se_{k+10}^2))
    with rel_tol=0.10, abs_tol=0.002, z=2: the change is below 10 % or not
    distinguishable from the epoch-to-epoch minibatch noise.
  * A run plateaus from k (k >= T0) when every later pair is flat and at least
    two pairs are observed (so k <= E_max - 20).
  * Suggested E* = max over the affected-shadow seeds (every colour flat).
    None means "not established within E_max".

The same rule is reported for the eval objective (CE + L2 on the full training
set, dropout off).  The gradient norm of that objective is reported next to
it, because a flat loss is not by itself the Eq. 56 stationarity premise.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "pathway2_A_convergence_curve_v1"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if value in ("", None):
                parsed[key] = None
                continue
            try:
                parsed[key] = int(value) if key in ("epoch", "seed") else float(value)
            except ValueError:
                parsed[key] = {"True": True, "False": False}.get(value, value)
        out.append(parsed)
    return out


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def windows(rows, field, width, e_max, stat="mean", with_se=False):
    out, se = {}, {}
    for k in range(width, e_max + 1, width):
        values = [r[field] for r in rows if k - width < r["epoch"] <= k and finite(r.get(field))]
        if values:
            out[k] = float(np.mean(values) if stat == "mean" else np.median(values))
            se[k] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.
    return (out, se) if with_se else out


def plateau_start(win, t0, e_max, width, rel_tol, abs_tol, min_pairs=2, se=None, z=2.):
    """Smallest k >= t0 from which every consecutive window pair is flat."""
    se = se or {}
    flat = {}
    for k in sorted(win):
        if k + width in win:
            noise = z * math.sqrt(se.get(k, 0.) ** 2 + se.get(k + width, 0.) ** 2)
            flat[k] = abs(win[k] - win[k + width]) <= max(rel_tol * abs(win[k]), abs_tol, noise)
    for k in sorted(win):
        if k < t0:
            continue
        tail = [flat[j] for j in sorted(flat) if j >= k]
        if len(tail) >= min_pairs and all(tail):
            return k, flat
    return None, flat


def load_runs(root):
    runs, problems = {}, []
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            continue
        name = manifest["run"]
        if manifest.get("status") != "complete":
            problems.append(f"{name}: status={manifest.get('status')} error={manifest.get('error')}")
            continue
        folder = manifest_path.parent
        curve = read_csv(folder / "curve.csv")
        spectrum = read_csv(folder / "spectrum.csv") if (folder / "spectrum.csv").exists() else []
        runs[name] = {"manifest": manifest, "curve": curve, "spectrum": spectrum}
    expected = set()
    for run in runs.values():
        expected.update(run["manifest"]["all_runs"])
    missing = sorted(expected - set(runs))
    return runs, problems, missing


def summarize(runs, width, rel_tol, abs_tol, obj_abs_tol, z=2.):
    any_manifest = next(iter(runs.values()))["manifest"]
    t0, e_max = any_manifest["original_epochs"], any_manifest["extended_epochs"]
    table, per_run = [], {}
    for name, run in runs.items():
        curve = run["curve"]
        epochs = sorted(r["epoch"] for r in curve)
        if epochs != list(range(0, e_max + 1)):
            raise RuntimeError(f"{name}: curve epochs are not 0..{e_max}")
        fields = {"online_train_ce": "mean", "eval_objective": "median", "eval_grad_norm": "median",
                  "relative_epoch_update": "median", "heldout_ce": "mean", "heldout_accuracy": "mean",
                  "generalization_gap_ce": "mean", "interface_tv_prev_mean": "median"}
        w = {f: windows(curve, f, width, e_max, s) for f, s in fields.items()}
        ce_mean, ce_se = windows(curve, "online_train_ce", width, e_max, "mean", with_se=True)
        obj_mean, obj_se = windows(curve, "eval_objective", width, e_max, "mean", with_se=True)
        ce_start, ce_flat = plateau_start(ce_mean, t0, e_max, width, rel_tol, abs_tol, se=ce_se, z=z)
        obj_start, obj_flat = plateau_start(obj_mean, t0, e_max, width, rel_tol, obj_abs_tol, se=obj_se, z=z)
        g = w["eval_grad_norm"]
        by_epoch = {r["epoch"]: r for r in curve}
        per_run[name] = {
            "role": run["manifest"]["role"], "seed": run["manifest"]["seed"],
            "replay_exact": run["manifest"].get("replay_exact"),
            "plateau_start_online_ce": ce_start, "plateau_start_eval_objective": obj_start,
            "online_ce_pair_flat": {str(k): v for k, v in ce_flat.items()},
            "grad_norm_window_T0": g.get(t0), "grad_norm_window_Emax": g.get(e_max),
            "grad_norm_ratio_Emax_over_T0": (g[e_max] / g[t0]) if g.get(t0) and g.get(e_max) else None,
            "grad_norm_min_window": min(g.values()) if g else None,
            "interface_tv_from_T0_at_Emax": by_epoch[e_max].get("interface_tv_from_T0_mean"),
            "epoch_T0": {k: by_epoch[t0].get(k) for k in ("online_train_ce", "eval_objective", "eval_grad_norm",
                                                          "relative_epoch_update", "heldout_ce", "heldout_accuracy")},
            "epoch_Emax": {k: by_epoch[e_max].get(k) for k in ("online_train_ce", "eval_objective", "eval_grad_norm",
                                                               "relative_epoch_update", "heldout_ce", "heldout_accuracy")},
        }
        for k in sorted(w["online_train_ce"]):
            table.append({"run": name, "role": run["manifest"]["role"], "window_end": k,
                          **{f"{f}_{s}": w[f].get(k) for f, s in fields.items()},
                          "online_ce_flat_vs_next": ce_flat.get(k)})
    affected = [n for n, r in per_run.items() if r["role"] == "affected_shadow"]
    starts = [per_run[n]["plateau_start_online_ce"] for n in affected]
    suggested = max(starts) if affected and all(s is not None for s in starts) else None
    others_flat_by = {n: (r["plateau_start_online_ce"] is not None and suggested is not None
                          and r["plateau_start_online_ce"] <= suggested)
                      for n, r in per_run.items() if r["role"] != "affected_shadow"}
    return {"original_epochs": t0, "extended_epochs": e_max, "window": width,
            "rule": {"series": "online_train_ce window mean", "rel_tol": rel_tol, "abs_tol": abs_tol,
                     "noise_z": z, "eval_objective_abs_tol": obj_abs_tol, "min_pairs": 2,
                     "suggestion": "max over affected-shadow seeds of the plateau start"},
            "suggested_E_star": suggested,
            "suggested_E_star_reason": ("all affected seeds flat" if suggested is not None else
                                        "plateau not established within E_max for: " +
                                        ", ".join(n for n in affected if per_run[n]["plateau_start_online_ce"] is None)),
            "fixed_models_flat_by_E_star": others_flat_by, "runs": per_run}, table


def spectrum_table(runs):
    rows = []
    for name, run in runs.items():
        for r in run["spectrum"]:
            rows.append({"run": name, **{k: r.get(k) for k in ("seed", "epoch", "min_ritz", "max_ritz", "damping_floor",
                                                                 "eval_grad_norm_07", "min_ritz_residual_scaled_max",
                                                                 "v11_min_ritz", "v11_max_ritz", "v11_grad_norm")}})
    return sorted(rows, key=lambda r: (r["run"], r["epoch"]))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax, title, xlabel="Epoch"):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.tick_params(which="both", labelsize=7, colors=MUTED, length=2)
    ax.grid(color=GRID, linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_curves(runs, names, summary, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("online_train_ce", "Online training CE (dropout on; y zoomed to the tail)", False),
              ("eval_objective", "Eval objective: CE + L2, full train set (log)", True),
              ("eval_grad_norm", "‖∇ eval objective‖ — Eq. 56 needs ≈ 0 (log)", True),
              ("relative_epoch_update", "Relative parameter update per epoch", False),
              ("heldout_ce", "Held-out CE (non-members)", False),
              ("interface_tv_prev_mean", "Interface drift: mean TV to previous epoch (log)", True)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    t0, e_star = summary["original_epochs"], summary["suggested_E_star"]
    for ax, (field, label, log) in zip(axes.flat, panels):
        for i, name in enumerate(names):
            rows = [r for r in runs[name]["curve"] if finite(r.get(field)) and (not log or r[field] > 0)]
            ax.plot([r["epoch"] for r in rows], [r[field] for r in rows], lw=1.4,
                    color=SERIES_COLORS[i % len(SERIES_COLORS)], label=name)
        if field == "online_train_ce":
            tail = [r[field] for n in names for r in runs[n]["curve"]
                    if finite(r.get(field)) and t0 - 10 < r["epoch"] <= t0]
            if tail:
                ax.set_ylim(0, 4 * float(np.mean(tail)))
        ax.axvline(t0, color=MUTED, lw=.9, ls="--")
        if e_star is not None and e_star != t0:
            ax.axvline(e_star, color=INK, lw=.9, ls=":")
        if log:
            ax.set_yscale("log")
        style_axis(ax, label)
    axes[0, 0].legend(fontsize=7, frameon=False)
    note = f"dashed = original {t0} epochs"
    note += f"; dotted = suggested E* = {e_star}" if e_star is not None else "; no plateau E* under the rule"
    fig.suptitle(f"{title}   ({note})", fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def plot_spectrum(rows, path):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, name in enumerate(sorted({r["run"] for r in rows})):
        use = [r for r in rows if r["run"] == name]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        axes[0].plot([r["epoch"] for r in use], [r["min_ritz"] for r in use], marker="o", ms=4, lw=1.4, color=color, label=name)
        axes[1].plot([r["epoch"] for r in use], [r["max_ritz"] for r in use], marker="o", ms=4, lw=1.4, color=color)
        axes[2].plot([r["epoch"] for r in use], [r["eval_grad_norm_07"] for r in use], marker="o", ms=4, lw=1.4, color=color)
    axes[0].axhline(0, color=MUTED, lw=.8)
    style_axis(axes[0], "Smallest Ritz value of H̄ (damping must exceed −this)")
    style_axis(axes[1], "Largest Ritz value of H̄")
    style_axis(axes[2], "‖∇ eval objective‖ at saved checkpoints")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.{digits}g}"


def markdown(summary, spec_rows, problems, missing):
    t0, e_max = summary["original_epochs"], summary["extended_epochs"]
    lines = [f"# Route A step 1 · Stage 1 extended to {e_max} epochs", "",
             f"Suggested E* (online-CE plateau rule, all affected seeds): **{fmt(summary['suggested_E_star'])}** — "
             f"{summary['suggested_E_star_reason']}", ""]
    if problems or missing:
        lines += ["**Incomplete runs:** " + "; ".join(problems + [f"{m}: missing" for m in missing]), ""]
    lines += ["| run | replay exact | plateau (online CE) | plateau (eval obj) | online CE @T0 → @Emax | "
              "eval obj @T0 → @Emax | grad norm window T0 → Emax | held-out CE @T0 → @Emax |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in summary["runs"].items():
        a, b = r["epoch_T0"], r["epoch_Emax"]
        lines.append(f"| {name} | {fmt(r['replay_exact'])} | {fmt(r['plateau_start_online_ce'])} | "
                     f"{fmt(r['plateau_start_eval_objective'])} | {fmt(a['online_train_ce'])} → {fmt(b['online_train_ce'])} | "
                     f"{fmt(a['eval_objective'])} → {fmt(b['eval_objective'])} | "
                     f"{fmt(r['grad_norm_window_T0'])} → {fmt(r['grad_norm_window_Emax'])} | "
                     f"{fmt(a['heldout_ce'])} → {fmt(b['heldout_ce'])} |")
    if spec_rows:
        lines += ["", "Hessian Ritz endpoints (H̄ = eval-CE Hessian + wd·I):", "",
                  "| run | epoch | min Ritz | max Ritz | damping floor | grad norm |", "|---|---|---|---|---|---|"]
        for r in spec_rows:
            lines.append(f"| {r['run']} | {r['epoch']} | {fmt(r['min_ritz'])} | {fmt(r['max_ritz'])} | "
                         f"{fmt(r['damping_floor'])} | {fmt(r['eval_grad_norm_07'])} |")
    lines += ["", "Reading guide: a flat online CE is the meeting's plateau criterion. Eq. 56 additionally needs the "
              "full-batch gradient of the eval objective to be small; compare the grad-norm column before calling the "
              "endpoint stationary. The damping floor is the smallest γ that makes H̄+γI positive on the Krylov estimate."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="out_root used by 20_A_convergence_curve.py")
    ap.add_argument("--out", default=None, help="report directory (default <root>/report)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--rel_tol", type=float, default=.10)
    ap.add_argument("--abs_tol", type=float, default=.002)
    ap.add_argument("--objective_abs_tol", type=float, default=.001)
    ap.add_argument("--noise_z", type=float, default=2.)
    ap.add_argument("--allow_incomplete", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / "report"
    runs, problems, missing = load_runs(root)
    if not runs:
        raise SystemExit(f"No completed runs under {root}")
    if (problems or missing) and not args.allow_incomplete:
        raise SystemExit("Incomplete: " + "; ".join(problems + [f"{m}: missing" for m in missing]) +
                         "\nRe-run the failed tasks or pass --allow_incomplete for a partial look.")
    out.mkdir(parents=True, exist_ok=True)
    summary, table = summarize(runs, args.window, args.rel_tol, args.abs_tol, args.objective_abs_tol, args.noise_z)
    summary.update(problems=problems, missing=missing)
    spec_rows = spectrum_table(runs)
    write_csv(out / "window_table.csv", table)
    write_csv(out / "spectrum_table.csv", spec_rows)
    (out / "plateau_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    affected = sorted(n for n, r in summary["runs"].items() if r["role"] == "affected_shadow")
    others = sorted(n for n, r in summary["runs"].items() if r["role"] != "affected_shadow")
    if affected:
        plot_curves(runs, affected, summary, out / "curves_affected_shadow", "Affected shadow, one colour per Stage 1 seed")
    if others:
        plot_curves(runs, others, summary, out / "curves_target_and_fixed_shadows", "Target and unaffected shadows")
    plot_spectrum(spec_rows, out / "spectrum")
    (out / "REPORT.md").write_text(markdown(summary, spec_rows, problems, missing), encoding="utf-8")
    print(json.dumps({"suggested_E_star": summary["suggested_E_star"], "reason": summary["suggested_E_star_reason"],
                      "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
