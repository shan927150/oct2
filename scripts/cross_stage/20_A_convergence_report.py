#!/usr/bin/env python3
"""Collect Route A Step-1 outputs into descriptive curves and window summaries.

Pure NumPy + matplotlib (no torch), so it can run on a login node.

The report deliberately does not label a plateau or choose E*.  Ten-epoch
means, slopes, ranges, parameter updates and gradient diagnostics are shown so
the team can review all five colours and then freeze exactly one checkpoint in
a separate, auditable selection record.  Sequential epoch values are not
treated as independent samples for an automatic statistical stopping rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "pathway2_A_convergence_curve_v2"
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


def window_shape(rows, field, width, e_max):
    """Descriptive within-window slope and range; never used to select E*."""
    out = {}
    for k in range(width, e_max + 1, width):
        points = [(r["epoch"], r.get(field)) for r in rows
                  if k - width < r["epoch"] <= k and finite(r.get(field))]
        if not points:
            continue
        x, y = np.asarray(points, dtype=float).T
        out[k] = {
            "slope_per_epoch": (float(np.polyfit(x, y, 1)[0]) if len(points) > 1 else None),
            "range": float(np.ptp(y)),
        }
    return out


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
        runs[name] = {"manifest": manifest, "curve": curve, "spectrum": spectrum,
                      "folder": str(folder)}
    expected = set()
    for run in runs.values():
        expected.update(run["manifest"]["all_runs"])
    missing = sorted(expected - set(runs))
    return runs, problems, missing


def validate_panel(runs):
    """Reject mixed commits, inputs, data snapshots or observation settings."""
    if not runs:
        return
    manifests = [r["manifest"] for r in runs.values()]
    for field in ("git_commit", "original_epochs", "extended_epochs", "all_runs", "source_sha256"):
        values = {json.dumps(m.get(field), sort_keys=True) for m in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Completed runs disagree on {field}")
    controlled = ("epochs", "checkpoint_epochs", "gradient_every", "dropout_mc_every",
                  "dropout_mc_reps", "hvp_batch", "eval_batch", "spectrum_epochs")
    for field in controlled:
        values = {json.dumps(m.get("args", {}).get(field), sort_keys=True) for m in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Completed runs disagree on argument {field}")
    for name, run in runs.items():
        m = run["manifest"]
        for flag in ("replay_exact", "input_files_unchanged", "source_files_unchanged",
                     "dataset_arrays_unchanged"):
            if m.get(flag) is not True:
                raise RuntimeError(f"{name}: required integrity flag {flag} is not true")
    dataset_rows = []
    input_rows = []
    for run in runs.values():
        folder = Path(run["folder"])
        dataset_rows.append(json.loads((folder / "dataset_fingerprint.json").read_text()))
        input_rows.append(json.loads((folder / "input_sha256.json").read_text()))
    if len({json.dumps(v, sort_keys=True) for v in dataset_rows}) != 1:
        raise RuntimeError("Completed runs used different dataset arrays")
    # Common inputs must have identical hashes. Seed-specific checkpoint/order paths may differ.
    common_paths = set(input_rows[0])
    for row in input_rows[1:]:
        common_paths &= set(row)
    for path in common_paths:
        if len({row[path] for row in input_rows}) != 1:
            raise RuntimeError(f"Completed runs disagree on common input {path}")


def summarize(runs, width):
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
        shapes = {f: window_shape(curve, f, width, e_max)
                  for f in ("online_train_ce", "eval_objective")}
        g = w["eval_grad_norm"]
        by_epoch = {r["epoch"]: r for r in curve}
        per_run[name] = {
            "role": run["manifest"]["role"], "seed": run["manifest"]["seed"],
            "replay_exact": run["manifest"].get("replay_exact"),
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
            previous = k - width
            current_ce = w["online_train_ce"].get(k)
            previous_ce = w["online_train_ce"].get(previous)
            change = current_ce - previous_ce if current_ce is not None and previous_ce is not None else None
            table.append({"run": name, "role": run["manifest"]["role"], "window_end": k,
                          **{f"{f}_{s}": w[f].get(k) for f, s in fields.items()},
                          "online_train_ce_slope_per_epoch": shapes["online_train_ce"].get(k, {}).get("slope_per_epoch"),
                          "online_train_ce_range": shapes["online_train_ce"].get(k, {}).get("range"),
                          "eval_objective_slope_per_epoch": shapes["eval_objective"].get(k, {}).get("slope_per_epoch"),
                          "eval_objective_range": shapes["eval_objective"].get(k, {}).get("range"),
                          "online_ce_change_from_previous_window": change,
                          "online_ce_relative_change_from_previous_window":
                              (change / abs(previous_ce)) if change is not None and previous_ce else None})
    return {"original_epochs": t0, "extended_epochs": e_max, "window": width,
            "selection_status": "not_frozen",
            "selected_E_star": None,
            "selection_policy": "Descriptive only. Review all five curves, then create frozen_E_star.json separately.",
            "runs": per_run}, table


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
    t0 = summary["original_epochs"]
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
        if log:
            ax.set_yscale("log")
        style_axis(ax, label)
    axes[0, 0].legend(fontsize=7, frameon=False)
    note = f"dashed = original {t0} epochs; no E* selected by code"
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
             "**E* is not selected by this report.** Review all five seed curves and diagnostics, then create a "
             "separate frozen selection record for one checkpoint.", ""]
    if problems or missing:
        lines += ["**Incomplete runs:** " + "; ".join(problems + [f"{m}: missing" for m in missing]), ""]
    lines += ["| run | replay exact | online CE @T0 → @Emax | "
              "eval obj @T0 → @Emax | grad norm window T0 → Emax | held-out CE @T0 → @Emax |",
              "|---|---|---|---|---|---|"]
    for name, r in summary["runs"].items():
        a, b = r["epoch_T0"], r["epoch_Emax"]
        lines.append(f"| {name} | {fmt(r['replay_exact'])} | {fmt(a['online_train_ce'])} → {fmt(b['online_train_ce'])} | "
                     f"{fmt(a['eval_objective'])} → {fmt(b['eval_objective'])} | "
                     f"{fmt(r['grad_norm_window_T0'])} → {fmt(r['grad_norm_window_Emax'])} | "
                     f"{fmt(a['heldout_ce'])} → {fmt(b['heldout_ce'])} |")
    if spec_rows:
        lines += ["", "Hessian Ritz endpoints (H̄ = eval-CE Hessian + wd·I):", "",
                  "| run | epoch | min Ritz | max Ritz | damping floor | grad norm |", "|---|---|---|---|---|---|"]
        for r in spec_rows:
            lines.append(f"| {r['run']} | {r['epoch']} | {fmt(r['min_ritz'])} | {fmt(r['max_ritz'])} | "
                         f"{fmt(r['damping_floor'])} | {fmt(r['eval_grad_norm_07'])} |")
    lines += ["", "Reading guide: inspect the ten-epoch window table, parameter update, eval gradient, held-out CE and "
              "interface drift together. A visually flat online CE alone is not a stationarity certificate. Optional "
              "spectrum rows, if explicitly run later, are finite Krylov diagnostics and do not select E*."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="out_root used by 20_A_convergence_curve.py")
    ap.add_argument("--out", default=None, help="report directory (default <root>/report)")
    ap.add_argument("--window", type=int, default=10)
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
    validate_panel(runs)
    summary, table = summarize(runs, args.window)
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
    print(json.dumps({"selected_E_star": None, "selection_status": "not_frozen",
                      "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
