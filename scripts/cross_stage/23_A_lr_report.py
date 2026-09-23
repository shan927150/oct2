#!/usr/bin/env python3
"""Paired report for the Route A convergence-control probe (23_A_lr_continuation.py).

Pure NumPy + matplotlib; runs on a login node.  Descriptive only: it marks
whether each arm reached the pre-declared exploratory gradient target, and it
never declares convergence, selects a checkpoint, or approves an LOO stage.

Reading rules fixed in advance:

  * The stationarity readout is the eval CE + L2 full-train-set gradient norm.
    The parameter update is reported next to it but proves nothing on its own:
    a small learning rate shrinks updates even far from a stationary point.
  * Target check uses the median over the last ``tail`` epochs; a single low
    value or a flat loss is not a pass.
  * Two large gradients do not establish an objective mismatch. A small
    dropout expected gradient and persistently larger eval gradient, resolved
    beyond Monte Carlo uncertainty, would support that explanation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "pathway2_A_lr_continuation_v1"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_ARMS = ("control", "decay")
ARM_COLORS = {"control": "#2a78d6", "decay": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", "None", None):
                row[key] = None
            else:
                try:
                    row[key] = int(value) if key in ("epoch", "seed") else float(value)
                except ValueError:
                    row[key] = {"True": True, "False": False}.get(value, value)
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_runs(root):
    runs, problems = {}, []
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            continue
        if manifest.get("status") != "complete":
            problems.append(f"{manifest.get('run')}: status={manifest.get('status')} error={manifest.get('error')}")
            continue
        name = manifest["run"]
        if name in runs:
            problems.append(f"Duplicate run: {name}")
            continue
        try:
            run = {"manifest": manifest, "curve": read_csv(manifest_path.parent / "curve.csv")}
            for key in ("input_sha256", "dataset_fingerprint"):
                run[key] = json.loads((manifest_path.parent / f"{key}.json").read_text())
            runs[name] = run
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: unreadable run files: {exc}")
    return runs, problems


def validate_runs(runs, expected_seeds=EXPECTED_SEEDS):
    """Check the declared panel, epoch coverage and evidence of paired inputs."""
    problems, by_key = [], {}
    expected = {(int(s), a) for s in expected_seeds for a in EXPECTED_ARMS}
    common = ("resume_epoch", "total_epochs", "base_lr", "weight_decay",
              "gradient_target_exploratory", "git_commit", "source_sha256")
    reference = next(iter(runs.values()))["manifest"] if runs else {}
    for name, run in runs.items():
        m, curve = run["manifest"], run["curve"]
        key = (m.get("seed"), m.get("arm"))
        for field in common:
            if m.get(field) is None or m.get(field) != reference.get(field):
                problems.append(f"{name}: cross-run {field} missing or mismatched")
        if key not in expected or key in by_key:
            problems.append(f"{name}: unexpected or duplicate seed/arm {key}")
        by_key[key] = run
        resume, total = m["resume_epoch"], m["total_epochs"]
        if [r.get("epoch") for r in curve] != list(range(resume, total + 1)):
            problems.append(f"{name}: missing, duplicate or unordered epochs")
        if m.get("n_curve_rows") != total - resume + 1:
            problems.append(f"{name}: manifest curve-row count mismatch")
        if m.get("status") != "complete":
            problems.append(f"{name}: run is not complete")
        for flag in ("input_files_unchanged", "source_files_unchanged", "dataset_arrays_unchanged"):
            if m.get(flag) is not True:
                problems.append(f"{name}: {flag} is not true")
        if m.get("resume_check", {}).get("passed") is not True:
            problems.append(f"{name}: resume check did not pass")
        every = int(m.get("args", {}).get("gradient_every", 1))
        for r in curve:
            e = r["epoch"]
            if (e % every == 0 or e in (resume, total)) and not finite(r.get("eval_grad_norm")):
                problems.append(f"{name}: missing/nonfinite gradient at epoch {e}")
        for field in ("input_sha256", "dataset_fingerprint"):
            if not run.get(field):
                problems.append(f"{name}: missing {field}")
    for key in sorted(expected - set(by_key)):
        problems.append(f"Missing seed/arm: {key}")
    for seed in expected_seeds:
        c, d = (by_key.get((int(seed), arm)) for arm in EXPECTED_ARMS)
        if c is None or d is None:
            continue
        for field in ("input_sha256", "dataset_fingerprint"):
            if c.get(field) != d.get(field):
                problems.append(f"seed {seed}: paired {field} mismatch")
        for field in ("base_run", "resume_epoch", "total_epochs", "base_lr", "weight_decay",
                      "gradient_target_exploratory", "epoch_order_sha256", "source_sha256",
                      "git_commit", "final_rng_sha256"):
            left, right = c["manifest"].get(field), d["manifest"].get(field)
            if left is None or right is None or left != right:
                problems.append(f"seed {seed}: paired {field} missing or mismatched")
        left = c["manifest"].get("resume_check", {}).get("rng_fingerprint")
        right = d["manifest"].get("resume_check", {}).get("rng_fingerprint")
        if not left or left != right:
            problems.append(f"seed {seed}: paired starting RNG missing or mismatched")
    return problems


def tail_values(curve, field, tail, total):
    # A missing tail value must not silently pull in an earlier epoch.
    return [r[field] for r in curve if total - tail < r["epoch"] <= total and finite(r.get(field))]


def summarize(runs, tail, expected_seeds=EXPECTED_SEEDS):
    if tail < 1:
        raise ValueError("tail must be positive")
    seeds, arms = sorted(expected_seeds), list(EXPECTED_ARMS)
    validation_problems = validate_runs(runs, seeds)
    per_run, pairing_ok = {}, not validation_problems
    for name, run in runs.items():
        m, curve = run["manifest"], run["curve"]
        resume, total = m["resume_epoch"], m["total_epochs"]
        target = m.get("gradient_target_exploratory")
        start = next((r for r in curve if r["epoch"] == resume), None)
        end = next((r for r in curve if r["epoch"] == total), None)
        grads = tail_values(curve, "eval_grad_norm", tail, total)
        updates = tail_values(curve, "relative_epoch_update", tail, total)
        mc = [r for r in curve if finite(r.get("dropout_mc_mean_gradient_norm"))]
        per_run[name] = {
            "seed": m["seed"], "arm": m["arm"], "resume_epoch": resume, "total_epochs": total,
            "decay_schedule": m.get("decay_schedule"),
            "grad_at_resume": start.get("eval_grad_norm") if start else None,
            "grad_at_end": end.get("eval_grad_norm") if end else None,
            "grad_tail_median": float(np.median(grads)) if grads else None,
            "grad_tail_max": float(max(grads)) if grads else None,
            "relative_update_tail_median": float(np.median(updates)) if updates else None,
            "eval_objective_at_resume": start.get("eval_objective") if start else None,
            "eval_objective_at_end": end.get("eval_objective") if end else None,
            "heldout_ce_at_end": end.get("heldout_ce") if end else None,
            "interface_tv_from_resume_at_end": end.get("interface_tv_from_T0_mean") if end else None,
            "gradient_target": target,
            "tail_gradient_count": len(grads),
            "target_met_tail_median": (float(np.median(grads)) <= target)
                if pairing_ok and len(grads) == tail and total - resume >= tail and target else None,
            "dropout_mc_last": ({"epoch": mc[-1]["epoch"],
                                 "mc_mean_grad": mc[-1]["dropout_mc_mean_gradient_norm"],
                                 "eval_grad_same_epoch": mc[-1].get("eval_grad_norm"),
                                 "mc_reps": mc[-1].get("dropout_mc_reps"),
                                 "mc_gradient_rms_spread": mc[-1].get("dropout_mc_gradient_rms_spread"),
                                 "mc_mean_gradient_rms_error_estimate": (
                                     mc[-1]["dropout_mc_gradient_rms_spread"] /
                                     math.sqrt(mc[-1]["dropout_mc_reps"] - 1)
                                     if finite(mc[-1].get("dropout_mc_gradient_rms_spread"))
                                     and (mc[-1].get("dropout_mc_reps") or 0) > 1 else None)} if mc else None),
        }
    pairs = {}
    for seed in seeds:
        entry = {}
        for arm in arms:
            match = [v for v in per_run.values() if v["seed"] == seed and v["arm"] == arm]
            entry[arm] = match[0] if len(match) == 1 else None
            if entry[arm] is None:
                pairing_ok = False
        c, d = entry.get("control"), entry.get("decay")
        entry["grad_ratio_decay_over_control"] = (
            d["grad_tail_median"] / c["grad_tail_median"]
            if c and d and finite(c.get("grad_tail_median")) and c["grad_tail_median"] > 0
            and finite(d.get("grad_tail_median")) else None)
        pairs[str(seed)] = entry
    return {"tail_epochs": tail, "seeds": seeds, "arms": arms, "pairing_complete": pairing_ok,
            "per_run": per_run, "per_seed_pairs": pairs, "problems": validation_problems,
            "note": "Descriptive probe on baselines only. The exploratory gradient target is not a "
                    "convergence certificate; no E*, truth or LOO decision is made here."}


def style(ax, title, log=False):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Epoch", fontsize=8, color=MUTED)
    ax.tick_params(which="both", labelsize=7, colors=MUTED, length=2)
    ax.grid(color=GRID, linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if log:
        ax.set_yscale("log")


def plot(runs, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("eval_grad_norm", "‖∇(eval CE + L2)‖ — the Eq. 56 readout (log)", True),
              ("relative_epoch_update", "Relative parameter update (small lr ⇒ small update; log)", True),
              ("eval_objective", "Eval objective (log)", True),
              ("online_train_ce", "Online training CE (dropout on)", False),
              ("interface_tv_prev_mean", "Interface drift: TV to previous epoch (log)", True),
              ("heldout_ce", "Held-out CE (non-members)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    target = next((v["gradient_target"] for v in summary["per_run"].values() if v.get("gradient_target")), None)
    for ax, (field, label, log) in zip(axes.flat, panels):
        for name in sorted(runs):
            m = runs[name]["manifest"]
            rows = [r for r in runs[name]["curve"] if finite(r.get(field)) and (not log or r[field] > 0)]
            ax.plot([r["epoch"] for r in rows], [r[field] for r in rows], lw=1.3,
                    color=ARM_COLORS.get(m["arm"], MUTED),
                    alpha=.85, label=None)
        if field == "eval_grad_norm":
            if target:
                ax.axhline(target, color=INK, lw=.9, ls=":")
            mc_plotted = False
            for name in sorted(runs):
                m = runs[name]["manifest"]
                mc = [r for r in runs[name]["curve"] if finite(r.get("dropout_mc_mean_gradient_norm"))]
                if mc:
                    ax.scatter([r["epoch"] for r in mc], [r["dropout_mc_mean_gradient_norm"] for r in mc],
                               s=12, marker="x", color=ARM_COLORS.get(m["arm"], MUTED), alpha=.8)
                    mc_plotted = True
            if target or mc_plotted:
                extras = []
                if target:
                    extras.append(f"dotted = exploratory target {target:g}")
                if mc_plotted:
                    extras.append("× = dropout-MC mean gradient")
                label += "\n(" + "; ".join(extras) + ")"
        style(ax, label, log)
    handles = [plt.Line2D([0], [0], color=ARM_COLORS[a], lw=2, label=a) for a in summary["arms"]]
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=False)
    fig.suptitle("Paired continuation from E100: one colour per arm, one line per seed", fontsize=10,
                 color=INK, x=.01, ha="left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def markdown(summary):
    lines = ["# Route A · paired lr-continuation probe (baselines only)", "",
             f"Tail window: last {summary['tail_epochs']} epochs. The gradient target is exploratory; "
             "this report never certifies convergence or triggers an LOO stage.", "",
             f"Panel complete and paired: {summary['pairing_complete']}. Expected {len(summary['seeds']) * 2} runs.", "",
             "| seed | arm | grad @resume | grad tail median | target met | rel. update tail | "
             "eval obj resume → end | held-out CE end |", "|---|---|---|---|---|---|---|---|"]
    for name in sorted(summary["per_run"]):
        r = summary["per_run"][name]
        lines.append(f"| {r['seed']} | {r['arm']} | {fmt(r['grad_at_resume'])} | {fmt(r['grad_tail_median'])} | "
                     f"{fmt(r['target_met_tail_median'])} | {fmt(r['relative_update_tail_median'])} | "
                     f"{fmt(r['eval_objective_at_resume'])} → {fmt(r['eval_objective_at_end'])} | "
                     f"{fmt(r['heldout_ce_at_end'])} |")
    lines += ["", "| seed | grad tail median: decay / control |", "|---|---|"]
    for seed, pair in summary["per_seed_pairs"].items():
        lines.append(f"| {seed} | {fmt(pair.get('grad_ratio_decay_over_control'))} |")
    lines += ["", "Reading guide: judge the decay arm by the gradient column, not the update column. "
              "Large eval and dropout-MC gradients together do not establish objective mismatch. "
              "A small dropout expected gradient and a persistently larger eval gradient would support "
              "mismatch only when resolved beyond MC uncertainty. The summary records an estimated RMS "
              "error of the mean gradient vector (not a confidence interval for its norm). "
              "Do not silently retrain without dropout. Whether to build the small-panel LOO on the decay recipe is a "
              "separate, explicitly approved step.", ""]
    if summary.get("problems"):
        lines += ["Incomplete/invalid inputs:", ""] + [f"- {p}" for p in summary["problems"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS),
                    help="expected panel, default 42..46; override only for explicitly smaller tests")
    ap.add_argument("--allow_incomplete", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    runs, problems = load_runs(root)
    if not runs:
        raise SystemExit(f"No completed probe runs under {root}")
    summary = summarize(runs, args.tail, args.seeds)
    problems += summary["problems"]
    if problems and not args.allow_incomplete:
        raise SystemExit("Incomplete: " + "; ".join(problems))
    out = Path(args.out).resolve() if args.out else root / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary["problems"] = problems
    summary["pairing_complete"] = not problems
    if problems:
        for row in summary["per_run"].values():
            row["target_met_tail_median"] = None
    (out / "probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(runs, summary, out / "lr_probe_curves")
    (out / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({k: {"grad_tail_median": v["grad_tail_median"], "target_met": v["target_met_tail_median"]}
                      for k, v in summary["per_run"].items()}, indent=2))


if __name__ == "__main__":
    main()
