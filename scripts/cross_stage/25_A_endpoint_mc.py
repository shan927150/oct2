#!/usr/bin/env python3
"""Route A meeting plan part C: endpoint dropout-vs-eval gradient comparison.

At the decay-arm E140 checkpoints only, compare the eval-mode full-batch
gradient g_eval with the dropout expected gradient g_dropout (mean over many
MC draws), as VECTORS, not as a difference of norms:

  * ||g_dropout - g_eval||            (the mismatch magnitude)
  * cos(g_dropout, g_eval)            (direction agreement)
  * ||g_eval||, ||g_dropout||         (context)
  * MC standard error of the mean gradient vector, and
    ||g_dropout - g_eval|| / SE       (reported as an uncalibrated ratio,
                                       NOT a significance test)

No training; loads checkpoints only. This only estimates whether the two
objectives' gradients differ beyond MC noise at the endpoint; it does not
decide any next step by itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_diagnostic_core as core  # noqa: E402
import route_a_smallpanel as common
_spec = importlib.util.spec_from_file_location("stage1_legacy_a25", HERE / "11_stage1_diagnostics.py")
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)
WD = legacy.WD


def gradient_comparison(model, X, y, indices, batch, mc_reps, mc_seed):
    """Return eval and dropout mean gradient vectors and the MC error of the mean.

    Streams the MC draws: keeps only the running sum vector and the scalar
    sum of squared norms, so memory is O(params), not O(mc_reps * params).
    """
    with core.preserve_rng_and_modes(model):
        _, g_eval = core.objective_snapshot(model, X, y, indices, batch, WD, with_grad=True, dropout=False)
        g_eval = g_eval.double()
        gsum = torch.zeros_like(g_eval)
        sum_sq_norm = 0.0
        for rep in range(mc_reps):
            torch.manual_seed(mc_seed + rep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(mc_seed + rep)
            _, g = core.objective_snapshot(model, X, y, indices, batch, WD, with_grad=True, dropout=True)
            g = g.double()
            gsum += g
            sum_sq_norm += float(torch.dot(g, g))
        gbar = gsum / mc_reps
    # SE of the mean vector: sqrt( (sum||g_r||^2 - R||gbar||^2) / (R (R-1)) )
    between = max(0.0, sum_sq_norm - mc_reps * float(torch.dot(gbar, gbar)))
    se_mean = math.sqrt(between / (mc_reps * (mc_reps - 1))) if mc_reps > 1 else float("nan")
    diff = gbar - g_eval
    return {
        "eval_grad_norm": core.norm(g_eval),
        "dropout_mc_mean_grad_norm": core.norm(gbar),
        "diff_norm_dropout_minus_eval": core.norm(diff),
        "cosine_dropout_eval": core.cosine(gbar, g_eval),
        "mc_mean_vector_standard_error": se_mean,
        "diff_norm_over_mc_error": (core.norm(diff) / se_mean) if se_mean and math.isfinite(se_mean) else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--mc_reps', type=int, default=128)
    ap.add_argument('--hvp_batch', type=int, default=64)
    ap.add_argument('--require_cuda', action='store_true')
    args = ap.parse_args(argv)
    root, cfg = common.load_config(args.config.parent)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this invocation')
    if args.mc_reps < 2 or args.hvp_batch < 1:
        raise ValueError('Invalid MC count or batch size')
    full, probe, out = Path(cfg['full_dir']), Path(cfg['probe_root']), args.out.resolve()
    if out != root / 'endpoint_mc':
        raise ValueError('Output must be the prepared endpoint_mc directory')
    common.disjoint(out, full, probe, common.REPO, cfg['data_dir'], cfg['e100_root'])
    out.mkdir(parents=True, exist_ok=False)
    score = legacy.module_from_path('score07_a25', HERE / '07_cross_stage_score_ladder.py')
    conv = legacy.module_from_path('curve20_a25', HERE / '20_A_convergence_curve.py')
    pilot = score.import_pilot_module()
    ns = argparse.Namespace(full_dir=str(full), dose_dir=None, mode='convergence',
                            data_dir=cfg['data_dir'], seeds=common.SEEDS)
    context = legacy.load_context(ns, pilot)
    X, y = context['X'], context['y']
    if conv.dataset_fingerprint(X,y) != cfg['dataset_fingerprint']:
        raise ValueError('Dataset differs from the paired probe')
    train = context['train']
    args.seeds, args.arm, args.epoch = common.SEEDS, 'decay', 140
    common.write_new(out/'running.json', dict(status='running',config_sha256=common.sha(args.config),
                                             git_commit=cfg['git_commit'],mc_reps=args.mc_reps))
    rows = []
    for seed in args.seeds:
        ckpt = probe / f'shadow3_seed{seed}_decay/checkpoints/epoch140.pt'
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}. Pass --baseline_glob to match your probe layout.")
        payload = legacy.load_payload(ckpt, pilot)
        common.check_baseline_metadata(payload,seed,'decay',cfg['probe_facts'][f'{seed}/decay'])
        pilot.seed_everything(910000 + seed * 1000, True)
        model = legacy.model_from_payload(payload, pilot, X, y).to(pilot.DEVICE)
        stats = gradient_comparison(model, X, y, train, args.hvp_batch, args.mc_reps, 910000 + seed * 1000)
        row = {"seed": seed, "arm": args.arm, "epoch": args.epoch, "mc_reps": args.mc_reps,
               "checkpoint": str(ckpt), **stats}
        rows.append(row)
        legacy.write_csv(out / "endpoint_mc.csv", rows)
        print(f"seed {seed}: ||g_eval||={stats['eval_grad_norm']:.4g}  ||g_drop||={stats['dropout_mc_mean_grad_norm']:.4g}"
              f"  ||g_drop-g_eval||={stats['diff_norm_dropout_minus_eval']:.4g}"
              f"  cos={stats['cosine_dropout_eval']:.3f}  SE={stats['mc_mean_vector_standard_error']:.4g}"
              f"  diff/SE={stats['diff_norm_over_mc_error']}", flush=True)
        del model
    ratios = [r["diff_norm_over_mc_error"] for r in rows if r["diff_norm_over_mc_error"] is not None]
    common.load_config(root)
    if conv.dataset_fingerprint(X,y) != cfg["dataset_fingerprint"]:
        raise ValueError("Dataset changed during MC")
    summary = {"schema": "pathway2_A_endpoint_mc_v3", "status": "complete",
               "config_sha256": common.sha(args.config), "git_commit": cfg["git_commit"], "arm": args.arm, "epoch": args.epoch,
               "mc_reps": args.mc_reps, "n": len(rows), "rows": rows,
               "min_diff_over_mc_error": float(min(ratios)) if ratios else None,
               "note": "Descriptive vector comparison of dropout vs eval gradients at the endpoint. "
                       "diff_norm_over_mc_error is an uncalibrated ratio, NOT a significance test. A large "
                       "ratio suggests the observed gradient-vector difference is larger than the estimated "
                       "MC error, without a calibrated significance claim or a causal conclusion about L3."}
    (out / "endpoint_mc_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"min_diff_over_mc_error": summary["min_diff_over_mc_error"]}, indent=2))


if __name__ == "__main__":
    main()
