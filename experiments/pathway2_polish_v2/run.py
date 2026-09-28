#!/usr/bin/env python3
"""Baseline polish only: prepare, submit, GPU execution and collect."""
import argparse
import copy
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import runtime as rt

SCHEMA = 'OCT_objective_polish_v2'


def prepare(a):
    if min(a.max_outer, a.eval_budget, a.history_size, a.diagnostic_every) < 1:
        raise ValueError('Budgets must be positive')
    c = rt.prepare_base(a, SCHEMA)
    cells = [x.copy() for x in c['audited_cells'] if x['arm'] == 'decay' and x['epoch'] == 140]
    if sorted(x['seed'] for x in cells) != [42, 44]:
        raise ValueError('Expected audited decay E140 seeds 42 and 44')
    c.update(cells=cells, slurm_time='02:00:00',
        lbfgs=dict(max_outer=a.max_outer, eval_budget=a.eval_budget,
                   history_size=a.history_size, diagnostic_every=a.diagnostic_every),
        rho_target=.1, rho_alpha=.1,
        request_note='User requested two isolated experiment branches on 2026-09-28; baseline polish only.',
        design='E140 decay -> eval CE+L2 full-batch LBFGS; fresh optimizer; no damping/deletion/attack')
    rt.save_config(c)


def gpu(a):
    import numpy as np
    import torch
    import polish_core as pc
    from audit_torch import snapshot
    c = rt.config(a.out, SCHEMA)
    if not 0 <= a.index < len(c['cells']):
        raise ValueError('Index out of bounds')
    if not a.smoke:
        rt.completed(Path(c['out']) / 'smoke', rt.sha(Path(c['out']) / 'config.json'))
    cell = c['cells'][a.index]
    dest = Path(c['out']) / ('smoke' if a.smoke else cell['name'])
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    ctx, legacy, pilot, conv = rt.load_context(c, cell['seed'])
    payload = legacy.load_payload(cell['checkpoint'], pilot)
    for key in ('seed', 'epoch', 'arm', 'epoch_order_sha256'):
        if payload['metadata'][key] != cell[key]:
            raise ValueError('Checkpoint metadata mismatch: ' + key)
    model = legacy.model_from_payload(payload, pilot, ctx['X'], ctx['y'])
    kwargs = dict(X=ctx['X'], y=ctx['y'], indices=ctx['train'],
                  patient_rows=ctx['patient_rows'], batch=c['batch'], wd=c['weight_decay'])

    def snap(m, dropout=False):
        return snapshot(m, **kwargs, dropout=dropout)

    def double_snap(m):
        m64 = copy.deepcopy(m).double()
        info, vectors = snap(m64)
        del m64
        return info, vectors

    before32, v032 = snapshot(model, **kwargs, check_identity=a.smoke)
    if abs(before32['ce'] - cell['reference_eval_ce']) > 1e-8 + 5e-5 * abs(cell['reference_eval_ce']):
        raise ValueError('Loaded endpoint does not reproduce audited eval CE')
    before64, v064 = double_snap(model)
    if a.smoke:
        again, vv = snap(model)
        if before32['ce'] != again['ce'] or not np.array_equal(v032, vv):
            raise ValueError('Deterministic repeated eval failed')
    options = dict(c['lbfgs'])
    if a.smoke:
        options.update(max_outer=2, eval_budget=12, diagnostic_every=1)
    out = pc.polish(model, ctx['X'], ctx['y'], ctx['train'], c['weight_decay'], c['batch'],
                    **options, snapshot_fn=snap, patients_for_snapshot=c['patients'])
    after32, v132 = snap(model)
    after64, v164 = double_snap(model)
    rows = []
    for stage, v32, v64 in [('before', v032, v064), ('after', v132, v164)]:
        ng = float(np.linalg.norm(v64[0]))
        eg = float(np.linalg.norm(v64[0] - v32[0]))
        for j, pid in enumerate(c['patients'], 1):
            nb = float(np.linalg.norm(v64[j]))
            eb = float(np.linalg.norm(v64[j] - v32[j]))
            for alpha in (.1, 1.):
                rho = ng / (alpha * nb) if nb else None
                resolved = bool(eg <= .1 * ng and eb <= .1 * nb and nb > 0)
                rows.append(dict(stage=stage, seed=cell['seed'], patient_id=pid, alpha=alpha,
                    g_norm_fp64=ng, b_norm_fp64=nb, rho_fp64=rho,
                    g_fp32_fp64_difference=eg, b_fp32_fp64_difference=eb,
                    precision_agreement_10pct=resolved,
                    engineering_target_met=bool(rho is not None and rho <= c['rho_target'] and resolved)))
    result = dict(cell=cell, stop_reason=out['stop_reason'], closure_calls=out['closure_calls'],
        accepted_iterations=out['accepted_iterations'], diagnostic_snapshot_calls=out['diagnostic_snapshot_calls'],
        relative_theta_shift=out['theta_shift'] / out['theta0_norm'],
        g_before_fp64=float(np.linalg.norm(v064[0])), g_after_fp64=float(np.linalg.norm(v164[0])),
        objective_before_fp64=before64['objective'], objective_after_fp64=after64['objective'],
        objective_before_fp32=before32['objective'], objective_after_fp32=after32['objective'],
        rho_rows=rows, smoke=a.smoke,
        interpretation='FP64 verification is not a rigorous error bound or an IF-validity certificate')
    if not np.isfinite(result['g_after_fp64']):
        raise RuntimeError('Non-finite final audit')
    if a.smoke and out['stop_reason'] == 'nonfinite_or_increase_rolled_back':
        raise RuntimeError('Smoke optimization was numerically invalid')
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        metadata=dict(schema=SCHEMA, source_checkpoint=cell['checkpoint'], source_sha256=cell['checkpoint_sha256'],
            config_sha256=rt.sha(Path(c['out']) / 'config.json'), seed=cell['seed'],
            objective='eval CE+L2', dropout_enabled=False, weight_decay=c['weight_decay'],
            training_n=2045, stop_reason=out['stop_reason'],
            resume_note='Weights only; new optimizer required, not exact optimizer continuation')), dest / 'polished.pt')
    # Same precision for BOTH g and b in every reported rho.
    np.savez_compressed(dest / 'gradient_vectors.npz', before_fp32=v032, before_fp64=v064,
                         after_fp32=v132, after_fp64=v164, patient_ids=c['patients'])
    rt.csv_write(dest / 'trajectory.csv', out['trajectory'])
    rt.csv_write(dest / 'rho.csv', rows)
    rt.write(dest / 'result.json', result)
    if conv.dataset_fingerprint(ctx['X'], ctx['y']) != c['dataset_fingerprint']:
        raise ValueError('Dataset changed during run')
    rt.config(a.out, SCHEMA)
    rt.finish(dest, c, cell, smoke=a.smoke, input_and_source_unchanged=True,
              elapsed_seconds=time.monotonic()-started, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    print('COMPLETE ' + str(dest), flush=True)


def collect(a):
    c = rt.config(a.out, SCHEMA)
    values = rt.results(c)
    rep = Path(c['out']) / 'report'
    rep.mkdir(exist_ok=False)
    rt.csv_write(rep / 'polish_summary.csv', [dict(seed=x['cell']['seed'], **{
        k: v for k, v in x.items() if k not in ('cell', 'rho_rows', 'interpretation')}) for x in values])
    rt.csv_write(rep / 'rho.csv', [r for x in values for r in x['rho_rows']])
    lines = ['# Baseline objective polish', '',
        'Fixed shadow 3, N=2045, eval CE + 1e-5/2 ||theta||². Seeds 42/44.',
        'FP32 optimization; both g and patient b recomputed in FP64 at both endpoints.',
        'The closure budget excludes diagnostic snapshots and endpoint FP64 verification.',
        'Small rho and numerical agreement are diagnostics, not a stationarity/Hessian/IF theorem.',
        'Budget/stall stop reasons do not mean convergence. Large movement changes the local question.',
        'polished.pt and gradient_vectors.npz remain on Delta and are SHA-locked in complete.json.', '']
    for x in values:
        lines.append(f"- seed{x['cell']['seed']}: g {x['g_before_fp64']:.5g} -> {x['g_after_fp64']:.5g}; "
                     f"relative shift {x['relative_theta_shift']:.5g}; stop {x['stop_reason']}")
    (rep / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    rt.write(rep / 'complete.json', dict(status='complete', config_sha256=rt.sha(Path(c['out']) / 'config.json')))
    rt.archive(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    q = rt.common_cli(p)
    q.add_argument('--max-outer', type=int, default=100)
    q.add_argument('--eval-budget', type=int, default=500)
    q.add_argument('--history-size', type=int, default=20)
    q.add_argument('--diagnostic-every', type=int, default=10)
    a = p.parse_args()
    if a.action == 'submit':
        rt.submit(a, SCHEMA, 'pol2')
    else:
        {'prepare': prepare, 'gpu': gpu, 'collect': collect}[a.action](a)


if __name__ == '__main__':
    main()
