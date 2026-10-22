#!/usr/bin/env python3
"""Interface LDS: does the Stage-1 influence function predict the counterfactual change of all four class
probabilities of the shadow-3 interface rows? Evaluated against the retrained change, without the attack.

Nothing is trained. The image LDS v3 run supplies every Stage-1 model and the truth (p_sub - p_full of the
4343 interface rows, 32 subsets x 8 seeds); v6 r1 (a299714) supplies the attack adjoints and its L3 scores,
which this run must reproduce. Per Stage-1 seed one job solves (H + gamma I) x = b_j for the 32 deleted-image
gradient sums on one shared Galerkin subspace and stores the logit JVPs of every prediction.

Prespecified primary: mean over outputs (nonmember row r, class c) of the Spearman correlation across the
32 subsets between the IF prediction J_p dtheta_hat (gamma = 0.03, mean over the score panel) and the
retrained change (mean over 8 seeds); one-sided joint-permutation p, subset-bootstrap CI.

Actions: prepare -> submit -> (status | resubmit) -> collect.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid
from functools import wraps
sys.dont_write_bytecode = True
import runtime as rt
import scheduler  # seed32 v2, unchanged

SCHEMA = 'OCT_interface_lds_v1'
V6_SCHEMA = 'OCT_cross_stage_lds_v6'
TIMES = dict(smoke='03:00:00', collect='01:00:00')
COPY = ('v3_out', 'v3_config_sha256', 'v3_cells', 'seed32_out', 'baseline_out', 'lds_pilot_out', 'seeds', 'epochs',
        'recipe', 'design_name', 'design', 'v3_prediction_tol', 'harvest_tol', 'attack', 'classes', 'n_shadow',
        'primary_classes', 'attack_gammas', 'attack_seeds', 'gammas', 'cg_fail_tol', 'probe_steps', 'batch',
        'weight_decay', 'cells', 'adjoint_identity_rel_tol')


def check_v6(v6):
    """The completed v6 r1 run: pin config, frozen panel, adjoints, truth cells and scores (not the h arrays)."""
    cfg = rt.read(v6 / 'config.json')
    h = rt.sha(v6 / 'config.json')
    if h != (v6 / 'config.sha256').read_text().strip():
        raise ValueError('v6 config checksum changed')
    if cfg['schema'] != V6_SCHEMA or cfg['git_commit'] != rt.V6_COMMIT:
        raise ValueError(f'Need the completed v6 r1 run at {rt.V6_COMMIT}')
    report = rt.read(v6 / 'report/complete.json')
    if report['status'] != 'complete' or report['config_sha256'] != h:
        raise ValueError('v6 r1 report is not complete')
    pins = [v6 / x for x in ('config.json', 'config.sha256', 'report/complete.json')]
    rt.completed(v6 / 'smoke', h, dict(kind='xsmoke'))
    pins += [v6 / 'smoke' / f for f in ('complete.json', 'frozen.json')]
    for s in cfg['seeds']:
        d = v6 / 'base' / f'seed{s}'
        rt.completed(d, h, dict(kind='base6', seed=s))
        pins += [d / f for f in ('complete.json', 'v.npz', 'result.json')]
    for cell in cfg['cells']:
        d = v6 / 'truth' / cell['name']
        rt.completed(d, h, dict(cell, kind='truth6'))
        pins += [d / 'complete.json', d / 'result.json']
    for s in cfg['score_seeds']:
        d = v6 / 'score' / f'seed{s}'
        rt.completed(d, h, dict(kind='score6', seed=s))
        pins += [d / f for f in ('complete.json', 'scores.npz', 'result.json')]
    fr = rt.read(v6 / 'smoke/frozen.json')
    if list(fr['attack_seeds']) != list(cfg['attack_seeds']) or fr['attack_gamma_star'] not in cfg['attack_gammas']:
        raise ValueError('v6 frozen panel is inconsistent with its config')
    return cfg, h, {str(p): rt.sha(p) for p in pins}


def prepare(a):
    if not 1 <= a.concurrency <= 32 or a.max_dim < 50 or a.max_new < 10 or a.pilot_rhs < 2:
        raise ValueError('Invalid concurrency / subspace / pilot settings')
    if not (0 < a.tol_main <= 1e-4 and 0 < a.tol_small <= 1e-3 and a.budget_gpu_hours > 0 and a.solve_hours > 0):
        raise ValueError('Invalid tolerances or budget')
    c = rt.prepare_base(a, SCHEMA)
    v6 = Path(a.v6_out).resolve()
    rt.disjoint(c['out'], v6)
    cfg, h6, pins = check_v6(v6)
    for key in ('full', 'data_dir', 'dataset_fingerprint'):
        if c[key] != cfg[key]:
            raise ValueError(f'{key} differs from the v6 r1 run')
    for k in COPY:
        c[k] = cfg[k]
    if a.gammas:
        sub = [float(x) for x in a.gammas.split(',') if x]
        if not sub or not set(sub) <= set(c['gammas']) or len(set(sub)) != len(sub):
            raise ValueError('--gammas must be distinct values of the v6 Stage-1 grid')
        c['gammas'] = [g for g in c['gammas'] if g in sub]
    if a.primary_gamma not in c['gammas']:
        raise ValueError('Primary gamma must be in the v6 Stage-1 grid')
    if list(cfg['score_seeds']) != list(cfg['seeds']):
        raise ValueError('Expected v6 scores for every Stage-1 seed')
    c['input_sha256'].update(cfg['input_sha256'])
    c['input_sha256'].update(pins)
    rt.verify_hashes(c['input_sha256'])
    for rel in rt.REUSED:
        c['source_sha256'][str(rt.REPO / rel)] = rt.sha(rt.REPO / rel)
    for d in (rt.LDS_V1, rt.STABLE_V1):
        for p in sorted(d.glob('*.py')):
            c['source_sha256'][str(p)] = rt.sha(p)
    c.update(v6_out=str(v6), v6_config_sha256=h6, primary_gamma=a.primary_gamma, tol_main=a.tol_main, tol_small=a.tol_small,
             max_dim=a.max_dim, max_new=a.max_new, pilot_rhs=a.pilot_rhs, budget_gpu_hours=a.budget_gpu_hours,
             solve_hours=a.solve_hours, concurrency=a.concurrency, partition=a.partition, cpu_account=a.cpu_account,
             cpu_partition=a.cpu_partition, permutations=2000, bootstraps=2000,
             primary=('Mean over the 2298 nonmember rows x 4 classes of Spearman across the 32 subsets between the IF '
                      'prediction J_p (H + gamma I)^{-1} b_j (gamma = primary_gamma, mean over the score panel) and the '
                      'retrained change p_sub - p_full (mean over 8 seeds). One-sided joint-permutation p < 0.05; '
                      'subset-bootstrap 95% CI; reported with the truth noise ceiling.'),
             coreported=('Same statistic on the 2045 training rows (all subsets; also retained-only and deleted-only); '
                         'row-group oracle of L3 = sum_r vbar . dp_hat (deleted / retained / nonmember rows replaced by the '
                         'retrained change) against the v6 six-attack-seed truth. Descriptive.'),
             holdout_policy=('Subsets, seeds, gamma grid and primary gamma are inherited from v6 r1. The score panel is fixed '
                             'in smoke from compute measurements only. No gamma, row group or class is chosen by outcome.'))
    rt.save_config(c)
    print(f"v6 r1 run {v6}; Stage-1 seeds {c['seeds']}; subsets {c['design']['m']}; gammas {c['gammas']} "
          f"(primary {c['primary_gamma']:g}); budget {c['budget_gpu_hours']:g} GPU-h; partition {c['partition']}")


def config(out):
    return rt.config(out, SCHEMA)


# ------------------------------------------------------------------ Slurm
def launcher(c, ledger):
    out = Path(c['out'])

    def launch(mode, extra, partition, label=None):
        label = label or mode
        account = c['cpu_account'] if mode == 'collect' else c['account']
        script = 'collect.slurm' if mode == 'collect' else 'job.slurm'
        cmd = ['sbatch', '--parsable', '--account=' + account, '--partition=' + partition,
               '--job-name=oct_if_' + label, '--output=' + str(out / 'logs/%x-%A_%a.out'),
               '--error=' + str(out / 'logs/%x-%A_%a.err'), *extra, str(rt.HERE / script), str(rt.HERE), str(out), mode]
        pending = out / 'submission_pending.json'
        rt.write(pending, dict(mode=mode, command=cmd, time=time.time()))
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected sbatch result')
        ledger.write(json.dumps(dict(mode=mode, job_id=job, command=cmd, time=time.time())) + '\n')
        ledger.flush()
        os.fsync(ledger.fileno())
        pending.unlink()
        return job
    return launch


def array(indices, conc):
    spec = f'0-{indices[-1]}' if indices == list(range(indices[-1] + 1)) else ','.join(map(str, indices))
    return f'--array={spec}%{conc}'


def submission_lock(fn):
    @wraps(fn)
    def wrapped(a):
        with rt.lock(Path(a.out) / '.locks' / 'submission.lock'):
            return fn(a)
    return wrapped


def hours(h):
    return f'{int(h):02d}:{int(round((h % 1) * 60)):02d}:00'


@submission_lock
def submit(a):
    c = config(a.out)
    out = Path(c['out'])
    after = lambda j: ['--dependency=afterok:' + j, '--kill-on-invalid-dep=yes']
    scheduler.receipts(out)
    with (out / 'submission.jsonl').open('x') as ledger:
        launch = launcher(c, ledger)
        jobs = dict(smoke=launch('smoke', ['--time=' + TIMES['smoke']], c['partition']))
        jobs['solve'] = launch('solve', ['--time=' + hours(c['solve_hours']), array(list(range(len(c['seeds']))), c['concurrency']),
                                         *after(jobs['smoke'])], c['partition'])
        jobs['collect'] = launch('collect', ['--time=' + TIMES['collect'], *after(jobs['solve'])], c['cpu_partition'])
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print('\n'.join(f'{k}={v}' for k, v in jobs.items()))


def panel(c):
    out = Path(c['out'])
    if not (out / 'smoke/complete.json').exists():
        return None
    rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='ifsmoke'))
    return rt.read(out / 'smoke/frozen.json')['score_panel']


def incomplete(c):
    out = Path(c['out'])
    h = rt.sha(out / 'config.json')
    seeds = panel(c) or c['seeds']

    def done(s):
        d = out / 'solve' / f'seed{s}'
        if not (d / 'complete.json').exists():
            return False
        rt.completed(d, h, dict(kind='ifsolve', seed=s))
        return True
    return [c['seeds'].index(s) for s in seeds if not done(s)]


def status(a):
    c = config(a.out)
    out = Path(c['out'])
    p = panel(c)
    print('smoke:', 'complete' if p is not None else 'NOT complete')
    for f in sorted(out.glob('smoke_budget_stop_*.json')):
        print('STOPPED BY BUDGET RULE', f.name, rt.read(f)['panel_rule'])
    if p is not None:
        fr = rt.read(out / 'smoke/frozen.json')
        r = rt.read(out / 'smoke/result.json')
        print('score panel:', p, '|', fr['panel_rule'])
        print(f"projected {fr['projected_hours_per_seed']:.2f} h / seed ({fr['projected_hvps_per_seed']:.0f} HVPs, "
              f"{r['timing']['hvp_seconds_median']:.1f} s each); pilot new vectors {[x['new_vectors'] for x in r['checks']['pilot']]}")
    todo = incomplete(c)
    print(f"solve: {len(p or c['seeds']) - len(todo)}/{len(p or c['seeds'])} complete; incomplete seeds {[c['seeds'][i] for i in todo]}")
    for d in sorted((out / 'solve').glob('seed*')):
        if (d / 'complete.json').exists():
            r = rt.read(d / 'result.json')
            print(d.name, 'qualified gammas:', [g['gamma'] for g in r['gammas'] if g['qualified']],
                  f"subspace {r['solver']['dim']} ({r['solver']['matvecs']} HVPs)")


@submission_lock
def resubmit(a):
    c = config(a.out)
    out = Path(c['out'])
    previous = scheduler.receipts(out)
    with (out / 'submission.jsonl').open('a') as ledger:
        launch = launcher(c, ledger)
        if panel(c) is None:
            scheduler.require_terminal([r for r in previous if r['mode'] == 'smoke'])
            print('smoke', launch('smoke', ['--time=' + TIMES['smoke']], c['partition'], 'smoke_re'))
            print('Resubmit again after smoke completes.')
            return
        todo = incomplete(c)
        scheduler.require_terminal([r for r in previous if r['mode'] == 'solve'])
        if not todo:
            print('All panel seeds complete; run collect.')
            return
        t = getattr(a, 'solve_hours', None) or c['solve_hours']
        print('solve', launch('solve', ['--time=' + hours(t), array(todo, c['concurrency'])], c['partition'], 'solve_re'), todo)


# -------------------------------------------------------------------- GPU
def ensure_designs(root, h):
    rt.completed(root / 'smoke', h, dict(kind='ifsmoke'))
    src, dst = root / 'smoke/designs.json', root / 'designs.json'
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise ValueError('Frozen designs differ from smoke manifest')
        return
    tmp = root / f'.designs-{uuid.uuid4().hex}.json'
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


def gpu(a):
    from core_if import ISession
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}"
    started = time.monotonic()
    if a.mode == 'smoke':
        final = root / 'smoke'
        if final.exists():
            ensure_designs(root, h)
            print('Smoke already complete')
            return
        with rt.lock(root / '.locks' / 'smoke.lock'):
            tmp = root / f'.smoke.tmp-{tag}'
            tmp.mkdir()
            session = ISession(c, freeze_to=tmp / 'designs.json')
            session.if_smoke(tmp)
            config(a.out)
            rt.finish(tmp, c, dict(kind='ifsmoke'), elapsed_seconds=time.monotonic() - started)
            os.rename(tmp, final)
        ensure_designs(root, h)
        print('COMPLETE smoke', flush=True)
        return
    ensure_designs(root, h)
    if a.mode != 'solve':
        raise ValueError('Unknown GPU mode')
    s = c['seeds'][a.index]
    if s not in panel(c):
        print(f'seed{s} is not in the frozen score panel; nothing to do', flush=True)
        return
    session = ISession(c)
    print(f'solve seed{s}', session.if_solve(s, tag), flush=True)
    config(a.out)
    print(f'COMPLETE solve index={a.index} elapsed={time.monotonic() - started:.0f}s', flush=True)


# ---------------------------------------------------------------- collect
def clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def collect(a):
    import analysis_if
    import report_if
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    ensure_designs(root, h)
    data = analysis_if.load(root, c, allow_partial=a.allow_partial)
    summary, tables = analysis_if.analyze(data, c)
    summary = clean(summary)
    partial = bool(data['missing'])
    rep = root / ('report' if not partial else 'report_partial_' + time.strftime('%Y%m%dT%H%M%S'))
    rep.mkdir(exist_ok=False)
    rt.write(rep / 'summary.json', summary)
    for key, rows in tables.items():
        if rows:
            rt.csv_write(rep / f'table_{key}.csv', clean(rows))
    (rep / 'REPORT_CN.md').write_text(report_if.render(summary, c, partial))
    rt.write(rep / 'complete.json', dict(status='partial' if partial else 'complete', config_sha256=h, missing=data['missing']))
    archive(root, rep)


def archive(root, rep):
    target = Path(str(root) + '.' + rep.name + '.tar.gz')
    keep = ('.json', '.jsonl', '.csv', '.md', '.txt', '.sha256', '.out', '.err')
    with tarfile.open(target, 'x:gz') as t:
        for p in sorted(root.rglob('*')):
            rel = p.relative_to(root)
            if not p.is_file() or any(part.startswith('.') for part in rel.parts):
                continue
            if rel.parts[0].startswith('report') and rel.parts[0] != rep.name:
                continue
            if p.suffix in keep:   # interface.npz (~25 MB per seed) stays on Delta
                t.add(p, arcname=str(Path(root.name) / rel))
    print(str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--out', required=True)
    q.add_argument('--v6-out', required=True, help='completed v6 r1 run (a299714)')
    q.add_argument('--audit', default=rt.AUDIT)
    q.add_argument('--primary-gamma', type=float, default=.03)
    q.add_argument('--gammas', default='', help='subset of the v6 Stage-1 grid (default: all); must contain the primary')
    q.add_argument('--tol-main', type=float, default=1e-6, help='relative residual for gamma >= primary gamma')
    q.add_argument('--tol-small', type=float, default=1e-4, help='relative residual for gamma < primary gamma')
    q.add_argument('--max-dim', type=int, default=1400, help='shared subspace size cap per seed')
    q.add_argument('--max-new', type=int, default=200, help='new directions per right-hand side cap')
    q.add_argument('--pilot-rhs', type=int, default=4, help='subsets solved in the smoke pilot')
    q.add_argument('--budget-gpu-hours', type=float, default=24.)
    q.add_argument('--solve-hours', type=float, default=12.)
    q.add_argument('--partition', default='gpuA100x4')
    q.add_argument('--cpu-account', default='bgjy-delta-cpu')
    q.add_argument('--cpu-partition', default='cpu')
    q.add_argument('--concurrency', type=int, default=8)
    for name in ('submit', 'status', 'resubmit', 'collect', 'gpu'):
        q = sub.add_parser(name)
        q.add_argument('--out', required=True)
        if name == 'gpu':
            q.add_argument('--mode', required=True)
            q.add_argument('--index', type=int, default=0)
        if name == 'collect':
            q.add_argument('--allow-partial', action='store_true')
        if name == 'resubmit':
            q.add_argument('--solve-hours', type=float, default=None)
    a = p.parse_args()
    globals()[a.action](a)


if __name__ == '__main__':
    main()
