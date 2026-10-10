#!/usr/bin/env python3
"""Image-level random 50% LDS (TRAK-aligned) for the Stage-1 shadow-3 CNN.

Mentor decision 2026-10-09: delete a fixed NUMBER OF IMAGES (50%), not patients,
so subset size cannot act as a confounder; normalise query targets by image
count; 32 subsets x 8 seeds. Image-level proxy (all seeds) and IF (score seeds,
fixed gamma grid). Full baselines are harvested from the seed32 v2 run.

Actions: prepare -> submit -> (status | resubmit | queue-report) -> collect.
"""
import argparse
import csv
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

SCHEMA = 'OCT_image_lds_v3'
SEED32_OUT = '/u/yli103/oct2-pathway2-runs/A/seed32_lds_v2_reviewed_20261007T041611331355841Z'
RECIPE = dict(lr=.001, warmup_frac=.1, ema_decay=.99, wd=1e-5, batch=128)
STABLE_SEEDS = (42, 43, 44)
MINUTES = dict(harvest=.5, half=2.2)
TIMES = dict(smoke='01:00:00', pack='00:30:00', proxy='01:30:00')


def floats(text):
    return [float(x) for x in text.split(',') if x]


def ints(text):
    return [int(x) for x in text.split(',') if x]


def check_seed32(c, s32, seeds):
    """The seed32 v2 run supplies full baselines and the frozen upstream inputs."""
    cfg = rt.read(s32 / 'config.json')
    h = rt.sha(s32 / 'config.json')
    if h != (s32 / 'config.sha256').read_text().strip():
        raise ValueError('seed32 config checksum changed')
    if (cfg['schema'] != 'OCT_seed32_lds_v2' or cfg['git_commit'] != rt.SEED32_COMMIT or cfg['recipe'] != RECIPE
            or cfg['epochs'] != 140 or cfg['dataset_fingerprint'] != c['dataset_fingerprint']
            or cfg['full'] != c['full'] or cfg['data_dir'] != c['data_dir']):
        raise ValueError('Need the seed32 v2 reviewed run (42f88f3, stable recipe, same data)')
    rt.completed(s32 / 'smoke', h)
    # Reused pure modules must be byte-identical to what the seed32 run hashed.
    recorded = {str(Path(k).relative_to(cfg['repo'])): v for k, v in cfg['source_sha256'].items()}
    for rel in rt.REUSED + ['experiments/pathway2_seed32_v2/scheduler.py']:
        if recorded.get(rel) != rt.sha(rt.REPO / rel):
            raise ValueError('Reused module differs from the seed32 run: ' + rel)
    cells = {x['name']: x for x in cfg['cells']}
    for s in seeds:
        name = f'full_seed{s}'
        if name not in cells:
            raise ValueError('seed32 run has no full baseline for seed ' + str(s))
        d = s32 / 'cells' / name
        rt.completed(d, h, cells[name])
        for rel in ('complete.json', 'result.json', 'interface.npz') + (() if s in STABLE_SEEDS else ('final.pt',)):
            c['input_sha256'][str(d / rel)] = rt.sha(d / rel)
    # Upstream frozen artefacts, verified against the hashes the seed32 run recorded.
    bpath, ppath = Path(cfg['baseline_out']), Path(cfg['lds_pilot_out'])
    wanted = [bpath / f'seed{s}_baseline' / rel for s in STABLE_SEEDS for rel in ('final.pt', 'interface.npz')]
    wanted += [ppath / f'score_seed{s}' / rel for s in STABLE_SEEDS for rel in ('scores.csv', 'design.json')]
    wanted += [Path(c['full']) / f'stage1_order_seed{s}.npz' for s in STABLE_SEEDS]
    for p in wanted:
        if cfg['input_sha256'].get(str(p)) is None:
            raise ValueError('seed32 config does not pin ' + str(p))
        c['input_sha256'][str(p)] = cfg['input_sha256'][str(p)]
    for rel in ('config.json', 'config.sha256', 'smoke/complete.json'):
        c['input_sha256'][str(s32 / rel)] = rt.sha(s32 / rel)
    rt.verify_hashes(c['input_sha256'])
    return cfg


def make_cells(c):
    cells = [dict(name=f'full_seed{s}', group='full', kind='full', seed=s, source='harvest', minutes=MINUTES['harvest'])
             for s in c['seeds']]
    for s in c['seeds']:  # seed-major: early packs complete whole seeds first
        for j in range(c['design']['m']):
            cells.append(dict(name=f"{c['design_name']}_s{j:03d}_seed{s}", group=c['design_name'], kind='subset',
                              subset=j, seed=s, source='train', minutes=MINUTES['half']))
    return cells


def make_packs(cells, budget):
    packs = {}
    for cell in cells:
        group = packs.setdefault(cell['group'], [])
        if not group or len(group[-1]) >= 8 or sum(x['minutes'] for x in group[-1]) + cell['minutes'] > budget:
            group.append([])
        group[-1].append(cell)
    return {g: [[x['name'] for x in p] for p in ps] for g, ps in packs.items()}


def prepare(a):
    gammas = floats(a.gammas)
    if 1. not in gammas or len(set(gammas)) != len(gammas) or any(not math.isfinite(g) or g <= 0 for g in gammas):
        raise ValueError('gammas must be distinct, positive and include 1')
    seeds = list(range(42, 42 + a.n_seeds))
    score_seeds = ints(a.score_seeds)
    targets = a.score_targets.split(',')
    import qtargets
    if not 3 <= a.n_seeds <= 32 or not set(score_seeds) <= set(seeds) or len(set(score_seeds)) != len(score_seeds):
        raise ValueError('Need 3..32 seeds starting at 42; score seeds inside the panel')
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(qtargets.TARGETS):
        raise ValueError('Unknown score target')
    if a.m < 8 or not 0 < a.retain < 1 or not 1 <= a.concurrency <= 32 or not 7.5 <= a.pack_minutes <= 25:
        raise ValueError('Invalid m/retain/concurrency/pack budget')
    c = rt.prepare_base(a, SCHEMA)
    s32 = Path(a.seed32_out).resolve()
    rt.disjoint(c['out'], s32)
    cfg = check_seed32(c, s32, seeds)
    for rel in rt.REUSED + ['experiments/pathway2_seed32_v2/scheduler.py']:
        c['source_sha256'][str(rt.REPO / rel)] = rt.sha(rt.REPO / rel)
    for d in (rt.LDS_V1, rt.STABLE_V1):
        for p in sorted(d.glob('*.py')):
            c['source_sha256'][str(p)] = rt.sha(p)
    c.update(seed32_out=str(s32), baseline_out=cfg['baseline_out'], lds_pilot_out=cfg['lds_pilot_out'],
             seeds=seeds, epochs=140, recipe=RECIPE, design_name=f'img{a.m}',
             design=dict(kind='image_random', m=a.m, retain=a.retain, seed=a.design_seed),
             score_seeds=score_seeds, score_targets=targets, gammas=gammas,
             lanczos_max_steps=a.lanczos_max, lanczos_est_tol=1e-5, probe_steps=a.probe_steps, cg_fail_tol=1e-3,
             harvest_tol=1e-10, cross_gpu_tol=1e-7, concurrency=a.concurrency, pack_minutes=a.pack_minutes,
             train_partition=a.train_partition, fp64_partition=a.fp64_partition, score_time=a.score_time,
             primary='image-mean heldout CE (2,298 query images equally weighted), raw endpoint; '
                     'Brier and 1-p_y exploratory; patient-macro weighting secondary',
             holdout_policy='Design seed, subsets, seeds, gammas and targets are fixed here, before outcomes.')
    c['cells'] = make_cells(c)
    c['packs'] = make_packs(c['cells'], a.pack_minutes)
    c['proxy_tasks'] = seeds
    c['score_tasks'] = score_seeds
    rt.save_config(c)
    train = [x for x in c['cells'] if x['source'] == 'train']
    print(f"cells={len(c['cells'])} (train {len(train)}, est {sum(x['minutes'] for x in train)/60:.1f} GPU-h on "
          f"{a.train_partition}); packs={ {g: len(p) for g, p in c['packs'].items()} }; "
          f"proxy tasks={len(seeds)}, score tasks={len(score_seeds)} on {a.fp64_partition}")


def config(out):
    return rt.config(out, SCHEMA)


# ------------------------------------------------------------------ Slurm
def launcher(c, ledger):
    out = Path(c['out'])

    def launch(mode, extra, partition, label=None):
        label = label or mode.replace(':', '_')
        cmd = ['sbatch', '--parsable', '--account=' + c['account'], '--partition=' + partition,
               '--job-name=oct_img3_' + label, '--output=' + str(out / 'logs/%x-%A_%a.out'),
               '--error=' + str(out / 'logs/%x-%A_%a.err'), *extra, str(rt.HERE / 'job.slurm'),
               str(rt.HERE), str(out), mode]
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


def walltime(c, mode, a):
    if mode == 'score':
        return getattr(a, 'score_time', None) or c['score_time']
    return TIMES['pack' if mode.startswith('pack') else mode]


def partition(c, mode, a):
    fp64 = getattr(a, 'fp64_partition', None) or c['fp64_partition']
    return fp64 if mode in ('proxy', 'score') else c['train_partition']


@submission_lock
def submit(a):
    c = config(a.out)
    out = Path(c['out'])
    after = lambda j, kind='afterok': ['--dependency=' + kind + ':' + j, '--kill-on-invalid-dep=yes']
    scheduler.receipts(out)
    with (out / 'submission.jsonl').open('x') as ledger:
        launch = launcher(c, ledger)
        jobs = dict(smoke=launch('smoke', ['--time=' + TIMES['smoke']], c['train_partition']))
        for group, packs in c['packs'].items():
            jobs['pack:' + group] = launch('pack:' + group, ['--time=' + TIMES['pack'], array(list(range(len(packs))), c['concurrency']),
                                                            *after(jobs['smoke'])], c['train_partition'])
        jobs['proxy'] = launch('proxy', ['--time=' + TIMES['proxy'], array(list(range(len(c['proxy_tasks']))), c['concurrency']),
                                         *after(jobs['pack:full'], 'afterany')], partition(c, 'proxy', a))
        if not a.defer_score:
            jobs['score'] = launch('score', ['--time=' + walltime(c, 'score', a), array(list(range(len(c['score_tasks']))), c['concurrency']),
                                             *after(jobs['smoke'])], partition(c, 'score', a))
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print('\n'.join(f'{k}={v}' for k, v in jobs.items()))


def incomplete(c):
    out = Path(c['out'])
    h = rt.sha(out / 'config.json')

    def done(d, cell):
        if not (d / 'complete.json').exists():
            return False
        rt.completed(d, h, cell)
        return True
    cells = {cell['name']: cell for cell in c['cells']}
    todo = {}
    for group, packs in c['packs'].items():
        todo['pack:' + group] = [i for i, p in enumerate(packs) if not all(done(out / 'cells' / n, cells[n]) for n in p)]
    todo['proxy'] = [i for i, s in enumerate(c['proxy_tasks']) if not done(out / 'proxy' / f'seed{s}', dict(kind='proxy', seed=s))]
    todo['score'] = []
    for i, s in enumerate(c['score_tasks']):
        names = list(c['score_targets']) + (['check'] if s in STABLE_SEEDS else [])
        if not all(done(out / 'score' / f'seed{s}_{t}', dict(kind='score', seed=s, target=t)) for t in names):
            todo['score'].append(i)
    return todo


def status(a):
    c = config(a.out)
    out = Path(c['out'])
    smoke = (out / 'smoke/complete.json').exists()
    print('smoke:', 'complete' if smoke else 'NOT complete')
    if smoke:
        rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='smoke'))
    for k, v in incomplete(c).items():
        total = len(c['packs'][k.split(':')[1]]) if k.startswith('pack:') else len(c[k + '_tasks'])
        print(f'{k}: {total - len(v)}/{total} complete; incomplete indices {v[:20]}{" ..." if len(v) > 20 else ""}')
    for p in sorted((out / 'score').glob('seed*/result.json')):
        r = rt.read(p)
        if 'gate' in r:
            print(p.parent.name, 'patient gate passed:', r['gate']['passed'], 'rel', r['gate']['rel_l2_difference'])
        elif (p.parent / 'complete.json').exists():
            print(p.parent.name, 'qualified gammas:', [g['gamma'] for g in r['gammas'] if g['qualified']],
                  'unqualified:', [g['gamma'] for g in r['gammas'] if not g['qualified']])


@submission_lock
def resubmit(a):
    c = config(a.out)
    out = Path(c['out'])
    previous = scheduler.receipts(out)
    modes = ['smoke', 'proxy', 'score', *('pack:' + g for g in c['packs'])]
    selected = a.only.split(',') if a.only else None
    if selected and any(m not in modes for m in selected):
        raise ValueError('Unknown --only mode')
    if selected == ['smoke']:
        scheduler.require_terminal([r for r in previous if r['mode'] == 'smoke'])
        if (out / 'smoke/complete.json').exists():
            print('Smoke already complete')
            return
        with (out / 'submission.jsonl').open('a') as ledger:
            print('smoke', launcher(c, ledger)('smoke', ['--time=' + TIMES['smoke']], c['train_partition'], 'smoke_re'))
        return
    rt.completed(out / 'smoke', rt.sha(out / 'config.json'))
    todo = incomplete(c)
    selected = selected or [m for m in todo if todo[m]]
    scheduler.require_terminal([r for r in previous if r['mode'] in selected])
    with (out / 'submission.jsonl').open('a') as ledger:
        launch = launcher(c, ledger)
        for mode in selected:
            idx = todo.get(mode, [])
            if not idx:
                continue
            deps = []
            if mode == 'proxy':
                full = [int(x[len('full_seed'):]) for p in c['packs']['full'] for x in p]
                missing = [s for s in full if not (out / 'cells' / f'full_seed{s}' / 'complete.json').exists()]
                if missing:
                    raise ValueError(f'Full cells incomplete for seeds {missing}; resubmit pack:full first')
            job = launch(mode, ['--time=' + walltime(c, mode, a), array(idx, c['concurrency']), *deps],
                         partition(c, mode, a), mode.replace(':', '_') + '_re')
            print(mode, job, idx)


def queue_report(a):
    """Queue wait = Start - Submit, from sacct (same accounting as seed32)."""
    from datetime import datetime
    c = config(a.out)
    ids = [r['job_id'] for r in scheduler.receipts(c['out'])]
    if not ids:
        raise ValueError('No submitted jobs to query')
    cmd = ['sacct', '-j', ','.join(ids), '-X', '-P', '--format=JobID,JobName,Partition,Submit,Start,End,Elapsed,State']
    rows = list(csv.DictReader(subprocess.check_output(cmd, text=True).splitlines(), delimiter='|'))
    parse = lambda s: datetime.fromisoformat(s) if s and s not in ('Unknown', 'None') else None
    import numpy as np
    by = {}
    for r in rows:
        if '[' in r['JobID'] or '.' in r['JobID']:
            continue
        sub, st = parse(r['Submit']), parse(r['Start'])
        x = by.setdefault((r['JobName'], r['Partition']), [])
        if sub and st:
            x.append((st - sub).total_seconds() / 60)
    report = {f'{k[0]} @ {k[1]}': dict(tasks=len(v), wait_minutes_median=float(np.median(v)) if v else None,
                                       wait_minutes_max=float(np.max(v)) if v else None) for k, v in by.items()}
    print(json.dumps(report, indent=2))


# -------------------------------------------------------------------- GPU
def ensure_designs(root, h):
    rt.completed(root / 'smoke', h)
    src, dst = root / 'smoke/designs.json', root / 'designs.json'
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise ValueError('Frozen designs differ from smoke manifest')
        return
    tmp = root / f'.designs-{uuid.uuid4().hex}.json'
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


def gpu(a):
    from core3 import Session
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}"
    started = time.monotonic()
    mode, _, group = a.mode.partition(':')
    if mode == 'smoke':
        final = root / 'smoke'
        if final.exists():
            rt.completed(final, h, dict(kind='smoke'))
            ensure_designs(root, h)
            print('Smoke already complete')
            return
        with rt.lock(root / '.locks' / 'smoke.lock'):
            tmp = root / f'.smoke.tmp-{tag}'
            tmp.mkdir()
            session = Session(c, freeze_to=tmp / 'designs.json')
            session.smoke(tmp)
            config(a.out)
            rt.finish(tmp, c, dict(kind='smoke'), elapsed_seconds=time.monotonic() - started)
            os.rename(tmp, final)
        ensure_designs(root, h)
        print('COMPLETE smoke', flush=True)
        return
    ensure_designs(root, h)
    session = Session(c)
    if mode == 'pack':
        cells = {x['name']: x for x in c['cells']}
        for n in c['packs'][group][a.index]:
            t0 = time.monotonic()
            print(n, session.run_cell(cells[n], tag), f'{time.monotonic() - t0:.0f}s', flush=True)
    elif mode == 'proxy':
        s = c['proxy_tasks'][a.index]
        print(f'proxy seed{s}', session.proxy_seed(s, tag), flush=True)
    elif mode == 'score':
        s = c['score_tasks'][a.index]
        print(f'score seed{s}', session.score_seed(s, tag), flush=True)
    else:
        raise ValueError('Unknown GPU mode')
    config(a.out)
    print(f'COMPLETE {a.mode} index={a.index} elapsed={time.monotonic() - started:.0f}s', flush=True)


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
    import analysis3
    import figures3
    import report3
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    ensure_designs(root, h)
    data = analysis3.load(root, c, allow_partial=a.allow_partial)
    summary, tables = analysis3.analyze(data, c)
    summary = clean(summary)
    partial = bool(data['missing'])
    rep = root / ('report' if not partial else 'report_partial_' + time.strftime('%Y%m%dT%H%M%S'))
    rep.mkdir(exist_ok=False)
    rt.write(rep / 'summary.json', summary)
    for key, rows in tables.items():
        if rows:
            rt.csv_write(rep / f'table_{key}.csv', clean(rows))
    figs = figures3.make_all(summary, rep)
    (rep / 'REPORT_CN.md').write_text(report3.render(summary, c, figs, partial))
    rt.write(rep / 'complete.json', dict(status='partial' if partial else 'complete', config_sha256=h, missing=data['missing']))
    archive(root, rep)


def archive(root, rep):
    target = Path(str(root) + '.' + rep.name + '.tar.gz')
    keep = ('.json', '.jsonl', '.csv', '.md', '.txt', '.sha256', '.out', '.err', '.png', '.npz')
    with tarfile.open(target, 'x:gz') as t:
        for p in sorted(root.rglob('*')):
            rel = p.relative_to(root)
            if not p.is_file() or any(part.startswith('.') for part in rel.parts):
                continue
            if rel.parts[0].startswith('report') and rel.parts[0] != rep.name:
                continue
            if p.suffix in keep:
                t.add(p, arcname=str(Path(root.name) / rel))
    print(str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--out', required=True)
    q.add_argument('--audit', default=rt.AUDIT)
    q.add_argument('--seed32-out', default=SEED32_OUT)
    q.add_argument('--n-seeds', type=int, default=8, help='seeds 42..; full baselines are reused from seed32 (max 32)')
    q.add_argument('--m', type=int, default=32, help='number of image subsets')
    q.add_argument('--retain', type=float, default=.5)
    q.add_argument('--design-seed', type=int, default=20261009)
    q.add_argument('--score-seeds', default='42,43,44')
    q.add_argument('--score-targets', default='ce,oneminusp,brier')
    q.add_argument('--gammas', default='1,0.3,0.1,0.03')
    q.add_argument('--lanczos-max', type=int, default=200)
    q.add_argument('--probe-steps', type=int, default=24)
    q.add_argument('--train-partition', default='gpuA40x4', help='must match the baseline GPU type for exact no-op replay')
    q.add_argument('--fp64-partition', default='gpuA100x4', help='proxy and IF score jobs (fp64 heavy)')
    q.add_argument('--score-time', default='04:00:00')
    q.add_argument('--pack-minutes', type=float, default=20.)
    q.add_argument('--concurrency', type=int, default=8)
    for name in ('submit', 'status', 'resubmit', 'collect', 'gpu', 'queue-report'):
        q = sub.add_parser(name)
        q.add_argument('--out', required=True)
        if name == 'gpu':
            q.add_argument('--mode', required=True)
            q.add_argument('--index', type=int, default=0)
        if name == 'collect':
            q.add_argument('--allow-partial', action='store_true')
        if name == 'submit':
            q.add_argument('--defer-score', action='store_true')
        if name == 'resubmit':
            q.add_argument('--only', default=None, help='comma list, e.g. pack:img32,proxy,score')
        if name in ('submit', 'resubmit'):
            q.add_argument('--score-time', default=None, help='override score walltime')
            q.add_argument('--fp64-partition', default=None, help='override proxy/score partition, e.g. gpuA40x4')
    a = p.parse_args()
    globals()[a.action.replace('-', '_')](a)


if __name__ == '__main__':
    main()
