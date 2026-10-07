#!/usr/bin/env python3
"""Stage-1 seed32 / deleter-LDS / size-controlled designs / gamma sweep / per-query LDS.

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
sys.dont_write_bytecode = True
import runtime as rt
import scheduler
from functools import wraps

SCHEMA = 'OCT_seed32_lds_v2'
BASELINE = '/u/yli103/oct2-pathway2-runs/A/stable_training_v1_baseline_20260928T181445049797182Z'
LDS_PILOT = '/u/yli103/oct2-pathway2-runs/A/patient_lds_formula_v1_20260929T200328953925Z'
RECIPE = dict(lr=.001, warmup_frac=.1, ema_decay=.99, wd=1e-5, batch=128)
PILOT_SEEDS = (42, 43, 44)
MINUTES = dict(harvest=.5, full=7.5, half=3.3)


def floats(text):
    return [float(x) for x in text.split(',') if x]


def ints(text):
    return [int(x) for x in text.split(',') if x]


def check_baseline(c, bpath):
    b = rt.read(bpath / 'config.json')
    bh = rt.sha(bpath / 'config.json')
    if bh != (bpath / 'config.sha256').read_text().strip():
        raise ValueError('Baseline config checksum changed')
    if (b['schema'] != 'OCT_stable_training_v1' or b['profile'] != 'baseline' or b['epochs'] != 140
            or b['recipe'] != RECIPE or b['git_commit'] != rt.STABLE_COMMIT
            or b['dataset_fingerprint'] != c['dataset_fingerprint']):
        raise ValueError('Need the frozen b14b69e stable baseline, epochs=140')
    report = rt.read(bpath / 'report/complete.json')
    if report['config_sha256'] != bh or report.get('noop_passed') is not True:
        raise ValueError('Stable baseline must be collected with a passed no-op')
    rt.verify_hashes(b['input_sha256'])
    for rel in ('config.json', 'config.sha256', 'report/complete.json'):
        c['input_sha256'][str(bpath / rel)] = rt.sha(bpath / rel)
    for seed in PILOT_SEEDS:
        d = bpath / f'seed{seed}_baseline'
        rt.completed(d, bh)
        for rel in ('complete.json', 'final.pt', 'interface.npz'):
            c['input_sha256'][str(d / rel)] = rt.sha(d / rel)


def check_pilot(c, ppath, bpath):
    p = rt.read(ppath / 'config.json')
    ph = rt.sha(ppath / 'config.json')
    if ph != (ppath / 'config.sha256').read_text().strip():
        raise ValueError('LDS pilot config checksum changed')
    if (p['schema'] != 'OCT_patient_LDS_formula_v1' or p['mode'] != 'subset' or p['m'] != 8
            or p['retain'] != .5 or p['alpha'] != 1. or p['seeds'] != list(PILOT_SEEDS)
            or p['epochs'] != 140 or p['recipe'] != RECIPE or p['subset_seed'] != 20260929
            or p.get('gamma') != 1.
            or Path(p['baseline_out']).resolve() != bpath or p['dataset_fingerprint'] != c['dataset_fingerprint']):
        raise ValueError('LDS pilot is not the frozen 8-subset x 3-seed subset-mode run')
    report = rt.read(ppath / 'report/complete.json')
    if report.get('status') != 'complete' or report['config_sha256'] != ph:
        raise ValueError('LDS pilot must be collected first')
    # The reused pure modules must be byte-identical to what the pilot ran.
    for rel in ('experiments/pathway2_lds_v1/design.py', 'experiments/pathway2_lds_v1/analysis_stats.py',
                'experiments/pathway2_stable_v1/stable_core.py'):
        recorded = {str(Path(k).relative_to(p['repo'])): v for k, v in p['source_sha256'].items()}
        if recorded.get(rel) != rt.sha(rt.REPO / rel):
            raise ValueError('Reused module differs from the LDS pilot run: ' + rel)
    dirs = [ppath / f'score_seed{s}' for s in PILOT_SEEDS] + [ppath / x['name'] for x in p['cells']]
    for d in dirs:
        rt.completed(d, ph)
        for rel in ('complete.json', 'final.pt', 'interface.npz', 'design.json', 'scores.csv'):
            if (d / rel).exists():
                c['input_sha256'][str(d / rel)] = rt.sha(d / rel)
    for rel in ('config.json', 'config.sha256', 'report/complete.json'):
        c['input_sha256'][str(ppath / rel)] = rt.sha(ppath / rel)


def make_cells(c):
    cells = []
    for s in c['seeds']:
        cells.append(dict(name=f'full_seed{s}', group='full', kind='full', seed=s,
                          source='harvest' if s in PILOT_SEEDS else 'train',
                          minutes=MINUTES['harvest'] if s in PILOT_SEEDS else MINUTES['full']))
    for name, spec in c['designs'].items():
        frac = 1 - spec['n_delete'] / 148 if spec['kind'] == 'smalldel' else spec['retain']
        for s in c['design_seeds'][name]:  # seed-major: early packs finish whole seeds first
            for j in range(spec['m']):
                harvest = spec['kind'] == 'pilot' and s in PILOT_SEEDS
                cells.append(dict(name=f'{name}_s{j:03d}_seed{s}', group=name, kind='subset', design=name,
                                  subset=j, seed=s, source='harvest' if harvest else 'train',
                                  minutes=MINUTES['harvest'] if harvest else max(MINUTES['half'], MINUTES['full'] * frac)))
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
    if not 1 <= a.concurrency <= 32:
        raise ValueError('Concurrency must be 1..32')
    if a.smalldel and (not a.mentor_note or '<' in a.mentor_note or '>' in a.mentor_note):
        raise ValueError('smalldel needs --mentor-note (mentor sign-off), without < or >')
    gammas = floats(a.gammas)
    if 1. not in gammas or len(set(gammas)) != len(gammas) or any(not math.isfinite(g) or g <= 0 for g in gammas):
        raise ValueError('gammas must be positive and include 1 (link to LDS v1)')
    score_seeds = ints(a.score_seeds)
    if not score_seeds or len(set(score_seeds)) != len(score_seeds):
        raise ValueError('Score seeds must be nonempty and distinct')
    if a.n_seeds < 3 or a.first_seed != 42 or not 0 <= a.eq_seeds <= a.n_seeds:
        raise ValueError('Need seeds starting at 42/43/44; eq-seeds inside the panel')
    if (a.eq_m < 3 or a.lanczos_max < 1 or a.probe_steps < 1
            or not math.isfinite(a.pack_minutes) or not 7.5 <= a.pack_minutes <= 25
            or any(not math.isfinite(v) or not 0 <= v < 1 for v in (a.eq_total_tol, a.eq_class_tol))):
        raise ValueError('Invalid design tolerance, size, or solver/pack budget')
    c = rt.prepare_base(a, SCHEMA)
    bpath, ppath = Path(a.baseline_out).resolve(), Path(a.lds_pilot_out).resolve()
    rt.disjoint(c['out'], bpath, ppath)
    check_baseline(c, bpath)
    check_pilot(c, ppath, bpath)
    rt.verify_hashes(c['input_sha256'])
    for d in (rt.LDS_V1, rt.STABLE_V1):
        for p in sorted(d.glob('*.py')):
            c['source_sha256'][str(p)] = rt.sha(p)
    for seed in PILOT_SEEDS:
        p = Path(c['full']) / f'stage1_order_seed{seed}.npz'
        c['input_sha256'][str(p)] = rt.sha(p)
    seeds = list(range(a.first_seed, a.first_seed + a.n_seeds))
    if seeds[:3] != list(PILOT_SEEDS):
        raise ValueError('Seed panel must start at 42 so 42/43/44 are reused')
    designs = dict(pilot8=dict(kind='pilot', m=8, retain=.5))
    design_seeds = dict(pilot8=seeds)
    if a.eq_seeds > 0:
        designs['eq32'] = dict(kind='eq', m=a.eq_m, retain=.5, seed=20261006,
                               total_tol_frac=a.eq_total_tol, class_tol_frac=a.eq_class_tol, optimizer_batch=RECIPE['batch'])
        design_seeds['eq32'] = seeds[:a.eq_seeds]
    if a.smalldel:
        designs['smalldel'] = dict(kind='smalldel', m=a.smalldel_m, n_delete=a.smalldel_n, seed=20261007)
        design_seeds['smalldel'] = seeds[:a.smalldel_seeds]
    if not set(score_seeds) <= set(seeds):
        raise ValueError('Score seeds must be inside the seed panel')
    c.update(baseline_out=str(bpath), lds_pilot_out=str(ppath), seeds=seeds, epochs=140, recipe=RECIPE,
             designs=designs, design_seeds=design_seeds, mentor_note=a.mentor_note,
             score_seeds=score_seeds, score_targets=a.score_targets.split(','), gammas=gammas,
             gamma_pilot=dict(seeds=list(PILOT_SEEDS), target='oneminusp', gammas=[1., .3, .1]),
             lanczos_max_steps=a.lanczos_max, lanczos_est_tol=1e-5, probe_steps=a.probe_steps,
             cg_fail_tol=1e-3, concurrency=a.concurrency, pack_minutes=a.pack_minutes,
             score_time=a.score_time, proxy_per_task=4,
             primary='class-balanced patient-macro heldout CE, raw endpoint; 1-p_y and Brier exploratory',
             holdout_policy='Gammas, designs, seeds and the size-baseline sign are fixed here, before outcomes.')
    import targets
    if not c['score_targets'] or len(set(c['score_targets'])) != len(c['score_targets']) or not set(c['score_targets']) <= set(targets.TARGETS):
        raise ValueError('Unknown score target')
    c['cells'] = make_cells(c)
    c['packs'] = make_packs(c['cells'], a.pack_minutes)
    c['score_tasks'] = [dict(seed=s, target=t) for t in c['score_targets'] for s in score_seeds]
    c['proxy_tasks'] = [seeds[i:i + c['proxy_per_task']] for i in range(0, len(seeds), c['proxy_per_task'])]
    rt.save_config(c)
    train = [x for x in c['cells'] if x['source'] == 'train']
    print(f"cells={len(c['cells'])} (train {len(train)}, est {sum(x['minutes'] for x in train)/60:.1f} GPU-h); "
          f"packs={ {g: len(p) for g, p in c['packs'].items()} }; score tasks={len(c['score_tasks'])}; "
          f"proxy tasks={len(c['proxy_tasks'])}")


def config(out):
    return rt.config(out, SCHEMA)


# ------------------------------------------------------------------ Slurm
def launcher(c, ledger):
    out = Path(c['out'])

    def launch(mode, extra, label=None):
        label = label or mode.replace(':', '_')
        cmd = ['sbatch', '--parsable', '--account=' + c['account'], '--partition=gpuA40x4',
               '--job-name=oct_s32_' + label, '--output=' + str(out / 'logs/%x-%A_%a.out'),
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
    if not indices:
        return None
    spec = (f'0-{indices[-1]}' if indices == list(range(indices[-1] + 1)) else ','.join(map(str, indices)))
    return f'--array={spec}%{conc}'


def submission_lock(fn):
    @wraps(fn)
    def wrapped(a):
        with rt.lock(Path(a.out) / '.locks' / 'submission.lock'):
            return fn(a)
    return wrapped


@submission_lock
def submit(a):
    c = config(a.out)
    out = Path(c['out'])
    if getattr(a, 'score_time', None):
        c = dict(c, score_time=a.score_time)  # walltime only; not part of the frozen science
    after = lambda j, kind='afterok': ['--dependency=' + kind + ':' + j, '--kill-on-invalid-dep=yes']
    scheduler.receipts(out)
    with (out / 'submission.jsonl').open('x') as ledger:
        launch = launcher(c, ledger)
        jobs = dict(smoke=launch('smoke', ['--time=01:00:00']))
        for group, packs in c['packs'].items():
            jobs['pack:' + group] = launch('pack:' + group, ['--time=00:30:00', array(list(range(len(packs))), c['concurrency']),
                                                            *after(jobs['smoke'])])
        if not getattr(a, 'defer_score', False):
            jobs['score'] = launch('score', ['--time=' + c['score_time'], array(list(range(len(c['score_tasks']))), c['concurrency']),
                                             *after(jobs['smoke'] if all(t['seed'] in PILOT_SEEDS for t in c['score_tasks']) else jobs['pack:full'])])
        # Proxies need the new full baselines; afterany so one failed pack never blocks them silently.
        jobs['proxy'] = launch('proxy', ['--time=00:45:00', array(list(range(len(c['proxy_tasks']))), c['concurrency']),
                                         *after(jobs['pack:full'], 'afterany')])
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print('\n'.join(f'{k}={v}' for k, v in jobs.items()))


def incomplete(c):
    out = Path(c['out'])
    h = rt.sha(out / 'config.json')
    def done(d, cell=None):
        if not (d / 'complete.json').exists():
            return False
        rt.completed(d, h, cell)
        return True
    cells = {cell['name']: cell for cell in c['cells']}
    todo = {}
    for group, packs in c['packs'].items():
        todo['pack:' + group] = [i for i, p in enumerate(packs) if not all(done(out / 'cells' / n, cells[n]) for n in p)]
    todo['proxy'] = [i for i, seeds in enumerate(c['proxy_tasks']) if not all(done(out / 'proxy' / f'seed{s}', dict(kind='proxy', seed=s)) for s in seeds)]
    todo['score'] = [i for i, t in enumerate(c['score_tasks']) if not done(out / 'score' / f"seed{t['seed']}_{t['target']}", dict(kind='score', **t))]
    return todo


def status(a):
    c = config(a.out)
    out = Path(c['out'])
    print('smoke:', 'complete' if (out / 'smoke/complete.json').exists() else 'NOT complete')
    if (out / 'smoke/complete.json').exists():
        rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='smoke'))
    for k, v in incomplete(c).items():
        total = len(c['packs'][k.split(':')[1]]) if k.startswith('pack:') else len(c[k + '_tasks'])
        print(f'{k}: {total - len(v)}/{total} complete; incomplete indices {v[:20]}{" ..." if len(v) > 20 else ""}')
    for t in c['score_tasks']:
        p = out / 'score' / f"seed{t['seed']}_{t['target']}"
        if (p / 'complete.json').exists():
            r = rt.read(p / 'result.json')
            print(p.name, 'qualified gammas:', [g['gamma'] for g in r['gammas'] if g['qualified']],
                  'unqualified:', [g['gamma'] for g in r['gammas'] if not g['qualified']])
    for p in sorted((out / 'score_pilot').glob('*/complete.json')):
        rt.completed(p.parent, rt.sha(out / 'config.json'))
        r = rt.read(p.parent / 'result.json')
        print('pilot', p.parent.name, [(g['gamma'], g['qualified'], g['true_rel_residual']) for g in r['gammas']])


@submission_lock
def resubmit(a):
    c = config(a.out)
    out = Path(c['out'])
    if getattr(a, 'score_time', None):
        c = dict(c, score_time=a.score_time)
    previous = scheduler.receipts(out)
    selected = a.only.split(',') if a.only else list(incomplete(c))
    if any(m not in ['smoke', 'proxy', 'score', 'score-pilot', *('pack:' + g for g in c['packs'])] for m in selected):
        raise ValueError('Unknown --only mode')
    if any(m in selected for m in ('smoke', 'score-pilot')) and len(selected) != 1:
        raise ValueError('Submit smoke or score-pilot as its own phase')
    scheduler.require_terminal([r for r in previous if r['mode'] in selected])
    if selected == ['smoke']:
        if (out / 'smoke/complete.json').exists():
            rt.completed(out / 'smoke', rt.sha(out / 'config.json'))
            print('Smoke already complete')
            return
        with (out / 'submission.jsonl').open('a') as ledger:
            print('smoke', launcher(c, ledger)('smoke', ['--time=01:00:00']))
        return
    rt.completed(out / 'smoke', rt.sha(out / 'config.json'))
    if selected == ['score-pilot']:
        pilot = c['gamma_pilot']
        idx = []
        for i, s in enumerate(pilot['seeds']):
            d = out / 'score_pilot' / f"seed{s}_{pilot['target']}"
            if (d / 'complete.json').exists():
                rt.completed(d, rt.sha(out / 'config.json'), dict(kind='score', seed=s, target=pilot['target']))
            else:
                idx.append(i)
        if idx:
            with (out / 'submission.jsonl').open('a') as ledger:
                print('score-pilot', launcher(c, ledger)('score-pilot', ['--time=' + c['score_time'], array(idx, c['concurrency'])]))
        return
    todo = incomplete(c)
    new_full = None
    with (out / 'submission.jsonl').open('a') as ledger:
        launch = launcher(c, ledger)
        for mode, idx in todo.items():
            if mode not in selected:
                continue
            if mode == 'score' and getattr(a, 'score_targets', None):
                names = a.score_targets.split(',')
                if not set(names) <= {t['target'] for t in c['score_tasks']}:
                    raise ValueError('Unknown score target filter')
                idx = [i for i in idx if c['score_tasks'][i]['target'] in names]
            if idx:
                t = c['score_time'] if mode == 'score' else ('00:45:00' if mode == 'proxy' else '00:30:00')
                deps = []
                needs_full = mode == 'proxy' or (mode == 'score' and any(c['score_tasks'][i]['seed'] not in PILOT_SEEDS for i in idx))
                if needs_full:
                    full_jobs = scheduler.active_ids([r for r in previous if r['mode'] == 'pack:full'])
                    if new_full:
                        full_jobs.append(new_full)
                    if full_jobs:
                        deps = ['--dependency=afterany:' + ':'.join(full_jobs), '--kill-on-invalid-dep=yes']
                    else:
                        seeds = [s for i in idx for s in c['proxy_tasks'][i]] if mode == 'proxy' else [c['score_tasks'][i]['seed'] for i in idx if c['score_tasks'][i]['seed'] not in PILOT_SEEDS]
                        for s in seeds:
                            rt.completed(out / 'cells' / f'full_seed{s}', rt.sha(out / 'config.json'))
                job = launch(mode, ['--time=' + t, array(idx, c['concurrency']), *deps], label=mode.replace(':', '_') + '_re')
                if mode == 'pack:full':
                    new_full = job
                print(mode, job, idx)


def queue_report(a):
    """Task-1 bookkeeping: queue wait = Start - Submit, from sacct."""
    from datetime import datetime
    if a.since:
        cmd = ['sacct', '-u', os.environ.get('USER', ''), '-S', a.since, '-X', '-P',
               '--format=JobID,JobName,Submit,Start,End,Elapsed,State,AllocTRES']
    else:
        c = config(a.out)
        ids = [r['job_id'] for r in scheduler.receipts(c['out'])]
        if not ids:
            raise ValueError('No submitted jobs to query')
        cmd = ['sacct', '-j', ','.join(ids), '-X', '-P', '--format=JobID,JobName,Submit,Start,End,Elapsed,State,AllocTRES']
    rows = list(csv.DictReader(subprocess.check_output(cmd, text=True).splitlines(), delimiter='|'))
    def parse(s):
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None
    def secs(e):
        if not e or e in ('Unknown', 'None'):
            return None
        d, _, hms = e.rpartition('-')
        h, m, s = (['0'] * 3 + hms.split(':'))[-3:]
        return int(d or 0) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    by = {}
    parents = {r['JobID'].split('_')[0] for r in rows if '_' in r['JobID']}
    for r in rows:
        if r['JobID'] in parents or '.' in r['JobID'] or '[' in r['JobID']:
            continue  # array parent/step/pending range is not another allocated GPU task
        sub, st = parse(r['Submit']), parse(r['Start'])
        x = by.setdefault(r['JobName'], dict(n=0, wait=[], run=[], gpu_hours=[], states={}))
        x['n'] += 1
        x['states'][r['State']] = x['states'].get(r['State'], 0) + 1
        if sub and st:
            x['wait'].append((st - sub).total_seconds() / 60)
        elapsed = secs(r['Elapsed'])
        if elapsed is not None and st:
            x['run'].append(elapsed / 60)
            alloc = dict(item.split('=', 1) for item in r.get('AllocTRES', '').split(',') if '=' in item)
            gpu = alloc.get('gres/gpu')
            if gpu is None:
                typed = [int(v) for k, v in alloc.items() if k.startswith('gres/gpu:')]
                gpu = sum(typed) if typed else (0 if alloc else None)
            if gpu is not None:
                x['gpu_hours'].append(elapsed * int(gpu) / 3600)
    import numpy as np
    report = {k: dict(tasks=v['n'], states=v['states'],
                      wait_minutes_median=float(np.median(v['wait'])) if v['wait'] else None,
                      wait_minutes_max=float(np.max(v['wait'])) if v['wait'] else None,
                      run_minutes_median=float(np.median(v['run'])) if v['run'] else None,
                      gpu_hours=float(np.sum(v['gpu_hours'])) if v['gpu_hours'] else None,
                      gpu_hours_known_tasks=len(v['gpu_hours'])) for k, v in by.items()}
    print(json.dumps(report, indent=2))
    if a.out and not a.since:
        rt.write(Path(a.out) / f'queue_report_{time.strftime("%Y%m%dT%H%M%S")}.json', report)


# -------------------------------------------------------------------- GPU
def gpu(a):
    from core2 import Session
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
        tmp = root / f'.smoke.tmp-{tag}'
        tmp.mkdir()
        session = Session(c, freeze_to=tmp / 'designs.json')
        session.smoke(tmp)
        config(a.out)
        rt.finish(tmp, c, dict(kind='smoke'), elapsed_seconds=time.monotonic() - started)
        os.rename(tmp, final)
        # Every later job verifies its regenerated designs against this file.
        ensure_designs(root, h)
        print('COMPLETE smoke', flush=True)
        return
    rt.completed(root / 'smoke', h)
    ensure_designs(root, h)
    session = Session(c)
    if mode == 'pack':
        names = c['packs'][group][a.index]
        cells = {x['name']: x for x in c['cells']}
        for n in names:
            t0 = time.monotonic()
            print(n, session.run_cell(cells[n], tag), f'{time.monotonic() - t0:.0f}s', flush=True)
    elif mode == 'proxy':
        for s in c['proxy_tasks'][a.index]:
            t0 = time.monotonic()
            print(f'proxy seed{s}', session.proxy_seed(s, tag), f'{time.monotonic() - t0:.0f}s', flush=True)
    elif mode == 'score':
        t = c['score_tasks'][a.index]
        print(f"score seed{t['seed']} {t['target']}", session.score_task(t['seed'], t['target'], tag), flush=True)
    elif mode == 'score-pilot':
        p = c['gamma_pilot']
        print('score-pilot', session.score_task(p['seeds'][a.index], p['target'], tag, pilot=True), flush=True)
    else:
        raise ValueError('Unknown GPU mode')
    config(a.out)
    print(f'COMPLETE {a.mode} index={a.index} elapsed={time.monotonic() - started:.0f}s', flush=True)


# ---------------------------------------------------------------- collect
def ensure_designs(root, h):
    """Recover the convenience copy after a smoke rename/driver interruption."""
    rt.completed(root / 'smoke', h)
    src, dst = root / 'smoke/designs.json', root / 'designs.json'
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise ValueError('Frozen designs differ from smoke manifest')
        return
    tmp = root / f'.designs-{uuid.uuid4().hex}.json'
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


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
    import analysis
    import figures
    import report_md
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    rt.completed(root / 'smoke', h)
    ensure_designs(root, h)
    data = analysis.load(root, c, allow_partial=a.allow_partial)
    summary, tables, truths = analysis.analyze(data, c)
    summary = clean(summary)
    partial = bool(data['missing_artifacts'])
    name = 'report' if not partial else 'report_partial_' + time.strftime('%Y%m%dT%H%M%S')
    rep = root / name
    rep.mkdir(exist_ok=False)
    rt.write(rep / 'summary.json', summary)
    for key, rows in tables.items():
        if rows:
            rt.csv_write(rep / f'table_{key}.csv', clean(rows))
    figs = []
    figs += figures.task1(summary, rep)
    figs += figures.task3_size(truths, summary, rep)
    figs += figures.task4(summary, rep)
    figs += figures.task5(summary, rep)
    (rep / 'REPORT_CN.md').write_text(report_md.render(summary, c, figs, partial=partial))
    rt.write(rep / 'complete.json', dict(status='complete' if not partial else 'partial',
                                         config_sha256=h, missing_cells=data['missing'], missing_artifacts=data['missing_artifacts']))
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
    q.add_argument('--baseline-out', default=BASELINE)
    q.add_argument('--lds-pilot-out', default=LDS_PILOT)
    q.add_argument('--first-seed', type=int, default=42)
    q.add_argument('--n-seeds', type=int, default=32)
    q.add_argument('--eq-seeds', type=int, default=8, help='seeds for the 32 equal-size subsets; 0 disables task 3 training')
    q.add_argument('--eq-m', type=int, default=32)
    q.add_argument('--eq-total-tol', type=float, default=.02)
    q.add_argument('--eq-class-tol', type=float, default=.10)
    q.add_argument('--smalldel', action='store_true', help='optional 5-15 patient deletion design (mentor sign-off)')
    q.add_argument('--smalldel-n', type=int, default=10)
    q.add_argument('--smalldel-m', type=int, default=32)
    q.add_argument('--smalldel-seeds', type=int, default=4)
    q.add_argument('--mentor-note', default=None)
    q.add_argument('--score-seeds', default='42,43,44')
    q.add_argument('--score-targets', default='oneminusp,ce,brier')
    q.add_argument('--gammas', default='1,0.3,0.1,0.03')
    q.add_argument('--lanczos-max', type=int, default=160)
    q.add_argument('--probe-steps', type=int, default=24)
    q.add_argument('--score-time', default='04:00:00')
    q.add_argument('--pack-minutes', type=float, default=22.)
    q.add_argument('--concurrency', type=int, default=8)
    for name in ('submit', 'status', 'resubmit', 'collect', 'gpu', 'queue-report'):
        q = sub.add_parser(name)
        q.add_argument('--out', required=name != 'queue-report')
        if name == 'gpu':
            q.add_argument('--mode', required=True)
            q.add_argument('--index', type=int, default=0)
        if name == 'collect':
            q.add_argument('--allow-partial', action='store_true')
        if name == 'resubmit':
            q.add_argument('--only', default=None, help='comma list, e.g. pack:pilot8,proxy')
            q.add_argument('--score-targets', default=None, help='optional filter within frozen score tasks, e.g. oneminusp')
        if name == 'submit':
            q.add_argument('--defer-score', action='store_true',
                           help='submit the gamma-sweep later with resubmit --only score --score-time ...')
        if name in ('submit', 'resubmit'):
            q.add_argument('--score-time', default=None, help='override score walltime, e.g. 06:00:00 (from smoke timing)')
        if name == 'queue-report':
            q.add_argument('--since', default=None, help='YYYY-MM-DD: all of your jobs since this date')
    a = p.parse_args()
    globals()[a.action.replace('-', '_')](a)


if __name__ == '__main__':
    main()
