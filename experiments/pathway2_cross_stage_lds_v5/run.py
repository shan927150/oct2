#!/usr/bin/env python3
"""Cross-stage LDS ladder v5 (r1): v4 (cc0cad6) with one new CNN, a stable-recipe target.

v4 used five stable-recipe shadows and the legacy target; the attack (AUC 0.90) had CE 0.95-1.50 on
the primary classes. v5 trains only the target with the same stable recipe on its original split.
Background shadows are reused from the v4 run (hash-checked, interface replayed), and every attack
is evaluated on the stable target, the legacy target (v4 path, must reproduce v4) and the legacy
checkpoint on the fp64 path, so old and new targets are paired. Smoke adds a CE quality gate
(every primary-class J00 < ln 2). Truth cells add a projection-only control and a logit-linearized
L2-retrain; hybrid cells measure deleted / retained / nonmember / deleted+nonmember rows; a small
interpolation panel compares J(t) - J(0) with t * L1. Everything else is v4, reused byte-for-byte.

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

SCHEMA = 'OCT_cross_stage_lds_v5r1'
V4_SCHEMA = 'OCT_cross_stage_lds_v4'
V4_OUT = '/u/yli103/oct2-pathway2-runs/A/cross_stage_lds_v4_stable_20261013T044559507358835Z'
V3_SCHEMA = 'OCT_image_lds_v3'
V3_OUT = '/u/yli103/oct2-pathway2-runs/A/image_lds_v3_reviewed_20261010T033831362446826Z'
TIMES = dict(background='01:00:00', smoke='02:30:00', base='01:00:00', pack='00:50:00', score='02:00:00', interp='01:00:00',
             collect='00:30:00')
CELL_MINUTES = 2.4   # v4 truth cell median 55 s (24 attack fits); r1 truth 36 fits + hybrid 24 fits ~ 130-140 s
INTERP_TS = [1e-3, 1e-2, .1, .3, 1.]


def floats(text):
    return [float(x) for x in text.split(',') if x]


def ints(text):
    return [int(x) for x in text.split(',') if x]


def check_v3(c, v3):
    """The v3 run supplies every Stage-1 model; pin its config, designs and every cell manifest."""
    cfg = rt.read(v3 / 'config.json')
    h = rt.sha(v3 / 'config.json')
    if h != (v3 / 'config.sha256').read_text().strip():
        raise ValueError('v3 config checksum changed')
    if (cfg['schema'] != V3_SCHEMA or cfg['git_commit'] != rt.V3_COMMIT or cfg['design_name'] != 'img32'
            or cfg['dataset_fingerprint'] != c['dataset_fingerprint'] or cfg['full'] != c['full']
            or cfg['data_dir'] != c['data_dir']):
        raise ValueError('Need the reviewed image LDS v3 run (bffa3b8, img32, same data)')
    rt.completed(v3 / 'smoke', h, dict(kind='smoke'))
    recorded = {str(Path(k).relative_to(cfg['repo'])): v for k, v in cfg['source_sha256'].items()}
    for rel in rt.REUSED:
        if rel.startswith(('experiments/pathway2_image_lds_v3/', rt.V4_PREFIX)):
            continue  # v3's own modules are pinned below; v4 postdates v3 and is pinned by check_repo
        if recorded.get(rel) != rt.sha(rt.REPO / rel):
            raise ValueError('Reused module differs from the v3 run: ' + rel)
    for rel in ('core3.py', 'imgdesign.py', 'qtargets.py'):
        if recorded.get('experiments/pathway2_image_lds_v3/' + rel) != rt.sha(rt.V3 / rel):
            raise ValueError('v3 module differs from the v3 run: ' + rel)
    cells = [x for x in cfg['cells']]
    for cell in cells:
        d = v3 / 'cells' / cell['name']
        rt.completed(d, h, cell)           # verifies final.pt / interface.npz hashes in the manifest
        c['input_sha256'][str(d / 'complete.json')] = rt.sha(d / 'complete.json')
    if sorted({x['seed'] for x in cells}) != cfg['seeds'] or len(cells) != len(cfg['seeds']) * (1 + cfg['design']['m']):
        raise ValueError('v3 run is incomplete')
    # Upstream full-model states pinned by the v3 config (stable baselines 42-44, seed32 cells 45+).
    for k, v in cfg['input_sha256'].items():
        if k.endswith(('final.pt', 'design.json', 'scores.csv')) or '/cells/full_seed' in k:
            c['input_sha256'][k] = v
    for rel in ('config.json', 'config.sha256', 'designs.json', 'smoke/complete.json'):
        c['input_sha256'][str(v3 / rel)] = rt.sha(v3 / rel)
    rt.verify_hashes(c['input_sha256'])
    return cfg, h


def legacy_attack(c):
    """Attack hyper-parameters and frozen shadow/target checkpoints of the calibration v4.1 run."""
    full = Path(c['full'])
    cfg = rt.read(full / 'experiment_config.json')['args']
    n_shadow, aff = int(cfg['n_shadow']), int(cfg['affected_shadow'])
    if aff != 3:
        raise ValueError('Expected affected shadow 3')
    if 'target_seed' not in cfg:
        raise ValueError('Legacy config has no target_seed')
    files = [full / 'experiment_config.json', full / 'splits/fresh_patient_split.json', full / 'checkpoints/target_fixed.pt']
    files += [full / f'checkpoints/shadow_{s}_fixed.pt' for s in range(n_shadow) if s != aff]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        c['input_sha256'][str(p)] = rt.sha(p)
    return cfg, n_shadow


def check_v4(c, v4):
    """The v4 run supplies the background shadows (reused) and the paired old-target ladder."""
    cfg = rt.read(v4 / 'config.json')
    h = rt.sha(v4 / 'config.json')
    if h != (v4 / 'config.sha256').read_text().strip():
        raise ValueError('v4 config checksum changed')
    same = ('full', 'data_dir', 'dataset_fingerprint', 'v3_out', 'v3_config_sha256', 'seeds', 'epochs', 'recipe', 'design',
            'attack', 'attack_seeds', 'classes', 'primary_classes', 'attack_gammas', 'gammas', 'score_seeds',
            'fixed_shadow_seed', 'background_shadows', 'batch', 'weight_decay')
    diff = [k for k in same if cfg.get(k) != c.get(k)]
    key = lambda cells: [(x['name'], x['subset'], x['seed']) for x in cells]
    if key(cfg['cells']) != key(c['cells']):
        diff.append('cells')
    if cfg['schema'] != V4_SCHEMA or cfg['git_commit'] != rt.V4_COMMIT or cfg['background_recipe'] != 'stable' or diff:
        raise ValueError(f'Need the completed v4 stable run at {rt.V4_COMMIT} with the same design; differs in {diff}')
    report = rt.read(v4 / 'report/complete.json')
    if report['status'] != 'complete' or report['config_sha256'] != h:
        raise ValueError('v4 report is not complete')
    pins = [v4 / x for x in ('config.json', 'config.sha256', 'report/complete.json', 'smoke/complete.json', 'smoke/fixed_rows.npz')]
    rt.completed(v4 / 'smoke', h, dict(kind='xsmoke'))
    source = {}
    for sid in cfg['background_shadows']:
        d = v4 / 'background' / f'shadow{sid}'
        rt.completed(d, h, dict(kind='background', shadow=sid))
        pins += [d / f for f in ('final.pt', 'interface.npz', 'result.json', 'complete.json')]
        source[str(sid)] = str(d)
    for s in cfg['seeds']:
        rt.completed(v4 / 'base' / f'seed{s}', h, dict(kind='base', seed=s))
    for cell in cfg['cells']:
        d = v4 / 'truth' / cell['name']
        rt.completed(d, h, cell)
        pins.append(d / 'complete.json')
    for s in cfg['score_seeds']:
        d = v4 / 'score' / f'seed{s}'
        rt.completed(d, h, dict(kind='xscore', seed=s))
        pins.append(d / 'complete.json')
    for p in pins:
        c['input_sha256'][str(p)] = rt.sha(p)
    return h, source


def make_cells(c):
    cells = []
    for s in c['seeds']:  # seed-major: a pack touches one seed, so its J00 is computed once
        for j in range(c['design']['m']):
            cells.append(dict(name=f'truth_s{j:03d}_seed{s}', kind='truth', subset=j, seed=s, minutes=CELL_MINUTES))
    return cells


def make_packs(cells, budget):
    packs = []
    for cell in cells:
        if (not packs or len(packs[-1]) >= 8 or sum(x['minutes'] for x in packs[-1]) + cell['minutes'] > budget
                or packs[-1][-1]['seed'] != cell['seed']):
            packs.append([])
        packs[-1].append(cell)
    return [[x['name'] for x in p] for p in packs]


def prepare(a):
    gammas, gA = floats(a.gammas), floats(a.attack_gammas)
    primary_classes = ints(a.primary_classes)
    if not primary_classes or len(set(primary_classes)) != len(primary_classes) or not set(primary_classes) <= {0, 1, 2, 3}:
        raise ValueError('Primary classes must be distinct OCT class indices')
    if a.primary_gamma not in gammas:
        raise ValueError('Primary gamma must be in the Stage-1 grid')
    for grid in (gammas, gA):
        if len(set(grid)) != len(grid) or any(not math.isfinite(g) or g <= 0 for g in grid):
            raise ValueError('Damping grids must be distinct and positive')
    if not 1 <= a.k_attack <= 10 or not 1 <= a.concurrency <= 32 or not 10 <= a.pack_minutes <= 40:
        raise ValueError('Invalid attack-seed count / concurrency / pack budget')
    c = rt.prepare_base(a, SCHEMA)
    v3 = Path(a.v3_out).resolve()
    rt.disjoint(c['out'], v3)
    cfg, v3hash = check_v3(c, v3)
    leg, n_shadow = legacy_attack(c)
    panel = ints(a.attack_seeds) if a.attack_seeds else [int(x) for x in (leg.get('attack_seeds') or [])]
    if len(panel) < a.k_attack:
        raise ValueError('Legacy config has no crossed attack-seed panel; pass --attack-seeds explicitly')
    score_seeds = ints(a.score_seeds) if a.score_seeds else list(cfg['seeds'])
    if not set(score_seeds) <= set(cfg['seeds']):
        raise ValueError('Score seeds must be v3 training seeds')
    for rel in rt.REUSED:
        c['source_sha256'][str(rt.REPO / rel)] = rt.sha(rt.REPO / rel)
    for d in (rt.LDS_V1, rt.STABLE_V1):
        for p in sorted(d.glob('*.py')):
            c['source_sha256'][str(p)] = rt.sha(p)
    c.update(v3_out=str(v3), v3_config_sha256=v3hash, v3_cells=cfg['cells'],
             seed32_out=cfg['seed32_out'], baseline_out=cfg['baseline_out'], lds_pilot_out=cfg['lds_pilot_out'],
             seeds=list(cfg['seeds']), epochs=cfg['epochs'], recipe=cfg['recipe'], design_name=cfg['design_name'],
             design=cfg['design'], v3_prediction_tol=1e-10, harvest_tol=cfg['harvest_tol'],
             h_chain_fd_rel_tol=5e-3, jvp_fd_max_rel_tol=2e-3,
             finite_difference_policy='diagnostic_only',
             attack=dict(epochs=int(leg['attack_epochs']), lr=float(leg['attack_lr']), batch=int(leg['attack_batch_size']),
                         hidden=64, deterministic=bool(leg.get('deterministic', True))),
             attack_seeds=panel[:a.k_attack], classes=[0, 1, 2, 3], n_shadow=n_shadow,
             primary_classes=primary_classes, primary_gamma=a.primary_gamma,
             background_recipe='stable', fixed_shadow_seed=int(leg.get('fixed_shadow_seed', 42100)),
             background_shadows=[s for s in range(n_shadow) if s != 3],
             target_recipe='stable', target_seed=int(leg['target_seed']), j00_gate=a.j00_gate,
             adjoint_identity_rel_tol=1e-9, attack_gammas=gA, attack_solve_tol=1e-10, j00_tol=a.j00_tol,
             score_seeds=score_seeds, gammas=gammas, lanczos_max_steps=a.lanczos_max, lanczos_est_tol=1e-5,
             probe_steps=a.probe_steps, cg_fail_tol=1e-3, concurrency=a.concurrency, pack_minutes=a.pack_minutes,
             partition=a.partition, fp64_partition=a.fp64_partition, score_time=a.score_time,
             cpu_account=a.cpu_account, cpu_partition=a.cpu_partition,
             primary='Delta_value = J10 - J00 of the image-weighted attack CE over prespecified classes; raw Stage-1 endpoint',
             holdout_policy='Subsets, seeds, attack seeds, classes and primary Stage-1 damping are fixed here; '
                            'gamma_A* is determined from full-baseline Hessians only. Attack quality does not filter classes.',
             gate_policy='CE quality gate (not a calibration test): smoke stops the run unless every primary-class J00 on '
                         'every full baseline is below j00_gate (ln 2 by default). Full baselines only, before any subset outcome.',
             pairing='Every attack is also evaluated on the legacy target (v4 path); after an exact replay check against the '
                     'v4 cells, LDS differences new - old target are bootstrapped over the same subsets.')
    if not 0 < a.j00_gate <= 1:
        raise ValueError('j00_gate must be in (0, 1]')
    c['cells'] = make_cells(c)
    v4 = Path(a.v4_out).resolve()
    rt.disjoint(c['out'], v4)
    c['v4_config_sha256'], c['background_source'] = check_v4(c, v4)
    c['v4_out'] = str(v4)
    c['background_jobs'] = [*c['background_shadows'], 'target']
    c['interp'] = dict(subsets=[0, 1, 2, 3], seeds=c['seeds'][:2], ts=INTERP_TS,
                       rule='fixed before any subset outcome: first four subset ids, first two seeds, all attack seeds, primary classes')
    c['packs'] = make_packs(c['cells'], a.pack_minutes)
    rt.save_config(c)
    print(f"background jobs={c['background_jobs']} (shadows reused from v4); CE gate={c['j00_gate']:.4f}; "
          f"truth+hybrid cells={len(c['cells'])} in {len(c['packs'])} packs (~{len(c['cells']) * CELL_MINUTES / 60:.1f} GPU-h); "
          f"base tasks={len(c['seeds'])}; score tasks={len(score_seeds)}; attack seeds={c['attack_seeds']}; "
          f"partition={a.partition}, score partition={a.fp64_partition}")


def config(out):
    return rt.config(out, SCHEMA)


# ------------------------------------------------------------------ Slurm
def launcher(c, ledger):
    out = Path(c['out'])

    def launch(mode, extra, partition, label=None):
        label = label or mode.replace(':', '_')
        account = c['cpu_account'] if mode == 'collect' else c['account']
        script = 'collect.slurm' if mode == 'collect' else 'job.slurm'
        cmd = ['sbatch', '--parsable', '--account=' + account, '--partition=' + partition,
               '--job-name=oct_x5_' + label, '--output=' + str(out / 'logs/%x-%A_%a.out'),
               '--error=' + str(out / 'logs/%x-%A_%a.err'), *extra, str(rt.HERE / script),
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
    if mode == 'score':
        return getattr(a, 'fp64_partition', None) or c['fp64_partition']
    return c['partition']   # smoke/base/truth share one GPU type so J00 replays exactly


@submission_lock
def submit(a):
    c = config(a.out)
    out = Path(c['out'])
    after = lambda j, kind='afterok': ['--dependency=' + kind + ':' + j, '--kill-on-invalid-dep=yes']
    scheduler.receipts(out)
    with (out / 'submission.jsonl').open('x') as ledger:
        launch = launcher(c, ledger)
        jobs = {}
        jobs['background'] = launch('background', ['--time=' + TIMES['background'],
               array(list(range(len(c['background_jobs']))), c['concurrency'])], c['partition'])
        jobs['smoke'] = launch('smoke', ['--time=' + TIMES['smoke'],
                    *(after(jobs['background']) if 'background' in jobs else [])], c['partition'])
        jobs['base'] = launch('base', ['--time=' + TIMES['base'], array(list(range(len(c['seeds']))), c['concurrency']),
                                       *after(jobs['smoke'])], c['partition'])
        jobs['pack:truth'] = launch('pack:truth', ['--time=' + TIMES['pack'], array(list(range(len(c['packs']))), c['concurrency']),
                                                   *after(jobs['base'])], c['partition'])
        jobs['score'] = launch('score', ['--time=' + walltime(c, 'score', a), array(list(range(len(c['score_seeds']))), c['concurrency']),
                                         *after(jobs['base'])], partition(c, 'score', a))
        jobs['interp'] = launch('interp', ['--time=' + TIMES['interp'], *after(jobs['base'])], c['partition'])
        jobs['collect'] = launch('collect', ['--time=' + TIMES['collect'],
                                *after(jobs['pack:truth'] + ':' + jobs['score'] + ':' + jobs['interp'])], c['cpu_partition'])
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
    def background_done(job):
        if job == 'target':
            return done(out / 'background' / 'target', dict(kind='target'))
        return done(out / 'background' / f'shadow{job}', dict(kind='background', shadow=job))

    def pack_done(names):
        return all(done(out / 'truth' / n, cells[n]) and
                   done(out / 'hybrid' / n, dict(name=n, kind='hybrid', subset=cells[n]['subset'], seed=cells[n]['seed']))
                   for n in names)
    return {'background': [i for i, job in enumerate(c['background_jobs']) if not background_done(job)],
            'base': [i for i, s in enumerate(c['seeds']) if not done(out / 'base' / f'seed{s}', dict(kind='base', seed=s))],
            'pack:truth': [i for i, p in enumerate(c['packs']) if not pack_done(p)],
            'score': [i for i, s in enumerate(c['score_seeds']) if not done(out / 'score' / f'seed{s}', dict(kind='xscore', seed=s))],
            'interp': [] if done(out / 'interp', dict(kind='interp')) else [0]}


def status(a):
    c = config(a.out)
    out = Path(c['out'])
    smoke = (out / 'smoke/complete.json').exists()
    print('smoke:', 'complete' if smoke else 'NOT complete')
    if smoke:
        rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='xsmoke'))
        fr = rt.read(out / 'smoke/frozen.json')
        print('primary classes:', fr['qualified_classes'], 'omega:', fr['omega'], 'gamma_A*:', fr['attack_gamma_star'])
        g = rt.read(out / 'smoke/gate.json')
        print('CE quality gate:', 'PASSED' if g['passed'] else 'FAILED', {k: round(v['J00_max'], 4) for k, v in g['per_class'].items()},
              'threshold', round(g['threshold'], 4), '| JVP identity rel', g['jvp_reverse_identity']['relative_error'],
              '| v4 table replay', g['v4_table_replay'].get('passed'))
    for p in sorted(out.glob('smoke_gate_failed_*.json')):
        g = rt.read(p)
        print('FAILED GATE', p.name, {k: round(v['J00_max'], 4) for k, v in g['per_class'].items()})
    sizes = dict(background=len(c['background_jobs']), base=len(c['seeds']), score=len(c['score_seeds']), interp=1)
    sizes['pack:truth'] = len(c['packs'])
    for k, v in incomplete(c).items():
        print(f'{k}: {sizes[k] - len(v)}/{sizes[k]} complete; incomplete indices {v[:20]}{" ..." if len(v) > 20 else ""}')
    for p in sorted((out / 'score').glob('seed*/result.json')):
        if (p.parent / 'complete.json').exists():
            r = rt.read(p)
            print(p.parent.name, 'qualified gammas:', [g['gamma'] for g in r['gammas'] if g['qualified']],
                  'unqualified:', [g['gamma'] for g in r['gammas'] if not g['qualified']])


@submission_lock
def resubmit(a):
    c = config(a.out)
    out = Path(c['out'])
    previous = scheduler.receipts(out)
    modes = ['background', 'smoke', 'base', 'pack:truth', 'score', 'interp']
    selected = a.only.split(',') if a.only else None
    if selected and any(m not in modes for m in selected):
        raise ValueError('Unknown --only mode')
    bg = incomplete(c)['background']
    if bg:
        scheduler.require_terminal([r for r in previous if r['mode'] in ('background', 'smoke')])
        with (out / 'submission.jsonl').open('a') as ledger:
            launch = launcher(c, ledger)
            j = launch('background', ['--time=' + TIMES['background'], array(bg, c['concurrency'])], c['partition'], 'background_re')
            print('background', j)
            print('smoke', launch('smoke', ['--time=' + TIMES['smoke'], '--dependency=afterok:' + j,
                                           '--kill-on-invalid-dep=yes'], c['partition'], 'smoke_re'))
        print('Resubmit the remaining tasks after smoke completes.')
        return
    if selected == ['background']:
        print('Background shadows already complete')
        return
    if selected == ['smoke'] or not (out / 'smoke/complete.json').exists():
        scheduler.require_terminal([r for r in previous if r['mode'] == 'smoke'])
        if (out / 'smoke/complete.json').exists():
            print('Smoke already complete')
            return
        with (out / 'submission.jsonl').open('a') as ledger:
            print('smoke', launcher(c, ledger)('smoke', ['--time=' + TIMES['smoke']], c['partition'], 'smoke_re'))
        print('Resubmit the rest after smoke completes.')
        return
    rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='xsmoke'))
    todo = incomplete(c)
    selected = selected or [m for m in ('base', 'pack:truth', 'score', 'interp') if todo[m]]
    scheduler.require_terminal([r for r in previous if r['mode'] in selected])
    if any(m in selected for m in ('pack:truth', 'score', 'interp')) and todo['base'] and 'base' not in selected:
        raise ValueError('Base tasks incomplete; resubmit base first (truth and score read base outputs)')
    with (out / 'submission.jsonl').open('a') as ledger:
        launch = launcher(c, ledger)
        base_job = None
        for mode in ('base', 'pack:truth', 'score', 'interp'):
            if mode not in selected or not todo[mode]:
                continue
            deps = ['--dependency=afterok:' + base_job, '--kill-on-invalid-dep=yes'] if base_job and mode != 'base' else []
            arr = [] if mode == 'interp' else [array(todo[mode], c['concurrency'])]
            job = launch(mode, ['--time=' + walltime(c, mode, a), *arr, *deps],
                         partition(c, mode, a), mode.replace(':', '_') + '_re')
            if mode == 'base':
                base_job = job
            print(mode, job, todo[mode])


def queue_report(a):
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
    print(json.dumps({f'{k[0]} @ {k[1]}': dict(tasks=len(v), wait_minutes_median=float(np.median(v)) if v else None,
                                              wait_minutes_max=float(np.max(v)) if v else None)
                      for k, v in by.items()}, indent=2))


# -------------------------------------------------------------------- GPU
def ensure_designs(root, h):
    rt.completed(root / 'smoke', h, dict(kind='xsmoke'))
    src, dst = root / 'smoke/designs.json', root / 'designs.json'
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise ValueError('Frozen designs differ from smoke manifest')
        return
    tmp = root / f'.designs-{uuid.uuid4().hex}.json'
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


def gpu(a):
    from core5 import X5Session as XSession
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}"
    started = time.monotonic()
    mode, _, group = a.mode.partition(':')
    if mode == 'background':
        # Background jobs precede smoke; generate no shared design output here.
        dest = root / f'.background-design-{tag}.json'
        try:
            session = XSession(c, freeze_to=dest)
            job = c['background_jobs'][a.index]
            if job == 'target':
                print('background target', session.background_target(tag), flush=True)
            elif str(job) in (c.get('background_source') or {}):
                print(f'background shadow{job}', session.reuse_background_shadow(job, tag), flush=True)
            else:
                print(f'background shadow{job}', session.background_shadow(job, tag), flush=True)
        finally:
            dest.unlink(missing_ok=True)
        return
    if mode == 'smoke':
        final = root / 'smoke'
        if final.exists():
            rt.completed(final, h, dict(kind='xsmoke'))
            ensure_designs(root, h)
            print('Smoke already complete')
            return
        with rt.lock(root / '.locks' / 'smoke.lock'):
            tmp = root / f'.smoke.tmp-{tag}'
            tmp.mkdir()
            session = XSession(c, freeze_to=tmp / 'designs.json')
            session.xsmoke(tmp)
            config(a.out)
            rt.finish(tmp, c, dict(kind='xsmoke'), elapsed_seconds=time.monotonic() - started)
            os.rename(tmp, final)
        ensure_designs(root, h)
        print('COMPLETE smoke', flush=True)
        return
    ensure_designs(root, h)
    session = XSession(c)
    if mode == 'base':
        s = c['seeds'][a.index]
        print(f'base seed{s}', session.base_seed(s, tag), flush=True)
    elif mode == 'pack' and group == 'truth':
        cells = {x['name']: x for x in c['cells']}
        for n in c['packs'][a.index]:
            t0 = time.monotonic()
            print(n, session.truth_cell(cells[n], tag), f'{time.monotonic() - t0:.0f}s', flush=True)
            t0 = time.monotonic()
            print(n, 'hybrid', session.hybrid_cell(cells[n], tag), f'{time.monotonic() - t0:.0f}s', flush=True)
    elif mode == 'score':
        s = c['score_seeds'][a.index]
        print(f'score seed{s}', session.xscore_seed(s, tag), flush=True)
    elif mode == 'interp':
        print('interp', session.interp_panel(tag), flush=True)
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
    import analysis5 as analysis4
    import figures4
    import report5 as report4
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    ensure_designs(root, h)
    data = analysis4.load(root, c, allow_partial=a.allow_partial)
    summary, tables = analysis4.analyze(data, c)
    summary = clean(summary)
    partial = bool(data['missing'])
    rep = root / ('report' if not partial else 'report_partial_' + time.strftime('%Y%m%dT%H%M%S'))
    rep.mkdir(exist_ok=False)
    rt.write(rep / 'summary.json', summary)
    for key, rows in tables.items():
        if rows:
            rt.csv_write(rep / f'table_{key}.csv', clean(rows))
    figs = figures4.make_all(summary, rep)
    (rep / 'REPORT_CN.md').write_text(report4.render(summary, c, figs, partial))
    rt.write(rep / 'complete.json', dict(status='partial' if partial else 'complete', config_sha256=h, missing=data['missing']))
    archive(root, rep)


def archive(root, rep):
    target = Path(str(root) + '.' + rep.name + '.tar.gz')
    keep = ('.json', '.jsonl', '.csv', '.md', '.txt', '.sha256', '.out', '.err', '.png')
    with tarfile.open(target, 'x:gz') as t:
        for p in sorted(root.rglob('*')):
            rel = p.relative_to(root)
            if not p.is_file() or any(part.startswith('.') for part in rel.parts):
                continue
            if rel.parts[0].startswith('report') and rel.parts[0] != rep.name:
                continue
            # score arrays are small; base v/h arrays (tens of MB per seed) stay on Delta
            if p.suffix in keep or (p.suffix == '.npz' and rel.parts[0] in ('score', 'smoke')):
                t.add(p, arcname=str(Path(root.name) / rel))
    print(str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--out', required=True)
    q.add_argument('--audit', default=rt.AUDIT)
    q.add_argument('--v3-out', default=V3_OUT)
    q.add_argument('--k-attack', type=int, default=3, help='attack seeds taken from the legacy crossed panel')
    q.add_argument('--j00-gate', type=float, default=math.log(2.), help='smoke gate on every primary-class J00')
    q.add_argument('--v4-out', default=V4_OUT, help='completed v4 run: background shadows reused, old-target ladder paired')
    q.add_argument('--primary-classes', default='1,2', help='fixed primary classes; all four classes are also reported')
    q.add_argument('--primary-gamma', type=float, default=.03)
    q.add_argument('--attack-seeds', default='', help='override the legacy panel, comma list')
    q.add_argument('--attack-gammas', default='1,0.3,0.1,0.03,0.01')
    q.add_argument('--gammas', default='1,0.3,0.1,0.03,0.01,0.003', help='Stage-1 damping grid')
    q.add_argument('--score-seeds', default='', help='default: all v3 training seeds')
    q.add_argument('--j00-tol', type=float, default=1e-6)
    q.add_argument('--lanczos-max', type=int, default=300)
    q.add_argument('--probe-steps', type=int, default=24)
    q.add_argument('--partition', default='gpuA100x4', help='smoke, base and truth (one GPU type)')
    q.add_argument('--cpu-account', default='bgjy-delta-cpu')
    q.add_argument('--cpu-partition', default='cpu')
    q.add_argument('--fp64-partition', default='gpuA100x4', help='Stage-1 IF score jobs')
    q.add_argument('--score-time', default=TIMES['score'])
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
        if name == 'resubmit':
            q.add_argument('--only', default=None, help='comma list of base,pack:truth,score,interp (or background/smoke)')
        if name in ('submit', 'resubmit'):
            q.add_argument('--score-time', default=None, help='override score walltime')
            q.add_argument('--fp64-partition', default=None, help='override the score partition')
    a = p.parse_args()
    globals()[a.action.replace('-', '_')](a)


if __name__ == '__main__':
    main()
