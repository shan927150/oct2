#!/usr/bin/env python3
"""Cross-stage LDS v6: v5 r1 (5143843) with attack seeds 3 -> 6 and a member-only h variant of L3.

Nothing Stage-1 is trained. The v5 r1 run supplies the stable-target attack table, the 3-seed truth
rows, adjoints V and h; v6 trains the attacks of three new seeds (default 5104-5106), extends the
truth and h to six seeds, computes member-only h for all six, and solves the Stage-1 multi-shift IF
for two right-hand sides (full and member-only hbar). Prespecified primary tests (one-sided,
Bonferroni alpha 0.025 each): P1 L3_full(gamma=0.03) and P2 L3_member(gamma=0.03) against the
6-seed truth; P3 (descriptive) the paired difference L3_member - L3_full.

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

SCHEMA = 'OCT_cross_stage_lds_v6'
V5_SCHEMA = 'OCT_cross_stage_lds_v5r1'
V5_OUT = '/u/yli103/oct2-pathway2-runs/A/cross_stage_lds_v5_20261015T022757824069027Z'
TIMES = dict(smoke='01:00:00', base='01:00:00', pack='00:40:00', score='03:00:00', collect='00:30:00')
PACK_CELLS = 8
COPY = ('full', 'data_dir', 'dataset_fingerprint', 'v3_out', 'v3_config_sha256', 'v3_cells', 'seed32_out', 'baseline_out',
        'lds_pilot_out', 'seeds', 'epochs', 'recipe', 'design_name', 'design', 'v3_prediction_tol', 'harvest_tol',
        'attack', 'classes', 'n_shadow', 'primary_classes', 'primary_gamma', 'attack_gammas', 'attack_solve_tol', 'j00_tol',
        'score_seeds', 'gammas', 'lanczos_max_steps', 'lanczos_est_tol', 'probe_steps', 'cg_fail_tol', 'target_seed',
        'j00_gate', 'adjoint_identity_rel_tol', 'h_chain_fd_rel_tol', 'jvp_fd_max_rel_tol', 'finite_difference_policy',
        'batch', 'weight_decay', 'cells', 'background_shadows', 'fixed_shadow_seed')


def ints(text):
    return [int(x) for x in text.split(',') if x]


def check_v5(v5):
    """The completed v5 r1 run: pin config, smoke table, base arrays, truth cells and scores."""
    cfg = rt.read(v5 / 'config.json')
    h = rt.sha(v5 / 'config.json')
    if h != (v5 / 'config.sha256').read_text().strip():
        raise ValueError('v5 config checksum changed')
    if cfg['schema'] != V5_SCHEMA or cfg['git_commit'] != rt.V5_COMMIT:
        raise ValueError(f'Need the completed v5 r1 run at {rt.V5_COMMIT}')
    report = rt.read(v5 / 'report/complete.json')
    if report['status'] != 'complete' or report['config_sha256'] != h:
        raise ValueError('v5 r1 report is not complete')
    pins = [v5 / x for x in ('config.json', 'config.sha256', 'report/complete.json')]
    m = rt.completed(v5 / 'smoke', h, dict(kind='xsmoke'))
    pins += [v5 / 'smoke' / f for f in ('complete.json', *m['output_sha256'])]
    for s in cfg['seeds']:
        d = v5 / 'base' / f'seed{s}'
        rt.completed(d, h, dict(kind='base', seed=s))
        pins += [d / f for f in ('complete.json', 'v.npz', 'h.npz', 'result.json')]
    for cell in cfg['cells']:
        d = v5 / 'truth' / cell['name']
        rt.completed(d, h, cell)
        pins += [d / 'complete.json', d / 'result.json']
    for s in cfg['score_seeds']:
        d = v5 / 'score' / f'seed{s}'
        rt.completed(d, h, dict(kind='xscore', seed=s))
        pins += [d / 'complete.json', d / 'scores.npz']
    return cfg, h, {str(p): rt.sha(p) for p in pins}


def make_packs(cells):
    packs = []
    for cell in cells:
        if not packs or len(packs[-1]) >= PACK_CELLS or packs[-1][-1]['seed'] != cell['seed']:
            packs.append([])
        packs[-1].append(cell)
    return [[x['name'] for x in p] for p in packs]


def prepare(a):
    if not 1 <= a.concurrency <= 32:
        raise ValueError('Invalid concurrency')
    c = rt.prepare_base(a, SCHEMA)
    v5 = Path(a.v5_out).resolve()
    rt.disjoint(c['out'], v5)
    cfg, h5, pins = check_v5(v5)
    for k in COPY:
        c[k] = cfg[k]
    for key in ('full', 'data_dir', 'dataset_fingerprint'):
        if c[key] != cfg[key]:
            raise ValueError(f'{key} differs from the v5 r1 run')
    c['input_sha256'].update(cfg['input_sha256'])
    c['input_sha256'].update(pins)
    rt.verify_hashes(c['input_sha256'])
    for rel in rt.REUSED:
        c['source_sha256'][str(rt.REPO / rel)] = rt.sha(rt.REPO / rel)
    for d in (rt.LDS_V1, rt.STABLE_V1):
        for p in sorted(d.glob('*.py')):
            c['source_sha256'][str(p)] = rt.sha(p)
    leg = rt.read(Path(c['full']) / 'experiment_config.json')['args']
    panel = [int(x) for x in (leg.get('attack_seeds') or [])]
    old = list(cfg['attack_seeds'])
    if panel[:len(old)] != old:
        raise ValueError('v5 r1 attack seeds are not the head of the legacy crossed panel')
    if len(panel) >= len(old) + a.n_new:
        new, source = panel[len(old):len(old) + a.n_new], 'legacy crossed panel'
        if a.new_attack_seeds and ints(a.new_attack_seeds) != new:
            raise ValueError(f'--new-attack-seeds {a.new_attack_seeds} disagrees with the legacy panel continuation {new}')
    else:
        new, source = ints(a.new_attack_seeds), 'explicit (legacy panel too short)'
        if len(new) != a.n_new:
            raise ValueError(f'Legacy panel has {len(panel)} seeds; pass --new-attack-seeds with {a.n_new} seeds')
    if len(set(old + new)) != len(old + new):
        raise ValueError('New attack seeds must differ from the old ones')
    c.update(v5_out=str(v5), v5_config_sha256=h5, old_attack_seeds=old, new_attack_seeds=new, attack_seeds=old + new,
             new_attack_seed_source=source, concurrency=a.concurrency, partition=a.partition, fp64_partition=a.fp64_partition,
             score_time=a.score_time, cpu_account=a.cpu_account, cpu_partition=a.cpu_partition,
             primary_tests=dict(
                 P1='L3_full(gamma=0.03) vs the 6-attack-seed truth; one-sided permutation p, alpha 0.025 (Bonferroni over P1, P2)',
                 P2='L3_member(gamma=0.03) vs the 6-attack-seed truth; one-sided permutation p, alpha 0.025',
                 P3='paired subset bootstrap of LDS(L3_member) - LDS(L3_full) at gamma=0.03, same truth (descriptive, 95% CI)',
                 status='P2 was suggested by the v5 r1 L1 decomposition on the same 32 subsets and 8 Stage-1 seeds; it is '
                        'prespecified before any L3_member is computed, but not independent of those subsets. Confirmation '
                        'needs fresh subsets.'),
             secondary='Same rungs on the new-seed-only truth (held-out attack randomness) and on the old-seed truth; '
                       'L1 / L2-lin full vs member; attack-seed reliability; LDS(L3_full, 6-seed hbar) vs v5 r1 L3 (3-seed hbar).')
    c['packs'] = make_packs(c['cells'])
    rt.save_config(c)
    print(f"v5 r1 run {v5}; attack seeds {old} + new {new} ({source}); truth cells={len(c['cells'])} in {len(c['packs'])} packs; "
          f"score tasks={len(c['score_seeds'])} (two right-hand sides each); partition={a.partition}")


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
               '--job-name=oct_x6_' + label, '--output=' + str(out / 'logs/%x-%A_%a.out'),
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


def walltime(c, mode, a):
    if mode == 'score':
        return getattr(a, 'score_time', None) or c['score_time']
    return TIMES['pack' if mode.startswith('pack') else mode]


def partition(c, mode, a):
    if mode == 'score':
        return getattr(a, 'fp64_partition', None) or c['fp64_partition']
    return c['partition']   # smoke/base/truth must match the v5 r1 GPU type so attacks replay exactly


@submission_lock
def submit(a):
    c = config(a.out)
    out = Path(c['out'])
    after = lambda j: ['--dependency=afterok:' + j, '--kill-on-invalid-dep=yes']
    scheduler.receipts(out)
    with (out / 'submission.jsonl').open('x') as ledger:
        launch = launcher(c, ledger)
        jobs = dict(smoke=launch('smoke', ['--time=' + TIMES['smoke']], c['partition']))
        jobs['base'] = launch('base', ['--time=' + TIMES['base'], array(list(range(len(c['seeds']))), c['concurrency']),
                                       *after(jobs['smoke'])], c['partition'])
        jobs['pack:truth'] = launch('pack:truth', ['--time=' + TIMES['pack'], array(list(range(len(c['packs']))), c['concurrency']),
                                                   *after(jobs['base'])], c['partition'])
        jobs['score'] = launch('score', ['--time=' + walltime(c, 'score', a), array(list(range(len(c['score_seeds']))), c['concurrency']),
                                         *after(jobs['base'])], partition(c, 'score', a))
        jobs['collect'] = launch('collect', ['--time=' + TIMES['collect'], *after(jobs['pack:truth'] + ':' + jobs['score'])],
                                 c['cpu_partition'])
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
    return {'base': [i for i, s in enumerate(c['seeds']) if not done(out / 'base' / f'seed{s}', dict(kind='base6', seed=s))],
            'pack:truth': [i for i, p in enumerate(c['packs'])
                           if not all(done(out / 'truth' / n, dict(cells[n], kind='truth6')) for n in p)],
            'score': [i for i, s in enumerate(c['score_seeds']) if not done(out / 'score' / f'seed{s}', dict(kind='score6', seed=s))]}


def status(a):
    c = config(a.out)
    out = Path(c['out'])
    smoke = (out / 'smoke/complete.json').exists()
    print('smoke:', 'complete' if smoke else 'NOT complete')
    if smoke:
        rt.completed(out / 'smoke', rt.sha(out / 'config.json'), dict(kind='xsmoke'))
        g = rt.read(out / 'smoke/gate.json')
        print('new seeds', c['new_attack_seeds'], 'CE quality gate:', 'PASSED' if g['passed'] else 'FAILED',
              {k: round(v['J00_max'], 4) for k, v in g['per_class'].items()},
              '| gamma_A* qualified:', g['gamma_star_qualified_on_new_seeds'], '| old attack replay:', g['old_attack_replay_max_abs'])
    for p in sorted(out.glob('smoke_gate_failed_*.json')):
        print('FAILED GATE', p.name)
    sizes = dict(base=len(c['seeds']), score=len(c['score_seeds']), **{'pack:truth': len(c['packs'])})
    for k, v in incomplete(c).items():
        print(f'{k}: {sizes[k] - len(v)}/{sizes[k]} complete; incomplete indices {v[:20]}{" ..." if len(v) > 20 else ""}')


@submission_lock
def resubmit(a):
    c = config(a.out)
    out = Path(c['out'])
    previous = scheduler.receipts(out)
    modes = ['smoke', 'base', 'pack:truth', 'score']
    selected = a.only.split(',') if a.only else None
    if selected and any(m not in modes for m in selected):
        raise ValueError('Unknown --only mode')
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
    selected = selected or [m for m in ('base', 'pack:truth', 'score') if todo[m]]
    scheduler.require_terminal([r for r in previous if r['mode'] in selected])
    if any(m in selected for m in ('pack:truth', 'score')) and todo['base'] and 'base' not in selected:
        raise ValueError('Base tasks incomplete; resubmit base first (truth and score read base outputs)')
    with (out / 'submission.jsonl').open('a') as ledger:
        launch = launcher(c, ledger)
        base_job = None
        for mode in ('base', 'pack:truth', 'score'):
            if mode not in selected or not todo[mode]:
                continue
            deps = ['--dependency=afterok:' + base_job, '--kill-on-invalid-dep=yes'] if base_job and mode != 'base' else []
            job = launch(mode, ['--time=' + walltime(c, mode, a), array(todo[mode], c['concurrency']), *deps],
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
    print(json.dumps({f'{k[0]} @ {k[1]}': dict(tasks=len(v), wait_minutes_median=float(np.median(v)) if v else None)
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
    from core6 import X6Session
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}"
    started = time.monotonic()
    mode, _, group = a.mode.partition(':')
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
            session = X6Session(c, freeze_to=tmp / 'designs.json')
            session.smoke6(tmp)
            config(a.out)
            rt.finish(tmp, c, dict(kind='xsmoke'), elapsed_seconds=time.monotonic() - started)
            os.rename(tmp, final)
        ensure_designs(root, h)
        print('COMPLETE smoke', flush=True)
        return
    ensure_designs(root, h)
    session = X6Session(c)
    if mode == 'base':
        s = c['seeds'][a.index]
        print(f'base seed{s}', session.base6(s, tag), flush=True)
    elif mode == 'pack' and group == 'truth':
        cells = {x['name']: x for x in c['cells']}
        for n in c['packs'][a.index]:
            t0 = time.monotonic()
            print(n, session.truth6(cells[n], tag), f'{time.monotonic() - t0:.0f}s', flush=True)
    elif mode == 'score':
        s = c['score_seeds'][a.index]
        print(f'score seed{s}', session.score6(s, tag), flush=True)
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
    import analysis6
    import report6
    c = config(a.out)
    root = Path(c['out'])
    h = rt.sha(root / 'config.json')
    ensure_designs(root, h)
    data = analysis6.load(root, c, allow_partial=a.allow_partial)
    summary, tables = analysis6.analyze(data, c)
    summary = clean(summary)
    partial = bool(data['missing'])
    rep = root / ('report' if not partial else 'report_partial_' + time.strftime('%Y%m%dT%H%M%S'))
    rep.mkdir(exist_ok=False)
    rt.write(rep / 'summary.json', summary)
    for key, rows in tables.items():
        if rows:
            rt.csv_write(rep / f'table_{key}.csv', clean(rows))
    (rep / 'REPORT_CN.md').write_text(report6.render(summary, c, partial))
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
            if p.suffix in keep or (p.suffix == '.npz' and rel.parts[0] == 'score'):
                t.add(p, arcname=str(Path(root.name) / rel))
    print(str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--out', required=True)
    q.add_argument('--audit', default=rt.AUDIT)
    q.add_argument('--v5-out', default=V5_OUT, help='completed v5 r1 run (5143843)')
    q.add_argument('--n-new', type=int, default=3, help='number of new attack seeds')
    q.add_argument('--new-attack-seeds', default='5104,5105,5106',
                   help='used only if the legacy crossed panel is shorter than old + n-new; must agree with it otherwise')
    q.add_argument('--partition', default='gpuA100x4', help='smoke, base and truth (same GPU type as the v5 r1 run)')
    q.add_argument('--fp64-partition', default='gpuA100x4', help='Stage-1 IF score jobs')
    q.add_argument('--score-time', default=TIMES['score'])
    q.add_argument('--cpu-account', default='bgjy-delta-cpu')
    q.add_argument('--cpu-partition', default='cpu')
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
            q.add_argument('--only', default=None, help='comma list of base,pack:truth,score (or smoke)')
        if name in ('submit', 'resubmit'):
            q.add_argument('--score-time', default=None, help='override score walltime')
            q.add_argument('--fp64-partition', default=None, help='override the score partition')
    a = p.parse_args()
    globals()[a.action.replace('-', '_')](a)


if __name__ == '__main__':
    main()
