"""Shared controller contracts, copied into each independent experiment branch."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = str(HERE.relative_to(REPO)) + '/'
BASE = 'ca332634a5f50860cea1eb9c1839755232b9434b'
AUDIT = '/u/yli103/oct2-pathway2-runs/A/objective_stationarity_pilot_20260926T203618874512340Z'
RUNS = '/u/yli103/oct2-pathway2-runs/A'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        raise ValueError('Empty table: ' + str(path))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def check_repo():
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Experiment worktree must be clean; commit experiment-only edits first')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
    rows = git('diff', '--name-status', BASE, 'HEAD').splitlines()
    if not rows or any(not x.startswith('A\t' + PREFIX) for x in rows):
        raise ValueError('Only additions under ' + PREFIX + ' are allowed')
    return git('rev-parse', 'HEAD')


def disjoint(out, *paths):
    out = Path(out).resolve()
    for raw in paths:
        p = Path(raw).resolve()
        if out == p or p in out.parents or out in p.parents:
            raise ValueError('Output overlaps protected input: ' + str(p))


def verify_hashes(items):
    for p, expected in items.items():
        if sha(p) != expected:
            raise ValueError('Frozen file changed: ' + str(p))


def completed(dest, config_hash):
    dest = Path(dest)
    m = read(dest / 'complete.json')
    if m['status'] != 'complete' or m['config_sha256'] != config_hash:
        raise ValueError('Wrong/incomplete result: ' + str(dest))
    verify_hashes({str(dest / rel): h for rel, h in m['output_sha256'].items()})
    return m


def prepare_base(a, schema):
    commit = check_repo()
    audit, out = Path(a.audit).resolve(), Path(a.out).resolve()
    old = read(audit / 'config.json')
    oldhash = sha(audit / 'config.json')
    if oldhash != (audit / 'config.sha256').read_text().strip():
        raise ValueError('Stationarity config checksum mismatch')
    if old['schema'] != 'OCT_matched_objective_stationarity_v1' or old['base_commit'] != BASE:
        raise ValueError('Expected the validated matched-objective audit')
    if old['patients'] != [807, 2085] or old['weight_decay'] != 1e-5:
        raise ValueError('Frozen patient/regularization contract differs')
    markers = [audit / 'smoke'] + [audit / x['name'] for x in old['cells']]
    for d in markers:
        completed(d, oldhash)
    report = read(audit / 'report/complete.json')
    if report.get('status') != 'complete' or report.get('config_sha256') != oldhash:
        raise ValueError('Run stationarity collect first')
    inputs = dict(old['input_sha256'])
    verify_hashes(inputs)
    # Check shared legacy code against the audited bytes, without requiring the
    # old worktree itself to remain at a particular HEAD.
    for p, h in old['source_sha256'].items():
        rel = Path(p).relative_to(old['repo'])
        if str(rel).startswith(('src/', 'scripts/cross_stage/')):
            if sha(REPO / rel) != h:
                raise ValueError('Legacy implementation differs from audit: ' + str(rel))
    for p in [audit / 'config.json', audit / 'config.sha256', audit / 'report/complete.json']:
        inputs[str(p)] = sha(p)
    for d in markers:
        inputs[str(d / 'complete.json')] = sha(d / 'complete.json')
    disjoint(out, REPO, audit, old['full'], old['data_dir'], old['probe'], old['smallpanel'],
             '/u/yli103/oct2-calibration-v4', '/u/yli103/oct2-exp-a',
             '/u/yli103/oct2-exp-a-smallpanel', '/u/yli103/oct2-exp-a-objective-stationarity')
    if out.exists():
        raise FileExistsError('Use a new output directory: ' + str(out))
    sources = [REPO / x for x in git('ls-files').splitlines()
               if x.startswith((PREFIX, 'src/', 'scripts/cross_stage/'))
               and Path(x).suffix in ('.py', '.sh', '.slurm', '.json', '.md')]
    return dict(schema=schema, repo=str(REPO), git_commit=commit, base_commit=BASE,
                out=str(out), audit=str(audit), audit_config_sha256=oldhash,
                full=old['full'], data_dir=old['data_dir'], account=old['account'],
                patients=old['patients'], batch=64, weight_decay=1e-5,
                dataset_fingerprint=old['dataset_fingerprint'], audited_cells=old['cells'],
                input_sha256=inputs, source_sha256={str(p): sha(p) for p in sources})


def save_config(c):
    out = Path(c['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    write(out / 'config.json', c)
    (out / 'config.sha256').write_text(sha(out / 'config.json') + '\n')
    print('PREPARED ' + str(out), flush=True)


def config(out, schema):
    out = Path(out).resolve()
    if sha(out / 'config.json') != (out / 'config.sha256').read_text().strip():
        raise ValueError('Prepared config changed')
    c = read(out / 'config.json')
    if c['schema'] != schema or c['repo'] != str(REPO) or c['out'] != str(out):
        raise ValueError('Wrong worktree/output/schema')
    if c['git_commit'] != check_repo():
        raise ValueError('Prepared commit changed; keep running attempts frozen')
    verify_hashes({**c['input_sha256'], **c['source_sha256']})
    return c


def load_context(c, seed):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use supplied Slurm job; GPU required for OCT runs')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cross = REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(cross), str(REPO / 'src')]
    spec = importlib.util.spec_from_file_location('isolated_legacy', cross / '11_stage1_diagnostics.py')
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    score = legacy.module_from_path('isolated_score', cross / '07_cross_stage_score_ladder.py')
    conv = legacy.module_from_path('isolated_conv', cross / '20_A_convergence_curve.py')
    pilot = score.import_pilot_module()
    pilot.seed_everything(seed, True)
    ctx = legacy.load_context(argparse.Namespace(full_dir=c['full'], dose_dir=None,
        mode='convergence', data_dir=c['data_dir'], seeds=[seed]), pilot)
    if ctx['pargs'].affected_shadow != 3 or len(ctx['train']) != 2045:
        raise ValueError('Expected shadow 3 / N=2045')
    if conv.dataset_fingerprint(ctx['X'], ctx['y']) != c['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    ctx['patient_rows'] = {p: next(x['raw_indices'] for x in ctx['patients'] if x['patient_id'] == p)
                           for p in c['patients']}
    split = read(Path(c['full']) / 'splits/fresh_patient_split.json')['shadow_models'][3]
    ctx['test'] = np.asarray(split['test_idx'], dtype=np.int64)
    ctx['interface'] = np.concatenate([ctx['train'], ctx['test']])
    if len(ctx['interface']) != 4343:
        raise ValueError('Expected fixed interface of 4343 rows')
    return ctx, legacy, pilot, conv


def finish(dest, c, cell, **extra):
    dest = Path(dest)
    files = {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    write(dest / 'complete.json', dict(status='complete', config_sha256=sha(Path(c['out']) / 'config.json'),
                                      cell=cell, output_sha256=files, **extra))


def submit(a, schema, label):
    c = config(a.out, schema)
    out = Path(c['out'])
    after = getattr(a, 'after', None)
    if after and not after.isdigit():
        raise ValueError('--after must be a Slurm job ID')
    with (out / 'submission.jsonl').open('x') as ledger:
        def launch(mode, extra):
            cmd = ['sbatch', '--parsable', '--account=' + c['account'], '--partition=gpuA40x4',
                   '--job-name=oct_' + label + '_' + mode,
                   '--output=' + str(out / 'logs/%x-%A_%a.out'),
                   '--error=' + str(out / 'logs/%x-%A_%a.err'), *extra,
                   str(HERE / 'job.slurm'), str(HERE), str(out), mode]
            raw = subprocess.check_output(cmd, text=True).strip()
            job = raw.split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Unexpected sbatch reply: ' + raw)
            ledger.write(json.dumps(dict(mode=mode, job_id=job, command=cmd)) + '\n')
            ledger.flush()
            (out / (mode + '_job.txt')).write_text(job + '\n')
            return job
        deps = ['--dependency=afterany:' + after, '--kill-on-invalid-dep=yes'] if after else []
        smoke = launch('smoke', ['--time=00:45:00', *deps])
        array = launch('run', ['--time=' + c['slurm_time'], '--dependency=afterok:' + smoke,
                               '--kill-on-invalid-dep=yes', '--array=0-' + str(len(c['cells'])-1) + '%1'])
    print(f'OUT={out}\nSMOKE={smoke}\nARRAY={array}', flush=True)


def results(c):
    root = Path(c['out'])
    h = sha(root / 'config.json')
    completed(root / 'smoke', h)
    values = []
    for cell in c['cells']:
        d = root / cell['name']
        m = completed(d, h)
        if m['cell'] != cell or m.get('smoke'):
            raise ValueError('Mismatched result cell')
        values.append(read(d / 'result.json'))
    return values


def archive(c):
    out = Path(c['out'])
    target = out.with_name(out.name + '.tar.gz')
    # Checkpoints and full gradient vectors remain on Delta. Interface arrays
    # are small enough to include for independent response checks.
    with tarfile.open(target, 'x:gz') as t:
        for p in sorted(out.rglob('*')):
            if p.is_file() and (p.suffix in ('.json', '.jsonl', '.csv', '.md', '.txt', '.sha256', '.out', '.err')
                                or p.name == 'interface.npz'):
                t.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
    print(str(target), flush=True)


def common_cli(p):
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--audit', default=AUDIT)
    prep.add_argument('--out', required=True)
    for name in ('submit', 'collect', 'gpu'):
        q = sub.add_parser(name)
        q.add_argument('--out', required=True)
        if name == 'submit':
            q.add_argument('--after', help='Wait for another array (afterany); still gate own smoke with afterok')
        if name == 'gpu':
            q.add_argument('--index', type=int, default=0)
            q.add_argument('--smoke', action='store_true')
    return prep
