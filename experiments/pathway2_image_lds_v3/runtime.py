"""Shared controller contracts, copied from seed32 v2 reviewed (PARENT and module paths adapted)."""
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
from contextlib import contextmanager

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = str(HERE.relative_to(REPO)) + '/'
BASE = 'ca332634a5f50860cea1eb9c1839755232b9434b'
PARENT = '42f88f39bd502255a4599194e00d565f65a35723'  # seed32 v2 reviewed head (the run on Delta)
LDS_V1_HEAD = 'c4880c9ea1a527e4db004b2199b48e21b6426bf7'
STABLE_COMMIT = 'b14b69e7652444204595687032a2f415fe968c44'
LDS_V1 = REPO / 'experiments/pathway2_lds_v1'
STABLE_V1 = REPO / 'experiments/pathway2_stable_v1'
SEED32 = REPO / 'experiments/pathway2_seed32_v2'
SEED32_COMMIT = PARENT
# Pure modules reused byte-for-byte, never edited: LDS v1 design.py / analysis_stats.py and
# seed32 targets.py / solver.py / stats2.py / formula_tests.py. Appended AFTER this package so
# that our own runtime/run modules always win name resolution.
for _p in (LDS_V1, SEED32):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
REUSED = ['experiments/pathway2_lds_v1/design.py', 'experiments/pathway2_lds_v1/analysis_stats.py',
          'experiments/pathway2_stable_v1/stable_core.py', 'experiments/pathway2_seed32_v2/targets.py',
          'experiments/pathway2_seed32_v2/solver.py', 'experiments/pathway2_seed32_v2/stats2.py',
          'experiments/pathway2_seed32_v2/formula_tests.py']
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


@contextmanager
def lock(path):
    """Fail closed on duplicate local/Slurm writers, including retry races."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another writer holds ' + str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', PARENT, 'HEAD'], check=True)
    rows = git('diff', '--name-status', PARENT, 'HEAD').splitlines()
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


def completed(dest, config_hash, expected_cell=None):
    dest = Path(dest)
    m = read(dest / 'complete.json')
    if m['status'] != 'complete' or m['config_sha256'] != config_hash:
        raise ValueError('Wrong/incomplete result: ' + str(dest))
    if not m.get('output_sha256'):
        raise ValueError('Empty output manifest: ' + str(dest))
    if expected_cell is not None and m.get('cell') != expected_cell:
        raise ValueError('Result identity mismatch: ' + str(dest))
    if expected_cell is not None:
        kind = expected_cell.get('kind')
        required = {'full': ['result.json', 'interface.npz'], 'subset': ['result.json', 'interface.npz'],
                    'proxy': ['result.json', 'proxy.npz'], 'score': ['result.json', 'scores.npz'],
                    'smoke': ['result.json', 'designs.json']}.get(kind, [])
        if not set(required) <= set(m['output_sha256']):
            raise ValueError('Required outputs missing from manifest: ' + str(dest))
    if any(Path(rel).is_absolute() or '..' in Path(rel).parts for rel in m['output_sha256']):
        raise ValueError('Output manifest escapes its result directory')
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


def load_context(c):
    """Seed-independent data context (shadow3 split, N=2045, 4343 interface rows).

    The legacy loader validates against the frozen full-truth panel, whose seed
    list is 42/43/44; it is called with seed 42 for VALIDATION ONLY. Each
    training cell re-seeds through pilot.seed_everything(seed) right before
    building its model, exactly like the stable/LDS v1 kernels.
    """
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
    pilot.seed_everything(42, True)
    original_loader = pilot.load_dataset
    captured = {}
    def capture(cfg):
        X, y, groups = original_loader(cfg)
        captured['groups'] = groups
        return X, y, groups
    pilot.load_dataset = capture
    try:
        ctx = legacy.load_context(argparse.Namespace(full_dir=c['full'], dose_dir=None,
            mode='convergence', data_dir=c['data_dir'], seeds=[42]), pilot)
    finally:
        pilot.load_dataset = original_loader
    ctx['groups'] = captured['groups']
    if ctx['groups'] is None:
        raise ValueError('Patient groups are required')
    if ctx['pargs'].affected_shadow != 3 or len(ctx['train']) != 2045:
        raise ValueError('Expected shadow 3 / N=2045')
    if conv.dataset_fingerprint(ctx['X'], ctx['y']) != c['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    split = read(Path(c['full']) / 'splits/fresh_patient_split.json')['shadow_models'][3]
    ctx['test'] = np.asarray(split['test_idx'], dtype=np.int64)
    ctx['interface'] = np.concatenate([ctx['train'], ctx['test']])
    if len(ctx['interface']) != 4343:
        raise ValueError('Expected fixed interface of 4343 rows')
    return ctx, legacy, score, pilot, conv


def finish(dest, c, cell, **extra):
    dest = Path(dest)
    files = {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    write(dest / 'complete.json', dict(status='complete', config_sha256=sha(Path(c['out']) / 'config.json'),
                                      cell=cell, output_sha256=files, **extra))
