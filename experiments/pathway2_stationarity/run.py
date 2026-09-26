#!/usr/bin/env python3
"""Isolated read-only matched-objective stationarity audit; Slurm controller.

prepare/submit/collect use the standard library. GPU work is imported lazily.
No optimizer, checkpoint write, legacy-script modification or branch checkout.
"""
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
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = 'ca332634a5f50860cea1eb9c1839755232b9434b'
PREFIX = 'experiments/pathway2_stationarity/'
SCHEMA = 'OCT_matched_objective_stationarity_v1'
PATIENTS = [807, 2085]
SEEDS = [42, 44]
WD = 1e-5
DEFAULT_SP = '/u/yli103/oct2-pathway2-runs/A/smallpanel_loo_R100_X40_20260924T035311975763Z'
DEFAULT_AUDIT = '/u/yli103/oct2-pathway2-runs/A/frozen_loss_audit_20260926T195320900000375Z'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty result table')
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def check_repo():
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('The isolated checkout must be clean before preparation/execution')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
    changed = git('diff', '--name-only', BASE, 'HEAD').splitlines()
    if not changed or any(not p.startswith(PREFIX) for p in changed):
        raise ValueError('Expected additive experiment-only branch based on ' + BASE)
    return git('rev-parse', 'HEAD')


def disjoint(out, *inputs):
    out = Path(out).resolve()
    for path in inputs:
        p = Path(path).resolve()
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError('Output overlaps protected input: ' + str(p))


def cells_for(profile):
    coords = [('decay', 140)] if profile == 'pilot' else [
        (a, e) for a in ('control', 'decay') for e in (120, 140)]
    return [dict(name=f'seed{s}_{a}_E{e}', seed=s, arm=a, epoch=e)
            for s in SEEDS for a, e in coords]


def prepare(a):
    commit = check_repo()
    sp, audit, out = a.smallpanel.resolve(), a.loss_audit.resolve(), a.out.resolve()
    spcfg = read(sp / 'smallpanel_config.json')
    probe, full, data = (Path(spcfg[k]).resolve() for k in ('probe_root', 'full_dir', 'data_dir'))
    disjoint(out, REPO, sp, audit, probe, full, data,
             '/u/yli103/oct2-exp-a', '/u/yli103/oct2-exp-a-smallpanel', '/u/yli103/oct2-calibration-v4')
    if sha(sp / 'smallpanel_config.json') != (sp / 'config.sha256').read_text().strip():
        raise ValueError('Small-panel configuration hash mismatch')
    if spcfg['git_commit'] != BASE:
        raise ValueError('Expected the inspected B/C experiment version')
    old = read(audit / 'config.json')
    old_sha = sha(audit / 'config.json')
    if old_sha != (audit / 'config.sha256').read_text().strip():
        raise ValueError('Loss audit configuration hash mismatch')
    for k, expected in dict(probe=str(probe), full=str(full), data_dir=str(data)).items():
        if old[k] != expected:
            raise ValueError('Current loss audit uses another ' + k)
    paths = [sp / 'smallpanel_config.json', sp / 'config.sha256',
             audit / 'config.json', audit / 'config.sha256']
    for seed in SEEDS:
        marker = audit / f'seed{seed}/complete.json'
        m = read(marker)
        if (m['status'] != 'complete' or m['config_sha256'] != old_sha or
                not m['model_states_unchanged'] or not m['input_files_unchanged']):
            raise ValueError('Prior loss audit integrity failure')
        paths += [marker, audit / f'seed{seed}/checkpoint_summary.csv',
                  audit / f'seed{seed}/paired_differences.csv']
    paths += [full / rel for rel in ('experiment_config.json', 'experiment_summary.json',
                                    'selected_patients.json', 'splits/fresh_patient_split.json')]
    panel = {p['patient_id']: p for p in read(full / 'selected_patients.json')['patients']}
    if [(p, panel[p]['oct_class'], len(panel[p]['raw_indices'])) for p in PATIENTS] != [
            (807, 1, 6), (2085, 2, 5)]:
        raise ValueError('Frozen patient panel differs')
    cells = cells_for(a.profile)
    fingerprints = []
    for cell in cells:
        s, arm, epoch = cell['seed'], cell['arm'], cell['epoch']
        d = probe / f'shadow3_seed{s}_{arm}'
        m = read(d / 'manifest.json')
        expected = dict(schema='pathway2_A_lr_continuation_v1', status='complete', seed=s,
                        arm=arm, resume_epoch=100, total_epochs=140, base_lr=.001,
                        weight_decay=WD, decay_schedule='20@1e-4,20@1e-5' if arm == 'decay' else None)
        for k, v in expected.items():
            if m.get(k) != v:
                raise ValueError(f'{d}: {k} differs from {v!r}')
        if m['args']['checkpoint_epochs'] != [120, 140] or not m['resume_check']['passed']:
            raise ValueError('Wrong checkpoint horizon or failed resume')
        for k in ('input_files_unchanged', 'source_files_unchanged', 'dataset_arrays_unchanged'):
            if m.get(k) is not True:
                raise ValueError('Probe integrity failure: ' + k)
        for rel, digest in m['source_sha256'].items():
            if sha(REPO / rel) != digest:
                raise ValueError('Legacy source differs from original training: ' + rel)
        cell['checkpoint'] = str(d / f'checkpoints/epoch{epoch:03d}.pt')
        checkpoint_hash = sha(cell['checkpoint'])
        if old['input_sha256'].get(cell['checkpoint']) != checkpoint_hash:
            raise ValueError('Checkpoint differs from the completed loss audit')
        cell['checkpoint_sha256'] = checkpoint_hash
        cell['epoch_order_sha256'] = m['epoch_order_sha256']
        ref = [r for r in read_csv(audit / f'seed{s}/checkpoint_summary.csv')
               if r['arm'] == arm and int(r['epoch']) == epoch]
        if len(ref) != 1:
            raise ValueError('Missing/duplicate loss audit cell')
        cell['reference_eval_ce'] = float(ref[0]['eval_ce'])
        paths += [Path(cell['checkpoint'])] + [d / rel for rel in
                    ('manifest.json', 'dataset_fingerprint.json', 'curve.csv', 'input_sha256.json')]
        # The original full-data config/split must match the probe's inputs.
        old_inputs = read(d / 'input_sha256.json')
        for rel in ('experiment_config.json', 'splits/fresh_patient_split.json'):
            if old_inputs.get(str(full / rel)) != sha(full / rel):
                raise ValueError('Probe belongs to another full-data experiment')
        fingerprints.append(read(d / 'dataset_fingerprint.json'))
    if any(x != fingerprints[0] for x in fingerprints) or fingerprints[0] != spcfg['dataset_fingerprint']:
        raise ValueError('Dataset fingerprints disagree')
    # Freeze actual input/code bytes, not mutable job ledgers in old branches.
    source = [REPO / rel for rel in git('ls-files').splitlines()
              if rel.startswith(PREFIX) or Path(rel).suffix in ('.py', '.sh', '.slurm', '.json')]
    cfg = dict(schema=SCHEMA, repo=str(REPO), git_commit=commit, base_commit=BASE,
        out=str(out), profile=a.profile, full=str(full), probe=str(probe), data_dir=str(data),
        smallpanel=str(sp), loss_audit=str(audit), account=spcfg['account'],
        cells=cells, patients=PATIENTS, alpha=[.1, 1.], mc_reps=a.reps, batch=64,
        weight_decay=WD, dataset_fingerprint=fingerprints[0],
        input_sha256={str(p): sha(p) for p in sorted(set(paths))},
        source_sha256={str(p): sha(p) for p in source},
        design='Read-only gradients for eval and dropout-expected CE+L2; fixed original N; shared-mask patient gradients.')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    write(out / 'config.json', cfg)
    (out / 'config.sha256').write_text(sha(out / 'config.json') + '\n')
    print('PREPARED ' + str(out), flush=True)


def load_config(out):
    out = Path(out).resolve()
    if sha(out / 'config.json') != (out / 'config.sha256').read_text().strip():
        raise ValueError('Prepared configuration changed')
    c = read(out / 'config.json')
    if c['schema'] != SCHEMA or c['out'] != str(out) or c['repo'] != str(REPO):
        raise ValueError('Wrong prepared directory/repository')
    if check_repo() != c['git_commit']:
        raise ValueError('Prepared commit changed')
    for p, digest in {**c['input_sha256'], **c['source_sha256']}.items():
        if sha(p) != digest:
            raise ValueError('Frozen input/source changed: ' + p)
    return c


def completed(dest, config_hash):
    m = read(dest / 'complete.json')
    if m['status'] != 'complete' or m['config_sha256'] != config_hash:
        raise ValueError('Missing complete result for this configuration')
    for rel, digest in m['output_sha256'].items():
        if sha(dest / rel) != digest:
            raise ValueError('Result changed: ' + str(dest / rel))
    return m


def gpu(a):
    c = load_config(a.out)
    out, config_hash = Path(c['out']), sha(Path(c['out']) / 'config.json')
    if not a.smoke:
        completed(out / 'smoke', config_hash)
    if not 0 <= a.index < len(c['cells']):
        raise ValueError('Array index out of bounds')
    cell = c['cells'][a.index]
    cross = REPO / 'scripts/cross_stage'
    sys.path[:0] = [str(HERE), str(cross), str(REPO / 'src')]
    import numpy as np
    import torch
    import stage1_diagnostic_core as core
    from audit_torch import snapshot
    from audit_math import VectorMoments, summary_rows, norm, cosine, ratio
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use the supplied Slurm script')
    spec = importlib.util.spec_from_file_location('stationarity_legacy', cross / '11_stage1_diagnostics.py')
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    score = legacy.module_from_path('stationarity_score', cross / '07_cross_stage_score_ladder.py')
    conv = legacy.module_from_path('stationarity_conv', cross / '20_A_convergence_curve.py')
    pilot = score.import_pilot_module()
    ns = argparse.Namespace(full_dir=c['full'], dose_dir=None, mode='convergence',
                            data_dir=c['data_dir'], seeds=[cell['seed']])
    context = legacy.load_context(ns, pilot)
    X, y, train = context['X'], context['y'], context['train']
    if conv.dataset_fingerprint(X, y) != c['dataset_fingerprint'] or len(train) != 2045:
        raise ValueError('Frozen dataset or training N differs')
    rows = {p: next(x['raw_indices'] for x in context['patients'] if x['patient_id'] == p)
            for p in c['patients']}
    payload = legacy.load_payload(cell['checkpoint'], pilot)
    md = payload['metadata']
    expected = dict(seed=cell['seed'], arm=cell['arm'], epoch=cell['epoch'],
        source='pathway2_A_lr_continuation_v1', resume_epoch=100, total_epochs=140,
        epoch_order_sha256=cell['epoch_order_sha256'], base_lr=.001,
        decay_schedule='20@1e-4,20@1e-5' if cell['arm'] == 'decay' else None)
    for k, v in expected.items():
        if md.get(k) != v:
            raise ValueError('Checkpoint metadata differs: ' + k)
    pilot.seed_everything(26092600 + cell['seed'], True)
    model = legacy.model_from_payload(payload, pilot, X, y).to(pilot.DEVICE)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    modes_before = [m.training for m in model.modules()]
    rng_before = core.rng_fingerprint()
    grad_before = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    dest = out / ('smoke' if a.smoke else cell['name'])
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    kwargs = dict(model=model, X=X, y=y, indices=train, patient_rows=rows,
                  batch=c['batch'], wd=c['weight_decay'])
    ev, evectors = snapshot(**kwargs, check_identity=a.smoke)
    again, evectors2 = snapshot(**kwargs)
    if not np.array_equal(evectors, evectors2) or ev['ce'] != again['ce']:
        raise RuntimeError('Repeated deterministic eval changed')
    reference_ce_error = abs(ev['ce'] - cell['reference_eval_ce'])
    if reference_ce_error > 1e-8 + 5e-5 * abs(cell['reference_eval_ce']):
        raise RuntimeError('Eval CE does not reproduce the completed frozen-loss audit')
    with core.preserve_rng_and_modes(model):
        legacy_stats, legacy_g = core.objective_snapshot(model, X, y, train, c['batch'], WD, True)
    legacy_g = legacy_g.detach().double().cpu().numpy()
    gradient_error = norm(evectors[0] - legacy_g)
    if gradient_error > 1e-8 + 5e-5 * norm(legacy_g):
        raise RuntimeError('Full gradient differs from legacy objective_snapshot')
    # This seed stream is fixed across arms/epochs; it is independent of training.
    reps = 2 if a.smoke else c['mc_reps']
    moments, draws, prefixes = VectorMoments(), [], []
    identities = list(ev['identity_checks'])
    for rep in range(reps):
        draw_seed = 26092600 + cell['seed'] * 10000 + rep
        values, vectors = snapshot(**kwargs, dropout=True, seed=draw_seed,
                                   check_identity=a.smoke and rep == 0)
        identities += values['identity_checks']
        moments.update(vectors)
        draw = dict(rep=rep, draw_seed=draw_seed, ce=values['ce'], objective=values['objective'],
                    g_norm=norm(vectors[0]))
        for j, p in enumerate(c['patients'], 1):
            draw[f'b_{p}_norm'] = norm(vectors[j])
            draw[f'patient_{p}_ce_contribution'] = values['patient_ce_contribution'][p]
        draws.append(draw)
        if rep + 1 in {16, 32, 64, 128, 256, reps}:
            prefixes += [dict(reps=rep+1, **r) for r in summary_rows(moments.mean, c['patients'], moments)]
            print(f"{cell['name']} MC {rep+1}/{reps}: ||mean g||={norm(moments.mean[0]):.6g}", flush=True)
    summary = []
    for objective, vectors, estimator in [('eval', evectors, None), ('dropout_expected', moments.mean, moments)]:
        summary += [dict(seed=cell['seed'], arm=cell['arm'], epoch=cell['epoch'],
                    objective=objective, n_train=len(train), mc_reps=reps if estimator else 0,
                    n_patient=len(rows[r['patient_id']]), **r)
                    for r in summary_rows(vectors, c['patients'], estimator)]
    mismatch = []
    for j, name in enumerate(['g'] + [f'b_{p}' for p in c['patients']]):
        se = moments.se(np.eye(len(moments.mean))[j])
        diff = norm(moments.mean[j] - evectors[j])
        mismatch.append(dict(vector=name, difference_norm=diff,
                             cosine_dropout_eval=cosine(moments.mean[j], evectors[j]),
                             dropout_vector_se=se, difference_over_se=ratio(diff, se)))
    if not core.exact_tree(before, {k: v.detach().cpu() for k, v in model.state_dict().items()}):
        raise RuntimeError('Model state changed')
    if [m.training for m in model.modules()] != modes_before or core.rng_fingerprint() != rng_before:
        raise RuntimeError('Model modes or RNG changed')
    if not core.exact_tree(grad_before, [p.grad for p in model.parameters()]):
        raise RuntimeError('Accumulated parameter gradients changed')
    if conv.dataset_fingerprint(X, y) != c['dataset_fingerprint']:
        raise RuntimeError('Dataset arrays changed')
    load_config(out)
    write_csv(dest / 'summary.csv', summary)
    write_csv(dest / 'draws.csv', draws)
    write_csv(dest / 'mc_prefixes.csv', prefixes)
    write_csv(dest / 'objective_mismatch.csv', mismatch)
    write(dest / 'checks.json', dict(repeated_eval_exact=True, reference_eval_ce_error=reference_ce_error,
        legacy_gradient_difference_norm=gradient_error, legacy_eval_ce=legacy_stats['ce'],
        identity_checks=identities, model_rng_modes_and_grad_unchanged=True,
        dataset_arrays_unchanged=True, input_and_code_files_unchanged=True))
    np.savez_compressed(dest / 'gradient_vectors.npz', eval_vectors=evectors,
        dropout_mean=moments.mean, mc_trace_m2=moments.m2, mc_reps=moments.n,
        patient_ids=np.asarray(c['patients']),
        parameter_names=np.asarray([name for name, _ in model.named_parameters()]),
        parameter_numel=np.asarray([p.numel() for p in model.parameters()]))
    write(dest / 'complete.json', dict(status='complete', config_sha256=config_hash,
        cell=cell, n_train=len(train), mc_reps=reps, smoke=a.smoke,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        elapsed_seconds=time.monotonic()-started,
        output_sha256={p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()}))
    print('COMPLETE ' + str(dest), flush=True)


def submit(a):
    c = load_config(a.out)
    out = Path(c['out'])
    # An exclusive record prevents accidental duplicate submissions. A failed
    # submission preserves any returned job ID for manual recovery.
    with (out / 'submission.jsonl').open('x') as ledger:
        def sbatch(mode, extra):
            cmd = ['sbatch', '--parsable', '--account=' + c['account'],
                '--partition=gpuA40x4', '--job-name=oct_stat_' + mode,
                '--output=' + str(out / 'logs/%x-%A_%a.out'),
                '--error=' + str(out / 'logs/%x-%A_%a.err'), *extra,
                str(HERE / 'audit.slurm'), str(HERE), str(out), mode]
            result = subprocess.check_output(cmd, text=True).strip()
            job = result.split(';', 1)[0]
            if not job.isdigit():
                raise RuntimeError('Unexpected sbatch response: ' + result)
            ledger.write(json.dumps(dict(mode=mode, job_id=job, command=cmd)) + '\n')
            ledger.flush()
            (out / f'{mode}_job.txt').write_text(job + '\n')
            return job
        smoke = sbatch('smoke', ['--time=00:30:00'])
        array = sbatch('run', ['--dependency=afterok:' + smoke, '--kill-on-invalid-dep=yes',
                              '--array=0-' + str(len(c['cells'])-1) + '%1'])
    print(f'OUT={out}\nSMOKE={smoke}\nARRAY={array}', flush=True)


def collect(a):
    c = load_config(a.out)
    out, config_hash = Path(c['out']), sha(Path(c['out']) / 'config.json')
    completed(out / 'smoke', config_hash)
    all_rows = []
    for cell in c['cells']:
        dest = out / cell['name']
        marker = completed(dest, config_hash)
        if marker['cell'] != cell or marker['mc_reps'] != c['mc_reps'] or marker['smoke']:
            raise ValueError('Incomplete/wrong result cell')
        rows = read_csv(dest / 'summary.csv')
        if len(rows) != 8:
            raise ValueError('Expected 2 objectives x 2 patients x 2 doses')
        all_rows += rows
    archive = out.with_name(out.name + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Archive already exists: ' + str(archive))
    report = out / 'report'
    report.mkdir(exist_ok=False)
    write_csv(report / 'stationarity_summary.csv', all_rows)
    table = ['| seed | arm | E | objective | patient | alpha | ||g||/(alpha||b||) | SE(g)/||g|| | SE(b)/||b|| |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in all_rows:
        fmt = lambda k: f"{float(r[k]):.4g}" if r[k] else 'undefined'
        table.append('| ' + ' | '.join([r['seed'], r['arm'], r['epoch'], r['objective'],
                     r['patient_id'], r['alpha'], fmt('residual_over_perturbation'),
                     fmt('g_se_over_norm'), fmt('b_se_over_norm')]) + ' |')
    text = '''# Matched-objective stationarity audit

PDF Eq. (52)-(56) assume a stationary baseline for the same loss whose Hessian
and patient gradient are used. Here g = grad(mean CE + lambda||theta||²/2),
b_p = sum(patient CE gradients)/N, and grad(F-alpha L_p) = g-alpha b_p.
N stays 2045. L2 occurs only in g. There is no Hessian solve or retraining.

Eval and dropout-expected objectives are reported separately. All MC vectors
within a draw share the same masks; SE of g-alpha b includes their covariance.
SE is an estimated RMS vector error, not a confidence radius or p-value.
Large relative SE makes norm ratios unreliable; inspect mc_prefixes.csv.
The signed norm²-minus-variance estimate can be negative when noise dominates.
Small raw gradient ratios alone do not prove stationarity, invertibility,
locality, or L3 validity. Hessian conditioning still matters. These are two
selected seeds/two patients, not a 40-cell population conclusion.

''' + '\n'.join(table) + '''

## What to do with these results

- If b or the dropout mean is unresolved relative to MC error, increase MC
  precision in a new attempt before interpreting its ratio; do not infer zero.
- A substantial residual relative to the deletion perturbation motivates an
  isolated, objective-matched polish experiment followed by a new audit.
- With an adequately resolved small residual, a later local fixed-mask dose
  sweep must match g, b and H objectives. Full cold-start Adam LOO remains a
  different estimand; PDF Eq. (70)-(74) needs derivatives through Adam state.
- Do not combine dropout gradients here with the existing eval-mode Hessian.
  No threshold automatically authorizes a polish, damping sweep or L3 run.

gradient_vectors.npz is retained on Delta, with parameter names and sizes.
This download archive deliberately contains only summaries/config/logs to
stay small. MC prefixes reuse draws and are not independent experiments.
'''
    (report / 'REPORT.md').write_text(text, encoding='utf-8')
    write(report / 'complete.json', dict(status='complete', config_sha256=config_hash,
                                        n_cells=len(c['cells']), n_rows=len(all_rows)))
    with tarfile.open(archive, 'x:gz') as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.md', '.txt', '.sha256', '.out', '.err'):
                tar.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
    print(str(archive), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--smallpanel', type=Path, default=Path(DEFAULT_SP))
    q.add_argument('--loss-audit', type=Path, default=Path(DEFAULT_AUDIT))
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--profile', choices=['pilot', 'matrix'], default='pilot')
    q.add_argument('--reps', type=int, choices=[32, 64, 128, 256], default=128)
    for name in ('submit', 'collect', 'gpu'):
        q = sub.add_parser(name)
        q.add_argument('--out', type=Path, required=True)
        if name == 'gpu':
            q.add_argument('--index', type=int, default=0)
            q.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    {'prepare': prepare, 'gpu': gpu, 'submit': submit, 'collect': collect}[a.action](a)


if __name__ == '__main__':
    main()
