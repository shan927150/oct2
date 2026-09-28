#!/usr/bin/env python3
"""Stable cold-start baseline pilot, then separately submitted paired patient doses."""
import argparse
import copy
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import runtime as rt

SCHEMA = 'OCT_stable_training_v1'
SEEDS = [42, 43, 44]


def prepare(a):
    if a.epochs < 1:
        raise ValueError('epochs must be positive')
    c = rt.prepare_base(a, SCHEMA)
    c.update(profile=a.profile, epochs=a.epochs, seeds=SEEDS, slurm_time='02:00:00',
        recipe=dict(lr=.001, warmup_frac=.1, ema_decay=.99, wd=1e-5, batch=128),
        design='From initialization, no dropout, Adam warmup/cosine; raw + EMA; fixed-mask patient weights; no attack/LDS',
        request_note='User requested isolated mentor-recipe experiment branch on 2026-09-28')
    for seed in SEEDS:
        path = Path(c['full']) / f'stage1_order_seed{seed}.npz'
        c['input_sha256'][str(path)] = rt.sha(path)
    if a.profile == 'baseline':
        if a.baseline_out:
            raise ValueError('--baseline-out is only for paired profile')
        c['cells'] = [dict(name=f'seed{s}_baseline', seed=s, patient=None, alpha=0.) for s in SEEDS]
        c['cells'].append(dict(name='seed42_noop', seed=42, patient=None, alpha=0.))
    else:
        if not a.baseline_out:
            raise ValueError('paired requires --baseline-out from a collected baseline pilot')
        b = rt.config(a.baseline_out, SCHEMA)
        report = rt.read(Path(b['out']) / 'report/complete.json')
        if b['profile'] != 'baseline' or b['epochs'] != a.epochs or b['recipe'] != c['recipe']:
            raise ValueError('Baseline recipe/horizon mismatch')
        if not report.get('noop_passed') or report['config_sha256'] != rt.sha(Path(b['out']) / 'config.json'):
            raise ValueError('Collect baseline and pass its full-horizon no-op before paired runs')
        rt.results(b)
        rt.disjoint(c['out'], b['out'])
        c['baseline_out'] = b['out']
        for path in [Path(b['out']) / 'config.json', Path(b['out']) / 'report/complete.json']:
            c['input_sha256'][str(path)] = rt.sha(path)
        for seed in SEEDS:
            d = Path(b['out']) / f'seed{seed}_baseline'
            for name in ('final.pt', 'interface.npz', 'complete.json'):
                c['input_sha256'][str(d / name)] = rt.sha(d / name)
        c['cells'] = [dict(name=f'seed{s}_patient{p}_a{alpha:g}', seed=s, patient=p, alpha=alpha)
                      for s in SEEDS for p in c['patients'] for alpha in (.1, 1.)]
    rt.save_config(c)


def gpu(a):
    import numpy as np
    import torch
    from audit_torch import snapshot, preserved
    from stable_core import train
    c = rt.config(a.out, SCHEMA)
    if not a.smoke:
        rt.completed(Path(c['out']) / 'smoke', rt.sha(Path(c['out']) / 'config.json'))
    if not 0 <= a.index < len(c['cells']):
        raise ValueError('Index out of bounds')
    cell = c['cells'][a.index]
    dest = Path(c['out']) / ('smoke' if a.smoke else cell['name'])
    dest.mkdir(exist_ok=False)
    ctx, legacy, pilot, conv = rt.load_context(c, cell['seed'])
    import stage1_diagnostic_core as core
    started = time.monotonic()
    epochs = 2 if a.smoke else c['epochs']
    # Smoke has its own 2-epoch schedule, formal runs use the frozen horizon.
    orders = pilot.make_epoch_orders(ctx['train'], epochs, cell['seed'] + 700000)
    recorded = np.load(Path(c['full']) / f"stage1_order_seed{cell['seed']}.npz")['raw_index_order']
    n = min(len(recorded), len(orders))
    if not np.array_equal(orders[:n], recorded[:n]):
        raise ValueError('Training order prefix differs from frozen seed panel')
    patients = ctx['patient_rows']
    selected = [] if cell['patient'] is None else patients[cell['patient']]
    snapkw = dict(X=ctx['X'], y=ctx['y'], indices=ctx['train'], patient_rows=patients,
                  batch=c['batch'], wd=c['weight_decay'])

    def audit(m):
        for p in m.parameters():
            p.requires_grad_(True)
        return snapshot(m, **snapkw)

    def metrics(m, label):
        info, vectors = audit(m)
        j = None if cell['patient'] is None else c['patients'].index(cell['patient']) + 1
        g = vectors[0] if j is None else vectors[0] - cell['alpha'] * vectors[j]
        weighted = info['ce'] if j is None else info['ce'] - cell['alpha'] * info['patient_ce_contribution'][cell['patient']]
        return {label+'_eval_ce': info['ce'], label+'_weighted_objective': weighted+info['l2_penalty'],
                label+'_eval_grad_norm': float(np.linalg.norm(g))}

    def observe(model, ema, epoch):
        row = metrics(model, 'raw')
        if epoch % 10 == 0 or epoch == epochs:
            row.update(metrics(ema, 'ema'))
        return row

    def checkpoint(model, ema, opt, epoch, step, name):
        torch.save(dict(raw={k: v.detach().cpu() for k, v in model.state_dict().items()},
            ema={k: v.detach().cpu() for k, v in ema.state_dict().items()}, optimizer=opt.state_dict(),
            rng_states=conv.rng_payload(), metadata=dict(schema=SCHEMA, cell=cell, epoch=epoch, step=step,
            schedule_horizon=epochs, recipe=c['recipe'], dropout=0., epoch_order_sha256=conv.order_sha(orders),
            config_sha256=rt.sha(Path(c['out']) / 'config.json'))), dest / name)

    def initialize():
        pilot.seed_everything(cell['seed'], True)
        return pilot.build_model('cnn', ctx['X'].shape[1], 128, int(np.max(ctx['y'])) + 1)

    model, ema, opt, curve = train(initialize(), ctx['X'], ctx['y'], orders, selected, cell['alpha'],
        **c['recipe'], observe=observe,
        save=lambda m, e, o, ep, st: checkpoint(m, e, o, ep, st, f'epoch{ep:03d}.pt'),
        save_epochs=() if a.smoke else [ep for ep in (50, 100) if ep < epochs])
    checkpoint(model, ema, opt, epochs, curve[-1]['step'], 'final.pt')
    smoke_exact = None
    if a.smoke:
        m2, e2, o2, _ = train(initialize(), ctx['X'], ctx['y'], orders, selected, cell['alpha'], **c['recipe'])
        smoke_exact = (core.exact_tree(model.state_dict(), m2.state_dict()) and
                       core.exact_tree(ema.state_dict(), e2.state_dict()) and
                       core.exact_tree(opt.state_dict(), o2.state_dict()))
        if not smoke_exact:
            raise RuntimeError('Smoke full training replay failed (including observation neutrality)')
        del m2, e2, o2
    endpoint = []
    arrays = dict(raw_index=ctx['interface'], membership=np.r_[np.ones(2045), np.zeros(len(ctx['test']))],
                  classes=ctx['y'][ctx['interface']])
    deltas = {}
    baseline = None
    if c['profile'] == 'paired' and not a.smoke:
        d = Path(c['baseline_out']) / f"seed{cell['seed']}_baseline"
        baseline = torch.load(d / 'final.pt', map_location='cpu', weights_only=False)
        with np.load(d / 'interface.npz') as z:
            base_probs = {key: z[key].copy() for key in ('raw', 'ema')}
    for label, m in [('raw', model), ('ema', ema)]:
        with preserved(m):
            probs = pilot.get_predictions(m, ctx['X'][ctx['interface']])
        arrays[label] = probs
        md = copy.deepcopy(m).double()
        info, vectors = audit(md)
        del md
        j = None if cell['patient'] is None else c['patients'].index(cell['patient'])+1
        g = vectors[0] if j is None else vectors[0]-cell['alpha']*vectors[j]
        test_probs = probs[2045:]
        test_y = ctx['y'][ctx['test']]
        row = dict(seed=cell['seed'], patient=cell['patient'], alpha=cell['alpha'], endpoint=label,
            eval_ce=info['ce'], weighted_objective=info['objective']-(0 if j is None else
            cell['alpha']*info['patient_ce_contribution'][cell['patient']]),
            eval_grad_norm_fp64=float(np.linalg.norm(g)),
            heldout_accuracy=float(np.mean(test_probs.argmax(1)==test_y)),
            heldout_ce=float(-np.log(np.maximum(test_probs[np.arange(len(test_y)), test_y], 1e-30)).mean()))
        for k, pid in enumerate(c['patients'], 1):
            bnorm = float(np.linalg.norm(vectors[k]))
            row[f'b_{pid}_norm_fp64'] = bnorm
            row[f'baseline_rho01_{pid}'] = float(np.linalg.norm(vectors[0])) / (.1*bnorm) if bnorm else None
        if baseline:
            # Identical parameter names/order for raw and EMA.
            delta = torch.cat([(p.detach().cpu().double()-baseline[label][name].double()).reshape(-1)
                               for name, p in m.named_parameters()]).numpy()
            base_theta = torch.cat([baseline[label][name].double().reshape(-1) for name, _ in m.named_parameters()])
            row['relative_theta_shift'] = float(np.linalg.norm(delta)/torch.linalg.vector_norm(base_theta))
            row['mean_interface_TV'] = float(np.abs(probs-base_probs[label]).sum(1).mean()/2)
            deltas[label] = delta
        endpoint.append(row)
    np.savez_compressed(dest / 'interface.npz', **arrays)
    if deltas:
        np.savez_compressed(dest / 'parameter_deltas.npz', **deltas)
    rt.csv_write(dest / 'curve.csv', curve)
    rt.csv_write(dest / 'endpoint.csv', endpoint)
    rt.write(dest / 'result.json', dict(cell=cell, endpoint=endpoint,
        epoch_order_sha256=conv.order_sha(orders), smoke_exact_replay=smoke_exact,
        final_rng_sha256=core.rng_fingerprint(), epochs=epochs))
    if conv.dataset_fingerprint(ctx['X'], ctx['y']) != c['dataset_fingerprint']:
        raise ValueError('Dataset mutated')
    rt.config(a.out, SCHEMA)
    rt.finish(dest, c, cell, smoke=a.smoke, input_and_source_unchanged=True,
              elapsed_seconds=time.monotonic()-started, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    print('COMPLETE ' + str(dest), flush=True)


def geometry(a, b):
    import numpy as np
    a, b = np.ravel(a).astype(float), np.ravel(b).astype(float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(R=float(na/(.1*nb)) if nb else None,
                cosine=float(a@b/(na*nb)) if na and nb else None,
                relative_linear_error=float(np.linalg.norm(a-.1*b)/(.1*nb)) if nb else None)


def collect(a):
    import numpy as np
    c = rt.config(a.out, SCHEMA)
    values = rt.results(c)
    root = Path(c['out'])
    comparisons = []
    if c['profile'] == 'baseline':
        # Compare saved state content, not serialization bytes (filenames alter archives).
        # Full-horizon parameter/optimizer/RNG replay is checked in this explicit
        # PyTorch stage; collect is run after loading the same Delta module.
        import torch
        sys.path.insert(0, str(rt.REPO / 'scripts/cross_stage'))
        from stage1_diagnostic_core import exact_tree
        left = torch.load(root/'seed42_baseline/final.pt', map_location='cpu', weights_only=False)
        right = torch.load(root/'seed42_noop/final.pt', map_location='cpu', weights_only=False)
        for k in ('raw', 'ema', 'optimizer', 'rng_states'):
            if not exact_tree(left[k], right[k]):
                raise ValueError('Full-horizon no-op replay failed: ' + k)
        with np.load(root/'seed42_baseline/interface.npz') as x, np.load(root/'seed42_noop/interface.npz') as y:
            if any(not np.array_equal(x[k], y[k]) for k in x.files):
                raise ValueError('No-op interface changed')
        for i, s in enumerate(SEEDS):
            for t in SEEDS[i+1:]:
                with np.load(root/f'seed{s}_baseline/interface.npz') as x, np.load(root/f'seed{t}_baseline/interface.npz') as y:
                    for label in ('raw', 'ema'):
                        comparisons.append(dict(seed1=s, seed2=t, endpoint=label,
                            mean_interface_TV=float(np.abs(x[label]-y[label]).sum(1).mean()/2)))
    else:
        for seed in SEEDS:
            with np.load(Path(c['baseline_out'])/f'seed{seed}_baseline/interface.npz') as z:
                baseline = {k: z[k].copy() for k in ('raw','ema')}
            for patient in c['patients']:
                d1 = root/f'seed{seed}_patient{patient}_a0.1'
                d2 = root/f'seed{seed}_patient{patient}_a1'
                with np.load(d1/'parameter_deltas.npz') as x, np.load(d2/'parameter_deltas.npz') as y:
                    for label in ('raw','ema'):
                        comparisons.append(dict(seed=seed, patient=patient, endpoint=label, space='theta', **geometry(x[label], y[label])))
                with np.load(d1/'interface.npz') as x, np.load(d2/'interface.npz') as y:
                    for label in ('raw','ema'):
                        comparisons.append(dict(seed=seed, patient=patient, endpoint=label, space='P',
                            **geometry(x[label]-baseline[label], y[label]-baseline[label])))
    rep = root / 'report'
    rep.mkdir(exist_ok=False)
    rt.csv_write(rep/'endpoint_summary.csv', [r for v in values for r in v['endpoint']])
    rt.csv_write(rep/'comparisons.csv', comparisons)
    (rep/'REPORT.md').write_text('# Stable recipe '+c['profile']+'\n\n'
        'Shadow 3 only. Fixed split, rows and original denominators. No dropout, 10% warmup, cosine, EMA .99.\n'
        'Raw and EMA are two endpoints of the same trajectory, not independent replicates.\n'
        'Cross-seed baseline TV is a sensitivity reference, not a strict deletion null distribution.\n'
        'No Stage 2 retraining, query Delta J, LDS or new L3 is computed.\n'
        'No-op checks reproducibility, not convergence or robustness to patient deletion.\n'
        'R and cosine in paired profile compare two REAL deletion doses, not IF vs truth.\n'
        'EMA is not guaranteed stationary. Gradient size/ratio is diagnostic only.\n'
        'Checkpoints/parameter deltas stay on Delta; interface.npz is included in the archive.\n')
    rt.write(rep/'complete.json', dict(status='complete', config_sha256=rt.sha(root/'config.json'),
                                      noop_passed=True if c['profile']=='baseline' else None))
    rt.archive(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    q = rt.common_cli(p)
    q.add_argument('--profile', choices=['baseline','paired'], default='baseline')
    q.add_argument('--epochs', type=int, default=140)
    q.add_argument('--baseline-out')
    a = p.parse_args()
    if a.action == 'submit':
        rt.submit(a, SCHEMA, 'stb1')
    else:
        {'prepare':prepare, 'gpu':gpu, 'collect':collect}[a.action](a)


if __name__ == '__main__':
    main()
