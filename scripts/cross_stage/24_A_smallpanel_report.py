#!/usr/bin/env python3
"""Strict paired E100/E140 report. Refuses any missing, duplicate or altered cell."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import route_a_smallpanel as common


def payload(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def flat(p, keys=None):
    state = p['state_dict']
    if keys is not None and list(state) != keys:
        raise ValueError('Parameter names/order differ')
    value = torch.cat([v.detach().cpu().reshape(-1).double() for v in state.values()]).numpy()
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite model parameters')
    return value


def geometry(full, dose, theta=None):
    full, dose = np.asarray(full).ravel(), np.asarray(dose).ravel()
    nf, nd = float(np.linalg.norm(full)), float(np.linalg.norm(dose))
    if min(nf, nd) <= 0 or not np.isfinite([nf, nd]).all():
        raise ValueError('Zero/nonfinite response; cannot interpret dose geometry')
    result = dict(full_norm=nf, dose_norm=nd, R=nd/(.1*nf),
                  cosine=float(np.dot(full, dose)/(nf*nd)),
                  relative_linear_error=float(np.linalg.norm(dose-.1*full)/(.1*nf)))
    if theta is not None:
        result['full_over_theta'] = nf / float(np.linalg.norm(theta))
    return result


def read_curve(path, epoch, column):
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r['epoch']) == epoch]
    if len(rows) != 1 or not np.isfinite(float(rows[0][column])):
        raise ValueError(f'Missing/duplicate/nonfinite gradient at E{epoch}: {path}')
    return float(rows[0][column])


def load_runs(root, cfg):
    expected = set(common.expected_cells())
    actual = {p.name for p in root.glob('seed*') if p.is_dir()}
    if actual != expected:
        raise ValueError(f'Panel incomplete or unexpected: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    runs = {}
    for name in sorted(expected):
        d = root/name
        m = common.read(d/'manifest.json')
        common.validate_cell_manifest(m, cfg, name)
        for epoch in (100, 140):
            p = d / f'checkpoints/epoch{epoch:03d}.pt'
            if common.sha(p) != m['checkpoint_sha256'][p.name]:
                raise ValueError(f'Checkpoint changed: {p}')
        for fname in ('curve.csv', 'replay_checks.json', 'interface.npz'):
            if common.sha(d/fname) != m['artifact_sha256'][fname]:
                raise ValueError(f'Artifact changed: {d/fname}')
        checks = common.read(d/'replay_checks.json')
        if len(checks) != 1 or not all(checks[0].get(k) is True for k in
            ('passed','parameters_equal','rng_fingerprint_equal','optimizer_hash_equal','recorded_orders_equal')):
            raise ValueError(f'E100 replay failed: {name}')
        runs[name] = (d, m)
    return runs


def check_loo(p, m, epoch, cfg):
    md = p['metadata']
    wanted = {k:m[k] for k in ('seed','patient_id','dose','arm','deletion_weight',
                               'resume_epoch','total_epochs','decay_schedule','base_lr')}
    wanted.update(epoch=epoch, source=common.SCHEMA)
    for k,v in wanted.items():
        if md.get(k) != v:
            raise ValueError(f'LOO checkpoint metadata mismatch: {k}')
    if not md.get('optimizer_state') or not md.get('rng_states'):
        raise ValueError('LOO checkpoint lacks continuation state')
    fact = cfg['probe_facts'][f"{m['seed']}/{m['arm']}"]
    if epoch == 140 and (md['epoch_order_sha256'] != fact['epoch_order_sha256'] or
                        md['rng_fingerprint'] != fact['final_rng_sha256']):
        raise ValueError('LOO/probe RNG or epoch order mismatch')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=Path)
    a = ap.parse_args(argv)
    root,cfg = common.load_config(a.root)
    cfg = dict(cfg, config_sha256=common.sha(root/common.CONFIG))
    runs = load_runs(root,cfg)
    e100,probe = Path(cfg['e100_root']),Path(cfg['probe_root'])
    geom_rows, interface_rows, grad_rows = [],[],[]
    for seed in common.SEEDS:
        base100 = payload(e100/f'shadow3_full/checkpoints/shadow_3_baseline_seed{seed}.pt')
        keys = list(base100['state_dict']); theta100 = flat(base100)
        dosebase = flat(payload(e100/f'shadow3_dose01/checkpoints/shadow_3_baseline_seed{seed}.pt'),keys)
        if not np.array_equal(theta100,dosebase):
            raise ValueError('E100 full/dose baselines differ')
        for pid in common.PATIENTS:
            truth = {dose:flat(payload(e100/f'shadow3_{dose}/checkpoints/shadow_3_seed{seed}_patient{pid}.pt'),keys)
                     for dose in common.DOSES}
            geom_rows.append(dict(seed=seed,patient_id=pid,endpoint='E100_truth',arm='truth',
                                  **geometry(truth['full']-theta100,truth['dose01']-theta100,theta100)))
            for arm in common.ARMS:
                bpath = probe/f'shadow3_seed{seed}_{arm}'
                bp = payload(bpath/'checkpoints/epoch140.pt')
                common.check_baseline_metadata(bp,seed,arm,cfg['probe_facts'][f'{seed}/{arm}'])
                theta = flat(bp,keys); delta = {}; interfaces = {}
                for dose in common.DOSES:
                    d,m = runs[common.cell_name(seed,pid,dose,arm)]
                    for epoch in (100,140):
                        p = payload(d/f'checkpoints/epoch{epoch:03d}.pt')
                        check_loo(p,m,epoch,cfg)
                        if epoch == 100 and not np.array_equal(flat(p,keys),truth[dose]):
                            raise ValueError('Saved E100 endpoint differs from truth')
                        if epoch == 140: delta[dose] = flat(p,keys)-theta
                        grad_rows.append(dict(seed=seed,patient_id=pid,arm=arm,dose=dose,epoch=epoch,
                            weighted_eval_grad_norm=read_curve(d/'curve.csv',epoch,'weighted_eval_grad_norm')))
                    with np.load(d/'interface.npz') as f:
                        interfaces[dose] = {k:f[k].copy() for k in f.files}
                geom_rows.append(dict(seed=seed,patient_id=pid,endpoint='E140',arm=arm,
                                      **geometry(delta['full'],delta['dose01'],theta)))
                for epoch in (100,140):
                    f,q = interfaces['full'],interfaces['dose01']
                    if not np.array_equal(f['raw_indices'],q['raw_indices']) or not np.array_equal(f[f'baseline_{epoch}'],q[f'baseline_{epoch}']):
                        raise ValueError('Interface rows/baseline not paired')
                    df=f[f'loo_{epoch}'].astype(float)-f[f'baseline_{epoch}']
                    dq=q[f'loo_{epoch}'].astype(float)-q[f'baseline_{epoch}']
                    if not np.isfinite(df).all() or not np.isfinite(dq).all():
                        raise ValueError('Nonfinite interface response')
                    interface_rows.append(dict(seed=seed,patient_id=pid,epoch=epoch,arm=arm,
                        full_mean_tv=float(np.abs(df).sum(axis=1).mean()/2),
                        dose_mean_tv=float(np.abs(dq).sum(axis=1).mean()/2),**geometry(df,dq)))
        for arm in common.ARMS:
            for epoch in (100,140):
                grad_rows.append(dict(seed=seed,patient_id='',arm=arm,dose='baseline',epoch=epoch,
                    weighted_eval_grad_norm=read_curve(probe/f'shadow3_seed{seed}_{arm}/curve.csv',epoch,'eval_grad_norm')))
    paired=[]
    for s in common.SEEDS:
        for p in common.PATIENTS:
            c,d = [next(r for r in geom_rows if (r['seed'],r['patient_id'],r['endpoint'],r['arm'])==(s,p,'E140',arm)) for arm in common.ARMS]
            paired.append(dict(seed=s,patient_id=p,
                delta_abs_R_minus1=abs(d['R']-1)-abs(c['R']-1),
                delta_cosine=d['cosine']-c['cosine'],
                delta_full_over_theta=d['full_over_theta']-c['full_over_theta']))
    summaries={}
    for ep,arm in [('E100_truth','truth'),('E140','control'),('E140','decay')]:
        rs=[r for r in geom_rows if r['endpoint']==ep and r['arm']==arm]
        summaries[f'{ep}/{arm}']={'n':len(rs),**{k:float(np.median([r[k] for r in rs])) for k in ('R','cosine','full_over_theta','relative_linear_error')}}
    summary=dict(schema='pathway2_A_smallpanel_report_v3',complete_cells=40,geometry_rows=30,
        patients=common.PATIENTS,seeds=common.SEEDS,geometry_medians=summaries,
        paired_differences=paired,config_sha256=cfg['config_sha256'],git_commit=cfg['git_commit'])
    out=root/'report';out.mkdir(exist_ok=True)
    for name,rows in [('geometry_rows',geom_rows),('interface_rows',interface_rows),('gradient_rows',grad_rows),('paired_differences',paired)]:
        write_csv(out/f'{name}.csv',rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Small-panel paired LOO continuation','',
        '40/40 cells verified. Patients 807 (DME), 2085 (DRUSEN); five seeds. No attack retraining or new IF scores.',
        '', '| Endpoint / arm | n | median R | median cosine | median full/θ |', '|---|---:|---:|---:|---:|']
    for k,v in summaries.items():
        lines.append(f"| {k} | {v['n']} | {v['R']:.5g} | {v['cosine']:.5g} | {v['full_over_theta']:.5g} |")
    lines += ['', 'Compare decay with control on the same ten patient-seed pairs. E100 is recomputed on this panel; the prior eight-patient average 9.163 is not an expected value for this subset.',
        '', 'paired_differences.csv: negative delta_abs_R_minus1, positive delta_cosine, and negative delta_full_over_theta indicate improvement. Inspect individual pairs, not only pooled medians.',
        '', 'gradient_rows.csv contains each condition’s deletion-weighted eval-mode gradient (dropout off), with baseline gradients from the paired LR probe. These are not gradients of the dropout expected training objective.',
        '', 'interface_rows.csv records ΔP geometry and mean TV on identical affected-shadow member/nonmember rows. This is an interface diagnostic, not attack performance.',
        '', 'Improvement supports further study of this tail-stabilization recipe. No improvement only means this 40-epoch schedule did not restore these properties; it cannot locate when trajectories diverged or rule out non-stationarity. Two patients do not support population-level conclusions.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
