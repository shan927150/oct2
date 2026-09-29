#!/usr/bin/env python3
"""Isolated patient-cluster LDS screening with a formula/diagnostic ladder."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
sys.dont_write_bytecode=True
import runtime as rt

SCHEMA='OCT_patient_LDS_formula_v1'
BASELINE='/u/yli103/oct2-pathway2-runs/A/stable_training_v1_baseline_20260928T181445049797182Z'


def prepare(a):
    if not 8<=a.m<=100 or not 0<a.retain<1 or not 0<a.alpha<=1:
        raise ValueError('Need 8<=m<=100, 0<retain<1, 0<alpha<=1')
    if a.mode=='subset' and a.alpha!=1:
        raise ValueError('Actual subset training requires alpha=1')
    if not 1<=a.concurrency<=4:
        raise ValueError('Concurrency must be 1..4')
    c=rt.prepare_base(a,SCHEMA)
    bpath=Path(a.baseline_out).resolve()
    b=rt.read(bpath/'config.json'); bh=rt.sha(bpath/'config.json')
    if bh!=(bpath/'config.sha256').read_text().strip():
        raise ValueError('Baseline config checksum changed')
    expected=dict(lr=.001,warmup_frac=.1,ema_decay=.99,wd=1e-5,batch=128)
    if (b['schema']!='OCT_stable_training_v1' or b['profile']!='baseline' or b['epochs']!=140
        or b['recipe']!=expected or b['git_commit']!=rt.PARENT or b['dataset_fingerprint']!=c['dataset_fingerprint']):
        raise ValueError('Need the frozen b14b69e stable baseline, epochs=140')
    report=rt.read(bpath/'report/complete.json')
    if report['config_sha256']!=bh or report.get('noop_passed') is not True:
        raise ValueError('Collect baseline and pass full no-op first')
    rt.disjoint(c['out'],bpath)
    rt.verify_hashes(b['input_sha256'])
    for rel in ['config.json','config.sha256','report/complete.json']:
        c['input_sha256'][str(bpath/rel)]=rt.sha(bpath/rel)
    for cell in b['cells']:
        d=bpath/cell['name']; rt.completed(d,bh)
        for rel in ('complete.json','final.pt','interface.npz'):
            c['input_sha256'][str(d/rel)]=rt.sha(d/rel)
    inherited=rt.REPO/'experiments/pathway2_stable_v1'
    for p in sorted(inherited.glob('*.py')):
        c['source_sha256'][str(p)]=rt.sha(p)
    for seed in (42,43,44):
        p=Path(c['full'])/f'stage1_order_seed{seed}.npz'
        c['input_sha256'][str(p)]=rt.sha(p)
    c.update(baseline_out=str(bpath),m=a.m,retain=a.retain,alpha=a.alpha,mode=a.mode,
        subset_seed=20260929,seeds=[42,43,44],epochs=140,recipe=expected,
        gamma=1.,cg_steps=80,cg_tol=1e-5,cg_fail_tol=1e-3,lanczos_steps=12,
        concurrency=a.concurrency,slurm_time='03:00:00',
        primary='patient/class macro heldout CE; raw endpoint; no attack retraining',
        endpoint='raw',holdout_policy='Do not tune gamma, sign, query or recipe on subset truth',
        interpretation='Screening only. LDS is not a proof of the local formula.')
    c['cells']=[dict(name=f'subset{j:03d}_seed{s}',subset=j,seed=s) for j in range(c['m']) for s in c['seeds']]
    rt.save_config(c)


def config(out):
    return rt.config(out,SCHEMA)


def submit(a):
    c=config(a.out); out=Path(c['out'])
    # One smoke, score array, full-horizon no-op, then subset array.
    with (out/'submission.jsonl').open('x') as ledger:
        def launch(mode,extra):
            cmd=['sbatch','--parsable','--account='+c['account'],'--partition=gpuA40x4',
                 '--job-name=oct_lds1_'+mode,'--output='+str(out/'logs/%x-%A_%a.out'),
                 '--error='+str(out/'logs/%x-%A_%a.err'),*extra,str(rt.HERE/'job.slurm'),
                 str(rt.HERE),str(out),mode]
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Unexpected sbatch result')
            ledger.write(json.dumps(dict(mode=mode,job_id=job,command=cmd))+'\n');ledger.flush()
            (out/(mode+'_job.txt')).write_text(job+'\n')
            return job
        def after(job):
            return ['--dependency=afterok:'+job,'--kill-on-invalid-dep=yes']
        smoke=launch('smoke',['--time=00:45:00'])
        score=launch('score',['--time='+c['slurm_time'],'--array=0-2%'+str(c['concurrency']),*after(smoke)])
        noop=launch('noop',['--time=00:45:00',*after(score)])
        truth=launch('truth',['--time=00:45:00','--array=0-'+str(len(c['cells'])-1)+'%'+str(c['concurrency']),*after(score+':'+noop)])
    print(f'OUT={out}\nSMOKE={smoke}\nSCORES={score}\nNOOP={noop}\nTRUTH={truth}')


def gpu(a):
    from core import score_seed,truth_cell
    c=config(a.out);root=Path(c['out']);h=rt.sha(root/'config.json')
    start=time.monotonic()
    if a.mode!='smoke':
        rt.completed(root/'smoke',h)
    if a.mode=='local':
        from local_response import run as local_run
        rt.completed(root/'score_seed42',h)
        d=root/'local_response';d.mkdir(exist_ok=False)
        local_run(c,d)
        config(a.out)
        rt.finish(d,c,dict(kind='recentered_proximal',seed=42),elapsed_seconds=time.monotonic()-start)
        print('COMPLETE '+str(d),flush=True)
        return
    if a.mode in ('smoke','score'):
        if not 0<=a.index<len(c['seeds']):
            raise ValueError('Bad seed index')
        seed=c['seeds'][a.index]
        cell=dict(kind=a.mode,seed=seed)
        d=root/('smoke' if a.mode=='smoke' else f'score_seed{seed}')
        d.mkdir(exist_ok=False)
        if a.mode=='smoke':
            from formula_checks import implicit_check,trajectory_check
            rt.write(d/'formula_checks.json',dict(implicit=implicit_check(),trajectory=trajectory_check()))
            # trajectory_check uses float64; restore the inherited float32 training default.
            import torch
            torch.set_default_dtype(torch.float32)
        score_seed(c,d,seed,smoke=a.mode=='smoke')
        if a.mode=='score' and not rt.read(d/'result.json')['solver_qualified']:
            raise ValueError('Unqualified inverse-Hessian solve: results retained; subset jobs remain gated. Do not choose damping from subset outcomes.')
    else:
        if a.mode=='noop':
            # No-op needs the seed42 linearization, so its explicit dependency is
            # checked here as well (submit schedules after the score array).
            cell=dict(name='noop',subset=None,seed=42)
        else:
            if not 0<=a.index<len(c['cells']):
                raise ValueError('Bad subset cell index')
            rt.completed(root/'noop',h)
            cell=c['cells'][a.index]
        rt.completed(root/f"score_seed{cell['seed']}",h)
        d=root/cell['name'];d.mkdir(exist_ok=False)
        truth_cell(c,d,cell)
    config(a.out)
    rt.finish(d,c,cell,elapsed_seconds=time.monotonic()-start)
    print('COMPLETE '+str(d),flush=True)


def read_scores(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def collect(a):
    import numpy as np
    from design import intervention
    from analysis_stats import evaluate,spearman,reliability,crossfit_ridge
    c=config(a.out);root=Path(c['out']);h=rt.sha(root/'config.json')
    rt.completed(root/'smoke',h);rt.completed(root/'noop',h)
    design=rt.read(root/'score_seed42/design.json')
    m,k=c['m'],len(c['seeds'])
    Y=np.zeros((m,k));D=Y.copy();L1=Y.copy();L2=Y.copy();L3=Y.copy();canonical=Y.copy();frozen=Y.copy()
    qualified=[];row_details=[]
    for si,seed in enumerate(c['seeds']):
        sd=root/f'score_seed{seed}';rt.completed(sd,h)
        if rt.read(sd/'design.json')!=design:
            raise ValueError('Seed designs differ')
        sr=rt.read(sd/'result.json');qualified.append(sr['solver_qualified'])
        rows=read_scores(sd/'scores.csv')
        if [int(x['patient']) for x in rows] != [x['patient'] for x in design['train_patients']]:
            raise ValueError('Score patient order mismatch')
        tau=np.array([float(x['tau_upweight']) for x in rows])
        proxy=np.array([float(x['frozen_h_upweight']) for x in rows])
        for j in range(m):
            cell=c['cells'][j*k+si];td=root/cell['name'];cm=rt.completed(td,h)
            if cm['cell']!=cell:
                raise ValueError('Cell marker mismatch')
            r=rt.read(td/'result.json')
            _,delta_weight=intervention(design,j,c['mode'],c['alpha'],sum(len(p['rows']) for p in design['train_patients']))
            mask=np.asarray(design['masks'][j])
            Y[j,si]=r['Y'];D[j,si]=r['delta_Y'];L1[j,si]=r['L1_true_P'];L2[j,si]=r['L2_true_theta']
            L3[j,si]=delta_weight@tau;canonical[j,si]=mask@tau;frozen[j,si]=delta_weight@proxy
            row_details.append(dict(subset=j,seed=seed,delta_Y=D[j,si],L1=L1[j,si],L2=L2[j,si],
                                    L3=L3[j,si],canonical_sum=canonical[j,si],frozen_h=frozen[j,si]))
    report=dict(protocol=dict(mode=c['mode'],alpha=c['alpha'],retain=c['retain'],m=m,k=k,
                             target=c['primary'],gamma=c['gamma']),
                solver_qualified_by_seed=dict(zip(map(str,c['seeds']),qualified)),
                reliability=reliability(Y),numerical_proof_scope='See smoke/formula_checks.json and chain_fd.csv',
                empirical_formula_proof=False)
    report['L1_diagnostic']=evaluate(L1.mean(1),D.mean(1))
    report['L2_diagnostic']=evaluate(L2.mean(1),D.mean(1))
    for label in ('L1_diagnostic','L2_diagnostic'):
        report[label].pop('permutation_p_one_sided',None)
        report[label]['interpretation']='Uses subset truth. Mechanistic diagnostic, not predictive attribution.'
    if all(qualified):
        report['L3_stage1_normalization_matched']=evaluate(L3.mean(1),D.mean(1))
        result=evaluate(canonical.mean(1),Y.mean(1))
        # Canonical sum has an arbitrary missing intercept and amplitude. Its
        # uncalibrated error is not a valid finite-difference accuracy measure.
        for key in ('uncalibrated_rmse','relative_error','slope_truth_on_prediction','intercept'):
            result.pop(key,None)
        report['LDS_canonical_retained_sum']=result
        report['L3_by_seed']=[dict(seed=s,**evaluate(L3[:,i],D[:,i])) for i,s in enumerate(c['seeds'])]
    else:
        report['L3_stage1_normalization_matched']=None
        report['LDS_canonical_retained_sum']=None
        report['L3_status']='UNQUALIFIED_SOLVER. No subset-selected damping or silent subset-of-seeds score.'
    report['frozen_h_proxy']=evaluate(frozen.mean(1),D.mean(1))
    oof=crossfit_ridge(design['masks'],Y.mean(1))
    report['nested_crossfit_ridge_diagnostic']=dict(spearman=spearman(oof,Y.mean(1)),
        mse=float(np.mean((oof-Y.mean(1))**2)),
        interpretation='Nested heldout-subset regression, severely data-limited at m=8; not an oracle or upper bound.')
    composition=[]
    for mask in design['masks']:
        counts=[]
        for cls in range(4):
            ps=[p for keep,p in zip(mask,design['train_patients']) if keep and p['oct_class']==cls]
            counts.extend([len(ps),sum(len(p['rows']) for p in ps)])
        composition.append(counts)
    counts_oof=crossfit_ridge(composition,Y.mean(1))
    report['composition_only_crossfit_diagnostic']=dict(spearman=spearman(counts_oof,Y.mean(1)),
        interpretation='Heldout-subset ridge using only retained patient/image counts by class; no upper-bound claim.')
    report['uncertainty_scope']='Bootstrap resamples whole subset rows jointly across shared seed columns; conditional on 42/43/44. m=8 is exploratory. No formal pass/fail claim.'
    rep=root/'report';rep.mkdir(exist_ok=False)
    rt.csv_write(rep/'ladder_rows.csv',row_details)
    rt.write(rep/'summary.json',report)
    (rep/'REPORT.md').write_text('# Patient LDS screening\n\n'
        'Read summary.json. Target: class-balanced patient-macro heldout CE, raw endpoint.\n'
        'Full cross-stage attack J is not computed in this pilot. L3_stage1 names only the Stage-1 inverse-Hessian factor.\n'
        'L1 and L2 use measured truth; their correlations are diagnostic, not attribution performance.\n'
        'Canonical retained-score LDS and normalization-matched finite-effect prediction are separate.\n'
        'Positive LDS does not establish stationarity, local parameter accuracy, additivity or the full Eq.79.\n'
        'Frozen_h is a one-checkpoint gradient proxy, not full TracIn. Ridge is cross-fitted, not an upper bound.\n'
        'Bootstrap uncertainty conditions on the seed panel. Expand only after inspecting repeatability and solver status.\n')
    rt.write(rep/'complete.json',dict(status='complete',config_sha256=h))
    archive(root)


def archive(root,debug=False):
    target=Path(str(root)+('.debug' if debug else '')+'.tar.gz')
    with tarfile.open(target,'x:gz') as t:
        for p in sorted(root.rglob('*')):
            if not debug and 'local_response' in p.relative_to(root).parts:
                continue
            if p.is_file() and (p.suffix in ('.json','.jsonl','.csv','.md','.txt','.sha256','.out','.err') or p.name=='interface.npz'):
                t.add(p,arcname=str(Path(root.name)/p.relative_to(root)))
    print(str(target))


def submit_local(a):
    c=config(a.out);root=Path(c['out'])
    dependency=(root/'score_job.txt').read_text().strip()
    if not dependency.isdigit():
        raise ValueError('Submit main smoke/scores first')
    with (root/'local_submission.json').open('x') as f:
        cmd=['sbatch','--parsable','--account='+c['account'],'--partition=gpuA40x4',
             '--job-name=oct_lds1_local','--time=03:00:00',
             '--output='+str(root/'logs/%x-%A_%a.out'),'--error='+str(root/'logs/%x-%A_%a.err'),
             '--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes',
             str(rt.HERE/'job.slurm'),str(rt.HERE),str(root),'local']
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected sbatch result')
        json.dump(dict(job_id=job,command=cmd),f,indent=2)
        (root/'local_job.txt').write_text(job+'\n')
    print('LOCAL='+job)


def collect_local(a):
    c=config(a.out);root=Path(c['out'])
    rt.completed(root/'local_response',rt.sha(root/'config.json'))
    target=Path(str(root)+'.local.tar.gz')
    with tarfile.open(target,'x:gz') as t:
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.parent==root or 'local_response' in p.parts or p.name.startswith('oct_lds1_local')
                               or (p.parent.name=='score_seed42' and p.suffix in ('.json','.csv'))):
                t.add(p,arcname=str(Path(root.name)/p.relative_to(root)))
    print(str(target))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out',required=True);q.add_argument('--audit',default=rt.AUDIT)
    q.add_argument('--baseline-out',default=BASELINE);q.add_argument('--m',type=int,default=8)
    q.add_argument('--retain',type=float,default=.5);q.add_argument('--alpha',type=float,default=1.)
    q.add_argument('--mode',choices=['subset','fixed_mask'],default='subset')
    q.add_argument('--concurrency',type=int,default=1)
    for name in ('submit','collect','debug','gpu','submit-local','collect-local'):
        q=sub.add_parser(name);q.add_argument('--out',required=True)
        if name=='gpu':
            q.add_argument('--mode',choices=['smoke','score','noop','truth','local'],required=True)
            q.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.action=='debug':
        archive(Path(a.out).resolve(),True)
    else:
        globals()[a.action.replace('-','_')](a)


if __name__=='__main__':
    main()
