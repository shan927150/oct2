#!/usr/bin/env python3
"""Staged submit: fresh smoke -> baseline calibration -> T0=130 pilot."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
sys.dont_write_bytecode=True
import runtime_late as rt


def launch(c,tag,mode,options,ledger):
    root=Path(c['out'])
    cmd=['sbatch','--parsable','--account='+c['account'],'--partition=gpuA40x4',
         '--job-name=oct_ld2_'+tag,'--output='+str(root/'logs/%x-%A_%a.out'),
         '--error='+str(root/'logs/%x-%A_%a.err'),*options,str(rt.HERE/'job.slurm'),str(rt.HERE),str(root),mode]
    def log(v):ledger.write(json.dumps(v)+'\n');ledger.flush();os.fsync(ledger.fileno())
    log(dict(state='submitting',tag=tag,command=cmd));r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:
        log(dict(state='failed',tag=tag,stdout=r.stdout,stderr=r.stderr))
        raise RuntimeError('Submission failed; preserve ledger and inspect status before retrying.\n'+r.stderr)
    if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',r.stdout.strip()):
        log(dict(state='unknown_response',tag=tag,stdout=r.stdout,stderr=r.stderr))
        raise RuntimeError('Unknown scheduler response; inspect jobs before retrying')
    job=r.stdout.strip().split(';')[0];log(dict(state='submitted',tag=tag,job_id=job))
    with (root/(tag+'_job.txt')).open('x') as f:f.write(job+'\n')
    return job


def submit(a):
    c=rt.config(a.out);root=Path(c['out'])
    with (root/'initial_submission.jsonl').open('x') as ledger:
        smoke=launch(c,'smoke','smoke',['--time=01:00:00'],ledger)
        base=launch(c,'baseline','baseline',['--time=02:00:00','--dependency=afterok:'+smoke,'--kill-on-invalid-dep=yes'],ledger)
        pilot=launch(c,'pilot','cell',['--time=02:00:00','--dependency=afterok:'+base,'--kill-on-invalid-dep=yes'],ledger)
    print('OUT='+str(root)+'\nSMOKE='+smoke+'\nBASELINE='+base+'\nPILOT='+pilot,flush=True)
    print('Only T0=130 pilot follows baseline. Use submit-rest after validated completion.')


def prerequisites(c,pilot=False):
    root=Path(c['out']);h=rt.sha(root/'config.json')
    rt.completed(root/'smoke',h);rt.completed(root/'baseline',h)
    if rt.read(root/'baseline/result.json').get('exact_parent_replay') is not True:raise ValueError('Baseline calibration failed')
    if pilot:
        d=root/c['cells'][0]['name'];rt.completed(d,h)
        if rt.read(d/'result.json').get('quality_gate_passed') is not True:raise ValueError('Pilot invalid')


def submit_rest(a):
    c=rt.config(a.out);prerequisites(c,True);root=Path(c['out'])
    with (root/'rest_submission.jsonl').open('x') as ledger:
        job=launch(c,'rest','cell',['--time=04:00:00','--array=1-'+str(len(c['cells'])-1)+'%'+str(c['concurrency'])],ledger)
    print('OUT='+str(root)+'\nREST='+job,flush=True)


def gpu(a):
    import engine
    c=rt.config(a.out);root=Path(c['out']);h=rt.sha(root/'config.json')
    if a.mode!='smoke':rt.completed(root/'smoke',h)
    if a.mode=='cell':
        if not 0<=a.index<len(c['cells']):raise ValueError('Cell index outside plan')
        prerequisites(c,a.index>0);name=c['cells'][a.index]['name']
    else:name=a.mode
    dest=root/name;dest.mkdir(exist_ok=False);start=time.monotonic()
    ctx,stable,design,orders,factory=engine.context(c)
    if a.mode=='smoke':engine.smoke(c,dest,ctx,stable,design,orders,factory)
    elif a.mode=='baseline':engine.make_baseline(c,dest,ctx,stable,design,orders,factory)
    else:engine.branch_cell(c,dest,root/'baseline',c['cells'][a.index],ctx,stable,design,orders,factory)
    rt.config(a.out)
    if rt.read(dest/'result.json').get('quality_gate_passed') is not True:raise ValueError('Quality gate missing')
    rt.finish(dest,c,mode=a.mode,index=a.index,elapsed_seconds=time.monotonic()-start)
    print('COMPLETE '+str(dest),flush=True)


def archive(root,debug=False):
    root=Path(root);path=Path(str(root)+('.debug' if debug else '')+'.tar.gz')
    if path.exists():raise FileExistsError(path)
    with tarfile.open(path,'x:gz') as t:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.pt':t.add(p,arcname=str(Path(root.name)/p.relative_to(root)))
    print(path,flush=True)


def collect(a):
    c=rt.config(a.out);root=Path(c['out']);prerequisites(c,True);h=rt.sha(root/'config.json')
    for cell in c['cells']:rt.completed(root/cell['name'],h)
    report=root/'report'
    if report.exists():rt.completed(report,h)
    else:
        from report import collect as analyze
        analyze(c);rt.finish(report,c)
    archive(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='action',required=True)
    q=sp.add_parser('prepare');q.add_argument('--out',required=True);q.add_argument('--baseline-out',default=rt.BASELINE)
    q.add_argument('--seed',type=int,choices=[42,43,44],default=42);q.add_argument('--patient',type=int,choices=[807,2085],default=807)
    q.add_argument('--concurrency',type=int,choices=[1,2],default=1)
    for action in ['submit','submit-rest','gpu','collect','debug']:
        q=sp.add_parser(action);q.add_argument('--out',required=True)
        if action=='gpu':q.add_argument('--mode',choices=['smoke','baseline','cell'],required=True);q.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.action=='prepare':rt.prepare(a)
    elif a.action=='debug':archive(a.out,True)
    else:globals()[a.action.replace('-','_')](a)


if __name__=='__main__':main()
