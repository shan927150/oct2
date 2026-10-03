"""Stdlib login actions, pinned inherited sources and completion artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PREFIX=str(HERE.relative_to(REPO))+'/'
BASE='c4880c9ea1a527e4db004b2199b48e21b6426bf7'
SCHEMA='OCT_late_deletion_v2'
BASELINE='/u/yli103/oct2-pathway2-runs/A/stable_training_v1_baseline_20260928T181445049797182Z'
BASELINE_HASH='322489b49f403ccd187cfea8d5c3cbeb02cd7e5e6d1808641e2fc6f8fad2fc18'


def read(p):return json.loads(Path(p).read_text())


def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify(items):
    for p,h in items.items():
        if sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))


def csv_write(p,rows):
    if not rows:raise ValueError('Empty table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def csv_read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True).strip()


def check_repo():
    if git('status','--porcelain','--untracked-files=all'):raise ValueError('Keep installed worktree clean')
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',BASE,'HEAD'],check=True)
    diff=git('diff','--name-status',BASE,'HEAD').splitlines()
    if not diff or any(not x.startswith('A\t'+PREFIX) for x in diff):raise ValueError('Only additive new experiment files allowed')
    return git('rev-parse','HEAD')


def completed(dest,h):
    dest=Path(dest);v=read(dest/'complete.json')
    if v.get('status')!='complete' or v.get('config_sha256')!=h:raise ValueError('Incomplete/mismatched '+str(dest))
    verify({str(dest/k):x for k,x in v['output_sha256'].items()});return v


def finish(dest,c,**info):
    dest=Path(dest)
    write(dest/'complete.json',dict(status='complete',config_sha256=sha(Path(c['out'])/'config.json'),
        output_sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},**info))


def config(out):
    out=Path(out).resolve();h=sha(out/'config.json')
    if (out/'config.sha256').read_text().strip()!=h:raise ValueError('Config changed')
    c=read(out/'config.json')
    if (c['schema'],c['repo'],c['out'])!=(SCHEMA,str(REPO),str(out)):raise ValueError('Wrong worktree/schema/output')
    if check_repo()!=c['git_commit']:raise ValueError('Prepared commit changed')
    verify(c['source_sha256']);verify(c['input_sha256']);return c


def disjoint(out,*paths):
    out=Path(out).resolve()
    for path in paths:
        p=Path(path).resolve()
        if out==p or out in p.parents or p in out.parents:raise ValueError('Output overlaps '+str(p))


def prepare(a):
    commit=check_repo();base=Path(a.baseline_out).resolve();out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(out)
    h=sha(base/'config.json');old=read(base/'config.json')
    if h!=BASELINE_HASH or (base/'config.sha256').read_text().strip()!=h:raise ValueError('Need reviewed stable baseline')
    report=read(base/'report/complete.json')
    if report.get('config_sha256')!=h or report.get('noop_passed') is not True:raise ValueError('Stable full-horizon replay gate absent')
    recipe=dict(lr=.001,warmup_frac=.1,ema_decay=.99,wd=1e-5,batch=128)
    if old['recipe']!=recipe or old['epochs']!=140 or a.seed not in old['seeds'] or a.patient not in old['patients']:
        raise ValueError('Unsupported recipe/seed/patient')
    disjoint(out,REPO,base,old['full'],old['data_dir'],old['audit'])
    inputs=dict(old['input_sha256']);verify(inputs)
    for p in [base/'config.json',base/'config.sha256',base/'report/complete.json']:
        inputs[str(p)]=sha(p)
    d=base/f'seed{a.seed}_baseline';v=completed(d,h)
    inputs[str(d/'complete.json')]=sha(d/'complete.json')
    inputs.update({str(d/k):x for k,x in v['output_sha256'].items()})
    # Pin all inherited loader/model/kernel files in THIS immutable c488 descendant.
    sources={str(p):sha(p) for folder in ['src','scripts/cross_stage','experiments/pathway2_lds_v1','experiments/pathway2_stable_v1',PREFIX]
             for p in sorted((REPO/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    c=dict(schema=SCHEMA,repo=str(REPO),git_commit=commit,base_commit=BASE,out=str(out),baseline_out=str(base),
        input_sha256=inputs,source_sha256=sources,full=old['full'],data_dir=old['data_dir'],
        dataset_fingerprint=old['dataset_fingerprint'],account=old['account'],seed=a.seed,patients=[a.patient],
        epochs=140,recipe=recipe,T0_grid=[0,60,100,130],alphas=[.01,.1,1.],window_epochs=10,
        batch=64,weight_decay=1e-5,concurrency=a.concurrency,
        cells=[dict(name=f'T0{t:03d}_p{a.patient}',T0=t,patient=a.patient,seed=a.seed) for t in [130,100,60,0]],
        design='Conditional fixed-mask late downweighting, original denominator and global stable schedule.',
        inference='Trajectory sensitivity pilot. No chaos confirmation, no Eq.56 validation, no full LOO or attack J.',
        first_difference_resolution='epoch; no step-level divergence claim',
        request_note='User requested repaired isolated late-deletion code and submission instructions on 2026-10-02.')
    out.mkdir(parents=True);(out/'logs').mkdir();write(out/'config.json',c)
    (out/'config.sha256').write_text(sha(out/'config.json')+'\n')
    print('PREPARED '+str(out),flush=True)
