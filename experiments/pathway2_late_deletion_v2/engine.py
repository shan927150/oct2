"""Native stable-recipe baseline and shared-prefix continuations; no IF solves."""
from pathlib import Path
import sys
import numpy as np
import torch
import kernel as k
import metrics
import runtime_late as rt

sys.path.insert(0,str(rt.REPO/'experiments/pathway2_lds_v1'))
import core as old
from design import make_design
sys.path.pop(0)


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def save(p,x):
    if Path(p).exists():raise FileExistsError(p)
    torch.save(x,p)


def context(c):
    ctx,legacy,pilot,conv=old.rt.load_context(c,c['seed'])
    stable=old.module('late_inherited_stable',rt.REPO/'experiments/pathway2_stable_v1/stable_core.py')
    design=make_design(ctx['train'],ctx['test'],ctx['groups'],ctx['y'],8,.5,20260929)
    torch.use_deterministic_algorithms(True,warn_only=False)
    orders=pilot.make_epoch_orders(ctx['train'],c['epochs'],c['seed']+700000)
    with np.load(Path(c['full'])/f"stage1_order_seed{c['seed']}.npz") as z:
        n=min(len(orders),len(z['raw_index_order']))
        np.testing.assert_array_equal(orders[:n],z['raw_index_order'][:n])
    factory=lambda:old.initialize(ctx,pilot,c['seed'])
    return ctx,stable,design,orders,factory


def small_state(model,ema):return dict(raw=k.clone_cpu(model.state_dict()),ema=k.clone_cpu(ema.state_dict()))


def probability_file(c,dest,stem,state,ctx,design,factory):
    arrays=dict(raw_index=np.asarray(design['query_rows']),labels=ctx['y'][ctx['test']],weights=np.asarray(design['query_weights']))
    rows=ctx['test'];w=arrays['weights'];summary={}
    with k.preserved_rng(),torch.no_grad():
        for endpoint in ['raw','ema']:
            model=factory();model.load_state_dict(state[endpoint]);model=model.double().eval();par=next(model.parameters());ps=[];ces=[]
            for i in range(0,len(rows),c['batch']):
                idx=rows[i:i+c['batch']]
                x=torch.as_tensor(ctx['X'][idx],device=par.device,dtype=torch.float64)
                y=torch.as_tensor(ctx['y'][idx],device=par.device,dtype=torch.long);logits=model(x)
                ps.append(logits.softmax(1).cpu().numpy());ces.append(torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().numpy())
            arrays[endpoint]=np.concatenate(ps);arrays[endpoint+'_ce']=np.concatenate(ces)
            summary[endpoint]=metrics.targets(arrays[endpoint],arrays['labels'],w,arrays[endpoint+'_ce'])
    path=dest/(stem+'_query.npz')
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**arrays);return summary


def require_same(actual,expected,label):
    for name in ['raw','ema','optimizer','epoch','step','total_steps','recipe','rng','parameter_names','model_training','ema_training']:
        if not k.same(actual[name],expected[name]):raise ValueError(label+': state mismatch in '+name)


def make_baseline(c,dest,ctx,stable,design,orders,factory,compare_parent=True):
    total=sum((len(x)+c['recipe']['batch']-1)//c['recipe']['batch'] for x in orders)
    T=len(orders);grid=c['T0_grid'];wanted=sorted(set([0,T]+grid+[min(T,t+c['window_epochs']) for t in grid]))
    for name in ['checkpoints','history']: (dest/name).mkdir()
    start=k.fresh(factory,c['recipe'],total);save(dest/'checkpoints/epoch000.pt',start)
    save(dest/'history/epoch000.pt',{x:start[x] for x in ['raw','ema']})
    def observe(model,ema,opt,epoch,step,row):
        save(dest/f'history/epoch{epoch:03d}.pt',small_state(model,ema))
        if epoch in wanted:save(dest/f'checkpoints/epoch{epoch:03d}.pt',k.snapshot(model,ema,opt,epoch,step,c['recipe'],total))
        if epoch==1 or epoch%10==0 or epoch==T:print('BASELINE',epoch,'/',T,flush=True)
    final,curve=k.train(start,factory,ctx['X'],ctx['y'],orders,[],0.,stable,observe=observe)
    # End-of-epoch observation must not alter full training state.
    require_same(final,load(dest/f'checkpoints/epoch{T:03d}.pt'),'baseline snapshot')
    exact_parent=None
    if compare_parent:
        parent=load(Path(c['baseline_out'])/f"seed{c['seed']}_baseline/final.pt")
        exact_parent=all(k.same(final[x],parent[x]) for x in ['raw','ema','optimizer'])
        if not exact_parent:raise ValueError('Full-horizon stable baseline differs; preserve diagnostic output, do not run branches')
    query_epochs=sorted(set([T]+[min(T,t+c['window_epochs']) for t in grid]));summaries={}
    for epoch in query_epochs:
        summaries[str(epoch)]=probability_file(c,dest,f'epoch{epoch:03d}',load(dest/f'checkpoints/epoch{epoch:03d}.pt'),ctx,design,factory)
    rt.csv_write(dest/'curve.csv',curve);rt.write(dest/'design.json',design)
    rt.write(dest/'query_targets.json',summaries)
    rt.write(dest/'result.json',dict(quality_gate_passed=True,exact_parent_replay=exact_parent,total_steps=total,
        snapshot_epochs=wanted,query_epochs=query_epochs,parameter_names=final['parameter_names'],
        query_precision='float64 forward from native float32 endpoints',order_sha256=k.digest(orders)))
    return start,final


def branch_cell(c,dest,base,cell,ctx,stable,design,orders,factory):
    T=len(orders);t=cell['T0'];endpoints=sorted(set([min(T,t+c['window_epochs']),T]));pid=cell['patient']
    by_patient={p['patient']:p['rows'] for p in design['train_patients']}
    if pid not in by_patient:raise ValueError('Patient missing from frozen train split')
    source_path=base/f'checkpoints/epoch{t:03d}.pt';s=load(source_path);frozen=k.digest(s);file_hash=rt.sha(source_path)
    expected=load(base/f'checkpoints/epoch{T:03d}.pt')
    replay,_=k.train(s,factory,ctx['X'],ctx['y'],orders,[],0.,stable)
    require_same(replay,expected,'full-state replay')
    # Also verify a genuinely unexposed direction, cheaply, in smoke only.
    names=s['parameter_names'];distance_rows=[];query_summary={};states={};trace=[]
    for alpha in c['alphas']:
        tag=f'a{alpha:g}';save_epochs={};dist=[]
        def observe(model,ema,opt,epoch,step,row):
            state=small_state(model,ema);reference=load(base/f'history/epoch{epoch:03d}.pt')
            row=dict(row,alpha=alpha,T0=t,patient=pid)
            for label in ['raw','ema']:
                delta=k.flat_state(state[label],names)-k.flat_state(reference[label],names)
                row[label+'_distance']=float(delta.norm())
                norm=float(k.flat_state(reference[label],names).norm())
                row[label+'_relative_distance']=float(delta.norm())/norm if norm else None
            dist.append(row)
            if epoch in endpoints:
                snap=k.snapshot(model,ema,opt,epoch,step,c['recipe'],s['total_steps'])
                save(dest/f'{tag}_epoch{epoch:03d}.pt',snap);save_epochs[epoch]=snap
        final,curve=k.train(s,factory,ctx['X'],ctx['y'],orders,by_patient[pid],alpha,stable,observe=observe)
        if t==T:
            save_epochs[T]=final;save(dest/f'{tag}_epoch{T:03d}.pt',final)
        if k.digest(s)!=frozen or rt.sha(source_path)!=file_hash:raise ValueError('Snapshot changed between branches')
        if t==T:require_same(final,expected,'terminal no-op')
        for epoch in endpoints:
            state=save_epochs[epoch];states[(alpha,epoch)]=state
            query_summary[f'{tag}_epoch{epoch:03d}']=probability_file(c,dest,f'{tag}_epoch{epoch:03d}',state,ctx,design,factory)
        trace+=dist
        distance_rows.append(dict(alpha=alpha,first_different_epoch=next((r['epoch'] for r in dist if r['raw_distance']>0),None)))
        print('BRANCH',t,pid,alpha,'complete',flush=True)
    summary=[];scalar_rows=[];baseline_targets=rt.read(base/'query_targets.json')
    for epoch in endpoints:
        reference=load(base/f'checkpoints/epoch{epoch:03d}.pt')
        with np.load(base/f'epoch{epoch:03d}_query.npz') as z:refq={key:z[key] for key in z.files}
        for label in ['raw','ema']:
            theta=k.flat_state(reference[label],names).numpy();deltas={};qdelta={}
            for alpha in c['alphas']:
                deltas[alpha]=(k.flat_state(states[(alpha,epoch)][label],names).numpy()-theta)
                tag=f'a{alpha:g}_epoch{epoch:03d}'
                with np.load(dest/(tag+'_query.npz')) as z:
                    np.testing.assert_array_equal(z['raw_index'],refq['raw_index'])
                    qdelta[alpha]=z[label]-refq[label]
                vals=query_summary[tag][label];before=baseline_targets[str(epoch)][label]
                scalar_rows.append(dict(T0=t,patient=pid,seed=c['seed'],epoch=epoch,endpoint=label,alpha=alpha,
                    query_TV=float(refq['weights']@(np.abs(qdelta[alpha]).sum(1)/2)),
                    **{key:vals[key] for key in vals},**{'delta_'+key:vals[key]-before[key] for key in vals}))
            for lo,hi in [(.01,.1),(.1,1.)]:
                for space,values in [('theta',deltas),('query_probability',{a:x*np.sqrt(refq['weights'])[:,None] for a,x in qdelta.items()})]:
                    g=metrics.geometry(values[lo],values[hi],lo/hi)
                    # Quantization flag uses actual snapshot tensor dtype and componentwise ULP scales.
                    resolution=float(np.linalg.norm(np.spacing(theta.astype(np.float32)).astype(float))) if space=='theta' else None
                    g.update(T0=t,seed=c['seed'],patient=pid,epoch=epoch,remaining_epochs=T-t,
                        comparison='terminal' if epoch==T else 'fixed_window',window_epochs=epoch-t,endpoint=label,
                        space=space,alpha_small=lo,alpha_full=hi,reference_norm=float(np.linalg.norm(theta)) if space=='theta' else None,
                        full_relative_shift=float(np.linalg.norm(deltas[hi])/np.linalg.norm(theta)) if space=='theta' and np.linalg.norm(theta) else None,
                        parameter_ulp_norm=resolution,small_over_parameter_ulp=g['norm_small']/resolution if resolution else None)
                    summary.append(g)
    rt.csv_write(dest/'geometry.csv',summary);rt.csv_write(dest/'targets.csv',scalar_rows)
    if trace:rt.csv_write(dest/'trajectory.csv',trace)
    rt.write(dest/'result.json',dict(quality_gate_passed=True,full_state_replay=True,source_snapshot_unchanged=True,
        snapshot_file_sha256=file_hash,snapshot_state_sha256=frozen,cell=cell,first_difference=distance_rows,
        replay_fields=['raw','ema','optimizer','epoch','step','total_steps','recipe','rng','parameter_names','training_flags'],
        cumulative_lr_after_branch=sum(float(x['lr_sum']) for x in trace if x['alpha']==c['alphas'][0]),
        scope='Conditional response and dose geometry only; no static IF, J, or chaos claim.'))


def smoke(c,dest,ctx,stable,design,orders,factory):
    c=dict(c,epochs=4,T0_grid=[0,2,4],window_epochs=1)
    # Use 4-epoch orders with its own 4-epoch schedule; formal jobs retain 140.
    orders=orders[:4];base=dest/'baseline';base.mkdir()
    start,final=make_baseline(c,base,ctx,stable,design,orders,factory,False)
    inherited=stable.train(factory(),ctx['X'],ctx['y'],orders,[],0.,**c['recipe'])
    if not all(k.same(final[label],m.state_dict()) for label,m in zip(['raw','ema','optimizer'],inherited[:3])):
        raise ValueError('Inherited stable kernel parity failed')
    s=load(base/'checkpoints/epoch002.pt');frozen=k.digest(s)
    check={}
    for alpha in [.1,1.]:
        state,_=k.train(s,factory,ctx['X'],ctx['y'],orders,ctx['patient_rows'][c['patients'][0]],alpha,stable)
        check[alpha]=state
    for alpha in [1.,.1]:
        state,_=k.train(s,factory,ctx['X'],ctx['y'],orders,ctx['patient_rows'][c['patients'][0]],alpha,stable)
        require_same(state,check[alpha],'repeated/reversed branch')
    if k.digest(s)!=frozen:raise ValueError('Smoke snapshot mutation')
    unexposed,_=k.train(s,factory,ctx['X'],ctx['y'],orders,[],1.,stable)
    require_same(unexposed,final,'absent direction')
    for t in c['T0_grid']:
        cell=dict(T0=t,patient=c['patients'][0],seed=c['seed']);d=dest/f'T0{t:03d}';d.mkdir()
        branch_cell(c,d,base,cell,ctx,stable,design,orders,factory)
    rt.write(dest/'result.json',dict(quality_gate_passed=True,epochs=4,all_T0_full_state_replay=True,
        repeated_and_reversed_branches_exact=True,source_snapshots_immutable=True,stable_kernel_exact=True,
        terminal_noop_exact=True,absent_direction_exact=True))
