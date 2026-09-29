"""Optional direct test of a recentered proximal estimand, not cold-start LOO.

F_tilde(theta, e) = F(theta) - g0.T(theta-theta0)
                 + gamma/2 ||theta-theta0||^2 + e CE_patient(theta)/N.
theta0 is stationary at e=0 by construction. Its local derivative, when the
Hessian is nonsingular and the local solution branch exists, is -(H+gamma I)^-1 b.
This controlled estimand makes the meaning of damping explicit.
"""
from pathlib import Path
import numpy as np
import runtime as rt
from core import context,baseline,initialize,predict,gradient
from design import output_and_v


def run(c,dest):
    import torch
    seed=42
    ctx,pilot,conv,score,stable,d=context(c,seed)
    ref=Path(c['out'])/'score_seed42'
    base=baseline(c,seed)
    reference=initialize(ctx,pilot,seed,base['raw'],double=True)
    theta0=score.flatten([p.detach() for p in reference.parameters()]).clone()
    with np.load(ref/'vectors.npz') as z:
        g0=torch.as_tensor(z['g'],device=pilot.DEVICE,dtype=torch.float64)
        h=torch.as_tensor(z['h'],device=pilot.DEVICE,dtype=torch.float64)
    with np.load(ref/'interface.npz') as z:
        Y0,_=output_and_v(z['raw'][len(ctx['train']):],ctx['y'][ctx['test']],np.asarray(d['query_weights']))
    import csv
    with (ref/'scores.csv').open() as f:
        tau={int(r['patient']):float(r['tau_upweight']) for r in csv.DictReader(f)}
    rows=[]
    for patient in (807,2085):
        raw=next(p['rows'] for p in d['train_patients'] if p['patient']==patient)
        b=gradient(reference,ctx,raw,len(ctx['train']),score,c['batch'])
        bnorm=float(b.norm())
        for epsilon in (-.01,-.001,.001,.01):
            model=initialize(ctx,pilot,seed,base['raw'],double=True)
            params=list(model.parameters())
            opt=torch.optim.LBFGS(params,lr=1.,max_iter=1,history_size=20,line_search_fn='strong_wolfe',
                                 tolerance_grad=0.,tolerance_change=0.)
            calls=0
            def evaluate():
                nonlocal calls
                calls+=1
                g=torch.zeros_like(theta0);obj=0.
                idx=ctx['train'];N=len(idx)
                for start in range(0,N,c['batch']):
                    rr=idx[start:start+c['batch']]
                    x=torch.as_tensor(ctx['X'][rr],device=pilot.DEVICE,dtype=torch.float64)
                    y=torch.as_tensor(ctx['y'][rr],device=pilot.DEVICE,dtype=torch.long)
                    weights=torch.as_tensor(1+epsilon*np.isin(rr,raw),device=pilot.DEVICE,dtype=torch.float64)
                    loss=(torch.nn.functional.cross_entropy(model(x),y,reduction='none')*weights).sum()/N
                    g+=score.flatten(torch.autograd.grad(loss,params))
                    obj+=float(loss.detach())
                theta=score.flatten([p.detach() for p in params]);delta=theta-theta0
                obj+=float(c['weight_decay']/2*(theta@theta)-g0@delta+c['gamma']/2*(delta@delta))
                g+=c['weight_decay']*theta-g0+c['gamma']*delta
                if not np.isfinite(obj) or not torch.isfinite(g).all():
                    raise FloatingPointError('Nonfinite recentered objective')
                return obj,g
            def closure():
                value,g=evaluate()
                for p,gi in zip(params,score.unflatten(g,params)):
                    p.grad=gi.clone()
                return params[0].new_tensor(value)
            trajectory=[]
            converged=False
            for iteration in range(41):
                value,g=evaluate()
                residual=float(g.norm())/max(abs(epsilon)*bnorm,1e-30)
                trajectory.append(dict(iteration=iteration,objective=value,scaled_residual=residual,closures=calls))
                if residual<=.001:
                    converged=True;break
                if iteration==40 or calls>=300:
                    break
                opt.step(closure)
            P=predict(model,ctx['X'],ctx['test'])
            Y,_=output_and_v(P,ctx['y'][ctx['test']],np.asarray(d['query_weights']))
            delta=score.flatten([p.detach() for p in params])-theta0
            row=dict(patient=patient,epsilon=epsilon,delta_Y=Y-Y0,
                     predicted_delta_Y=epsilon*tau[patient],L2_true_theta=float(h@delta),
                     relative_theta_shift=float(delta.norm()/theta0.norm()),
                     scaled_stationarity_residual=residual,qualified=converged,closures=calls)
            rows.append(row)
            rt.csv_write(dest/f'trajectory_p{patient}_e{epsilon:g}.csv',trajectory)
            print(row,flush=True)
    finite=[]
    for patient in (807,2085):
        for eps in (.01,.001):
            plus=next(r for r in rows if r['patient']==patient and r['epsilon']==eps)
            minus=next(r for r in rows if r['patient']==patient and r['epsilon']==-eps)
            fd=(plus['delta_Y']-minus['delta_Y'])/(2*eps)
            finite.append(dict(patient=patient,epsilon=eps,finite_difference=fd,formula=tau[patient],
                relative_error=abs(fd-tau[patient])/max(abs(tau[patient]),1e-30),
                qualified=plus['qualified'] and minus['qualified']))
    rt.csv_write(dest/'local_rows.csv',rows)
    rt.csv_write(dest/'finite_difference.csv',finite)
    rt.write(dest/'result.json',dict(estimand='Recentered proximal full-data objective, not original cold-start retraining',
        reference_seed=seed,gamma=c['gamma'],rows=rows,finite_difference=finite,
        note='Only qualified +/- pairs support a numerical derivative comparison. Positive Lanczos estimate is not an SPD proof.'))
