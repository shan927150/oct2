"""OCT raw-endpoint scores and truth. No edits to the inherited stable kernel."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import runtime as rt
from design import array_sha, make_design, intervention, output_and_v


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def context(c, seed):
    ctx, legacy, pilot, conv = rt.load_context(c, seed)
    score = module('lds_score_ladder', rt.REPO/'scripts/cross_stage/07_cross_stage_score_ladder.py')
    stable = module('lds_stable_core', rt.REPO/'experiments/pathway2_stable_v1/stable_core.py')
    d = make_design(ctx['train'], ctx['test'], ctx['groups'], ctx['y'], c['m'], c['retain'], c['subset_seed'])
    return ctx, pilot, conv, score, stable, d


def initialize(ctx, pilot, seed, state=None, double=False):
    import torch
    pilot.seed_everything(seed, True)
    m = pilot.build_model('cnn', ctx['X'].shape[1], 128, int(ctx['y'].max())+1)
    for layer in m.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.
    if state is not None:
        m.load_state_dict(state)
    return m.double().eval() if double else m


def predict(model, X, rows, batch=64):
    import torch
    model.eval()
    param = next(model.parameters())
    out = []
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            x = torch.as_tensor(X[rows[start:start+batch]],device=param.device,dtype=param.dtype)
            out.append(model(x).softmax(1).cpu().numpy())
    return np.concatenate(out)


def baseline(c, seed):
    import torch
    return torch.load(Path(c['baseline_out'])/f'seed{seed}_baseline/final.pt',map_location='cpu',weights_only=False)


def gradient(model, ctx, rows, denominator, score, batch):
    import torch
    params = list(model.parameters())
    out = torch.zeros(sum(p.numel() for p in params),dtype=params[0].dtype,device=params[0].device)
    for i in range(0,len(rows),batch):
        idx = rows[i:i+batch]
        x = torch.as_tensor(ctx['X'][idx],dtype=params[0].dtype,device=params[0].device)
        y = torch.as_tensor(ctx['y'][idx],dtype=torch.long,device=params[0].device)
        loss = torch.nn.functional.cross_entropy(model(x),y,reduction='sum')/denominator
        out += score.flatten(torch.autograd.grad(loss,params))
    return out


def fd_chain(model, ctx, d, h, direction, score):
    """Eq.48 chain rule at fixed theta, independent of stationarity/CG quality."""
    import torch
    from torch.func import functional_call
    names = [n for n,_ in model.named_parameters()]
    params = list(model.parameters())
    direction = direction/direction.norm()
    pieces = score.unflatten(direction,params)
    qrows = np.asarray(d['query_rows'])
    weights = np.asarray(d['query_weights'])
    expected = float(h@direction)
    def output(eps):
        state = {n:p+eps*v for n,p,v in zip(names,params,pieces)}
        total = 0.
        with torch.no_grad():
            for i in range(0,len(qrows),64):
                rows = qrows[i:i+64]
                x = torch.as_tensor(ctx['X'][rows],device=params[0].device,dtype=params[0].dtype)
                y = torch.as_tensor(ctx['y'][rows],device=params[0].device,dtype=torch.long)
                w = torch.as_tensor(weights[i:i+64],device=params[0].device,dtype=params[0].dtype)
                total += float((torch.nn.functional.cross_entropy(functional_call(model,state,(x,)),y,reduction='none')*w).sum())
        return total
    rows = []
    for eps in (1e-2,1e-3,1e-4,1e-5):
        fd = (output(eps)-output(-eps))/(2*eps)
        rows.append(dict(epsilon=eps, finite_difference=fd, analytic=expected,
                         absolute_error=abs(fd-expected),relative_error=abs(fd-expected)/max(abs(expected),1e-12)))
    return rows


def score_seed(c, dest, seed, smoke=False):
    import torch
    ctx,pilot,conv,score,stable,d = context(c,seed)
    rt.write(dest/'design.json',d)
    base = baseline(c,seed)
    model = initialize(ctx,pilot,seed,base['raw'],double=True)
    P = predict(model,ctx['X'],ctx['interface'])
    q = P[len(ctx['train']):]
    y = ctx['y'][ctx['test']]
    Y,v = output_and_v(q,y,np.asarray(d['query_weights']))
    h = score.shadow_h(model,ctx['X'][ctx['test']],v,pilot.DEVICE,c['batch'])
    g = score.shadow_loss_grad(model,ctx['X'],ctx['y'],ctx['train'],pilot.DEVICE,c['weight_decay'],c['batch'])
    # Stored fp32 predictions are a provenance check; derivatives use fp64 throughout.
    with np.load(Path(c['baseline_out'])/f'seed{seed}_baseline/interface.npz') as z:
        if not np.array_equal(z['raw_index'],ctx['interface']):
            raise ValueError('Baseline interface row mismatch')
        probability_precision_difference = float(np.abs(z['raw'].astype(float)-P).max())
        if probability_precision_difference > 1e-4:
            raise ValueError('FP64 baseline predictions differ by more than 1e-4')
    selected = next(p for p in d['train_patients'] if p['patient']==807)
    b807 = gradient(model,ctx,selected['rows'],len(ctx['train']),score,c['batch'])
    fd = fd_chain(model,ctx,d,h,b807,score)
    rt.csv_write(dest/'chain_fd.csv',fd)
    if not any(r['absolute_error'] <= 1e-7+1e-3*abs(r['analytic']) for r in fd):
        raise ValueError('Chain-rule FD check failed; inspect chain_fd.csv')
    if smoke:
        # Two-epoch replay on an actual patient subset, including fixed-mask or filtering.
        removed,_ = intervention(d,0,c['mode'],c['alpha'],len(ctx['train']))
        orders = pilot.make_epoch_orders(ctx['train'],2,seed+700000)
        if c['mode']=='subset':
            orders=[o[~np.isin(o,removed)] for o in orders]
            removed=[]
        m,e,opt,_=stable.train(initialize(ctx,pilot,seed),ctx['X'],ctx['y'],orders,removed,c['alpha'],**c['recipe'])
        m2,e2,opt2,_=stable.train(initialize(ctx,pilot,seed),ctx['X'],ctx['y'],orders,removed,c['alpha'],**c['recipe'])
        import stage1_diagnostic_core as dc
        if not all(dc.exact_tree(a,b) for a,b in [(m.state_dict(),m2.state_dict()),(e.state_dict(),e2.state_dict()),(opt.state_dict(),opt2.state_dict())]):
            raise ValueError('Smoke deterministic replay failed')
        rt.write(dest/'result.json',dict(seed=seed,smoke=True,chain_fd_passed=True,exact_replay=True))
        return
    hvp=score.make_hvp(model,ctx['X'],ctx['y'],ctx['train'],pilot.DEVICE,c['weight_decay'],c['gamma'],c['batch'])
    ev=score.lanczos_extreme_eigs(hvp,h.numel(),c['lanczos_steps'],pilot.DEVICE,h.dtype,20260929)
    cg=score.conjugate_gradient(hvp,h,c['cg_steps'],c['cg_tol'])
    w=cg.pop('x')
    qualified=score.cg_reliable(cg,ev['lambda_min_est']>0,c['cg_fail_tol'])
    # A positive Lanczos estimate is a diagnostic, not a proof of positive definiteness.
    # Never increase gamma using subset outcome correlations.
    rows=[]
    for patient in d['train_patients']:
        b=gradient(model,ctx,patient['rows'],len(ctx['train']),score,c['batch'])
        rows.append(dict(seed=seed,patient=patient['patient'],n_images=len(patient['rows']),
            tau_upweight=-float(w@b),frozen_h_upweight=-float(h@b),b_norm=float(b.norm()),
            solver_qualified=qualified))
    # Same-dtype scalar consistency: h=-sum of weights/P gradients of P.
    np.savez_compressed(dest/'vectors.npz',h=h.detach().cpu().numpy(),g=g.detach().cpu().numpy())
    np.savez_compressed(dest/'interface.npz',raw=P,raw_index=ctx['interface'],classes=ctx['y'][ctx['interface']])
    rt.csv_write(dest/'scores.csv',rows)
    rt.write(dest/'result.json',dict(seed=seed,Y=Y,gradient_norm=float(g.norm()),gamma=c['gamma'],
        cg=cg,lanczos=ev,solver_qualified=qualified,chain_fd_passed=True,
        baseline_fp32_fp64_probability_max_difference=probability_precision_difference,
        stationarity_assumption_satisfied=False,
        note='Stationarity is not established. This is a damped endpoint IF approximation, not proof of Eq.56.'))


def truth_cell(c,dest,cell):
    import torch
    ctx,pilot,conv,score,stable,d=context(c,cell['seed'])
    ref=Path(c['out'])/f"score_seed{cell['seed']}"
    if rt.read(ref/'design.json') != d:
        raise ValueError('Patient design changed')
    noop=cell['subset'] is None
    removed=np.array([],dtype=np.int64) if noop else intervention(d,cell['subset'],c['mode'],c['alpha'],len(ctx['train']))[0]
    orders=pilot.make_epoch_orders(ctx['train'],c['epochs'],cell['seed']+700000)
    saved=np.load(Path(c['full'])/f"stage1_order_seed{cell['seed']}.npz")['raw_index_order']
    if not np.array_equal(orders[:len(saved)],saved[:len(orders)]):
        raise ValueError('Baseline order prefix mismatch')
    if c['mode']=='subset' and not noop:
        orders=[o[~np.isin(o,removed)] for o in orders]
        removed=np.array([],dtype=np.int64)
    m,ema,opt,curve=stable.train(initialize(ctx,pilot,cell['seed']),ctx['X'],ctx['y'],orders,removed,c['alpha'],**c['recipe'])
    base=baseline(c,cell['seed'])
    if noop:
        import stage1_diagnostic_core as dc
        for a,b in [(m.state_dict(),base['raw']),(ema.state_dict(),base['ema']),(opt.state_dict(),base['optimizer'])]:
            # exact_tree normalizes tensor locations to CPU in legacy helper.
            if not dc.exact_tree(a,b):
                raise ValueError('Full-horizon no-op differs from frozen stable baseline')
    torch.save(dict(raw={k:v.detach().cpu() for k,v in m.state_dict().items()},
                    ema={k:v.detach().cpu() for k,v in ema.state_dict().items()},cell=cell),dest/'final.pt')
    delta=torch.cat([(p.detach().cpu().double()-base['raw'][n].double()).reshape(-1) for n,p in m.named_parameters()]).numpy()
    m=m.double().eval()
    P=predict(m,ctx['X'],ctx['interface'])
    q=P[len(ctx['train']):]
    Y,_=output_and_v(q,ctx['y'][ctx['test']],np.asarray(d['query_weights']))
    with np.load(ref/'interface.npz') as z:
        P0=z['raw']
    Y0,v=output_and_v(P0[len(ctx['train']):],ctx['y'][ctx['test']],np.asarray(d['query_weights']))
    with np.load(ref/'vectors.npz') as z:
        L2=float(z['h']@delta)
    L1=float((v*(q-P0[len(ctx['train']):])).sum())
    rt.csv_write(dest/'curve.csv',curve)
    np.savez_compressed(dest/'interface.npz',raw=P,raw_index=ctx['interface'],classes=ctx['y'][ctx['interface']])
    rt.write(dest/'result.json',dict(cell=cell,Y=Y,Y0=Y0,delta_Y=Y-Y0,L1_true_P=L1,L2_true_theta=L2,
        delta_theta_norm=float(np.linalg.norm(delta)),full_horizon_noop_exact=True if noop else None,
        optimizer_steps=curve[-1]['step'],mode=c['mode']))
