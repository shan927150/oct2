"""Stable-v1-compatible native Adam continuation with immutable CPU snapshots."""
import copy
import hashlib
import random
from contextlib import contextmanager
import numpy as np
import torch


def clone_cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: clone_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [clone_cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(clone_cpu(v) for v in x)
    return copy.deepcopy(x)


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def digest(x):
    h=hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode()+b':')
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,np.ndarray): h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda k:repr(k)): visit(k);visit(v[k])
        elif isinstance(v,(tuple,list)):
            for q in v:visit(q)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
                cuda=clone_cpu(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])


@contextmanager
def preserved_rng():
    saved=rng()
    try:yield
    finally:restore_rng(saved)


def snapshot(model,ema,opt,epoch,step,recipe,total):
    return dict(raw=clone_cpu(model.state_dict()),ema=clone_cpu(ema.state_dict()),
        optimizer=clone_cpu(opt.state_dict()),epoch=epoch,step=step,total_steps=total,
        recipe=copy.deepcopy(recipe),rng=clone_cpu(rng()),parameter_names=[n for n,_ in model.named_parameters()],
        model_training=model.training,ema_training=ema.training)


def fresh(factory,recipe,total):
    model=factory()
    if any(isinstance(x,torch.nn.modules.batchnorm._BatchNorm) for x in model.modules()):raise ValueError('BatchNorm outside contract')
    for x in model.modules():
        if isinstance(x,torch.nn.Dropout):x.p=0.
    ema=copy.deepcopy(model)
    for p in ema.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=recipe['lr'],weight_decay=recipe['wd'])
    return snapshot(model,ema,opt,0,0,recipe,total)


def restore(s,factory):
    model=factory();model.load_state_dict(clone_cpu(s['raw']));model.train(s['model_training'])
    if [n for n,_ in model.named_parameters()]!=s['parameter_names']:raise ValueError('Parameter order changed')
    for x in model.modules():
        if isinstance(x,torch.nn.Dropout):x.p=0.
    ema=copy.deepcopy(model);ema.load_state_dict(clone_cpu(s['ema']));ema.train(s['ema_training'])
    for p in ema.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=s['recipe']['lr'],weight_decay=s['recipe']['wd'])
    # Never lend source tensors to the optimizer, even when device/dtype agree.
    opt.load_state_dict(copy.deepcopy(s['optimizer']));restore_rng(s['rng'])
    return model,ema,opt


def flat_state(state,names):return torch.cat([state[n].reshape(-1).double().cpu() for n in names])


def train(s,factory,X,y,orders,patient_rows,alpha,stable,stop=None,observe=None):
    """Global schedule never restarts; observations must preserve all RNG/state."""
    if not 0<=alpha<=1:raise ValueError('alpha outside [0,1]')
    recipe=s['recipe'];stop=len(orders) if stop is None else stop
    total=sum((len(o)+recipe['batch']-1)//recipe['batch'] for o in orders)
    if total!=s['total_steps'] or not s['epoch']<=stop<=len(orders):raise ValueError('Horizon mismatch')
    expected=sum((len(o)+recipe['batch']-1)//recipe['batch'] for o in orders[:s['epoch']])
    if s['step']!=expected:raise ValueError('Snapshot epoch/step mismatch')
    frozen=digest(s);model,ema,opt=restore(s,factory);step=s['step'];curve=[]
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    for epoch in range(s['epoch'],stop):
        model.train();order=orders[epoch];ce_sum=0.;lr_sum=0.;first_exposed=None
        for start in range(0,len(order),recipe['batch']):
            idx=order[start:start+recipe['batch']]
            rate=stable.learning_rate(step,total,recipe['lr'],recipe['warmup_frac'])
            for group in opt.param_groups:group['lr']=rate
            xb=torch.as_tensor(X[idx],device=device,dtype=dtype);yb=torch.as_tensor(y[idx],device=device,dtype=torch.long)
            mask=torch.as_tensor(np.isin(idx,patient_rows),device=device)
            if bool(mask.any()) and first_exposed is None:first_exposed=step
            opt.zero_grad(set_to_none=True)
            loss,_=stable.fixed_mask_loss(model(xb),yb,mask,alpha)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            loss.backward();opt.step()
            with torch.no_grad():
                for avg,p in zip(ema.parameters(),model.parameters()):avg.mul_(recipe['ema_decay']).add_(p,alpha=1-recipe['ema_decay'])
            ce_sum+=float(loss.detach())*len(idx)/len(order);lr_sum+=rate;step+=1
        row=dict(epoch=epoch+1,step=step,lr=rate,lr_sum=lr_sum,online_weighted_ce=ce_sum,
                 first_exposed_step_in_epoch=first_exposed)
        if observe:
            with preserved_rng():observe(model,ema,opt,epoch+1,step,row)
        curve.append(row)
    result=snapshot(model,ema,opt,stop,step,recipe,total)
    if digest(s)!=frozen:raise RuntimeError('Source snapshot mutated')
    return result,curve
