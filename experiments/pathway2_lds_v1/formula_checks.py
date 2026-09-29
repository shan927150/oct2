"""Deterministic tests of the derivation, separate from empirical OCT IF accuracy.

Exact smooth two-stage minimizers test Eq.48/56/79, including signs. A two-stage
Adam trajectory test checks all moment/state paths and finite differences.
These are controlled implementation tests, not evidence of OCT stationarity.
"""
import argparse
import json
import numpy as np


def implicit_check():
    rng=np.random.default_rng(529)
    n,d,r,a=9,4,5,3
    X=rng.normal(size=(n,d)); y=rng.normal(size=n)
    C=rng.normal(size=(r,d)); A=rng.normal(size=(r,a)); target=rng.normal(size=a)
    H=X.T@X/n+.3*np.eye(d)
    HA=A.T@A/r+.2*np.eye(a)
    # Full-risk stationary point, smooth nonlinear interface, convex attack risk.
    def solve(eps):
        theta=np.linalg.solve(H+eps*np.outer(X[2],X[2])/n, X.T@y/n+eps*X[2]*y[2]/n)
        P=np.tanh(C@theta)
        phi=np.linalg.solve(HA,A.T@P/r)
        return theta,P,phi,.5*np.sum((phi-target)**2)
    theta,P,phi,Y=solve(0.)
    b=X[2]*(X[2]@theta-y[2])/n
    B=-A.T/r
    q=phi-target
    v=-B.T@np.linalg.solve(HA.T,q)
    h=C.T@((1-P**2)*v)
    analytic=-h@np.linalg.solve(H,b)
    eps=1e-4
    fd=(solve(eps)[3]-solve(-eps)[3])/(2*eps)
    error=abs(fd-analytic)/max(abs(analytic),1e-12)
    if error>1e-6:
        raise AssertionError((analytic,fd,error))
    # Patient deletion is negative upweight, therefore reverses this sign.
    deletion_fd=(solve(-eps)[3]-solve(eps)[3])/(2*eps)
    if not np.isclose(deletion_fd,-analytic,rtol=1e-6,atol=1e-10):
        raise AssertionError('Deletion sign failed')
    return dict(analytic=float(analytic),finite_difference=float(fd),relative_error=float(error),
                scope='Smooth stationary two-stage risk; no damping; exact solves')


def adam_map(params, moments, squares, step, grads, lr, beta1=.9,beta2=.999,eps=1e-8):
    """Differentiable Adam, coupled L2 is supplied as part of grads."""
    step=step+1
    moments=[beta1*m+(1-beta1)*g for m,g in zip(moments,grads)]
    squares=[beta2*v+(1-beta2)*g.square() for v,g in zip(squares,grads)]
    params=[p-lr*(m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+eps)
            for p,m,v in zip(params,moments,squares)]
    return params,moments,squares,step


def trajectory_check():
    import torch
    torch.set_default_dtype(torch.float64)
    X=torch.tensor([[.3,.8],[-.2,.5],[.9,-.4]])
    y=torch.tensor([.2,-.3,.7])
    query=torch.tensor([.4,-.6])
    def chain(epsilon):
        p=[torch.tensor([.1,-.2],requires_grad=True)]
        m=[torch.zeros_like(p[0])]; v=[torch.zeros_like(p[0])]; t=0
        for _ in range(4):
            residual=X@p[0]-y
            loss=(residual.square()*torch.stack([1+epsilon,epsilon*0+1,epsilon*0+1])).mean()/2+.01*p[0].square().sum()/2
            g=torch.autograd.grad(loss,p,create_graph=True)
            p,m,v,t=adam_map(p,m,v,t,g,.02)
        P=torch.sigmoid(X@p[0])
        attack=[torch.tensor([.15,-.25],requires_grad=True)]
        am=[torch.zeros_like(attack[0])]; av=[torch.zeros_like(attack[0])]; at=0
        for _ in range(3):
            pred=attack[0][0]*P+attack[0][1]
            loss=(pred-y).square().mean()/2
            ag=torch.autograd.grad(loss,attack,create_graph=True)
            attack,am,av,at=adam_map(attack,am,av,at,ag,.01)
        return ((attack[0]@query)-.2).square()/2
    e=torch.tensor(0.,requires_grad=True)
    analytic=float(torch.autograd.grad(chain(e),e)[0])
    rows=[]
    for epsilon in (1e-2,1e-3,1e-4):
        fd=float(((chain(torch.tensor(epsilon))-chain(torch.tensor(-epsilon)))/(2*epsilon)).detach())
        rows.append(dict(epsilon=epsilon,analytic=analytic,finite_difference=fd,relative_error=abs(fd-analytic)/max(abs(analytic),1e-12)))
    if min(r['relative_error'] for r in rows)>1e-5:
        raise AssertionError(rows)
    # Independent PyTorch Adam update reference, including nonzero moments.
    p=torch.tensor([.3,-.7],requires_grad=True)
    opt=torch.optim.Adam([p],lr=.02)
    pp=[p.detach().clone()]; mm=[torch.zeros_like(p)]; vv=[torch.zeros_like(p)]; tt=0
    for g in [torch.tensor([.1,.4]),torch.tensor([-.2,.3]),torch.tensor([.5,-.6])]:
        p.grad=g.clone(); opt.step()
        pp,mm,vv,tt=adam_map(pp,mm,vv,tt,[g],.02)
    if not torch.allclose(p,pp[0],rtol=1e-12,atol=1e-12):
        raise AssertionError('Adam update disagrees with PyTorch')
    return dict(finite_difference=rows,pytorch_update_matched=True,
                scope='Full differentiated two-stage toy Adam trajectory, not frozen moments')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--require-torch',action='store_true')
    a=p.parse_args()
    results=dict(implicit=implicit_check())
    if a.require_torch:
        results['trajectory']=trajectory_check()
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
