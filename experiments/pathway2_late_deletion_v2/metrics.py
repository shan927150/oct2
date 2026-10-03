"""Vector geometry and fixed patient/class macro query metrics."""
import numpy as np


def geometry(small,full,ratio=.1):
    x=np.asarray(small,dtype=np.float64).reshape(-1);y=np.asarray(full,dtype=np.float64).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid displacement')
    nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y));den=ratio*ny
    # Undefined zero response stays NA; it is never coded as perfect linearity.
    return dict(norm_small=nx,norm_full=ny,R=nx/den if den else None,
        cosine=float(x@y/(nx*ny)) if nx and ny else None,
        relative_linear_error=float(np.linalg.norm(x-ratio*y)/den) if den else None,
        absolute_linear_error=float(np.linalg.norm(x-ratio*y)),denominator=den,
        zero_response=not bool(nx and ny))


def targets(p,y,w,ce):
    p=np.asarray(p,float);y=np.asarray(y);w=np.asarray(w,float);ce=np.asarray(ce,float)
    if p.shape[0]!=len(y) or w.shape!=y.shape or ce.shape!=y.shape:raise ValueError('Query shape mismatch')
    if not all(np.isfinite(v).all() for v in [p,w,ce]) or np.any(p<0) or np.any(p>1):raise ValueError('Nonfinite output')
    if not np.allclose(p.sum(1),1,atol=1e-10,rtol=0) or not np.isclose(w.sum(),1):raise ValueError('Bad query weights/probabilities')
    one=np.eye(p.shape[1])[y]
    return dict(ce=float(w@ce),brier=float(w@np.sum((p-one)**2,1)),
                accuracy=float(w@(p.argmax(1)==y)),oneminusp=float(w@(1-p[np.arange(len(y)),y])))
