"""Mentor warmup/cosine + EMA recipe within the fixed-batch intervention design."""
import math


def learning_rate(step, total, base=.001, warmup_frac=.1):
    if total < 1 or not 0 <= step < total or not 0 <= warmup_frac < 1 or base <= 0:
        raise ValueError('Invalid schedule coordinate')
    warm = int(warmup_frac * total)
    if step < warm:
        return base * (step + 1) / warm
    return base * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def fixed_mask_loss(logits, labels, removed_mask, alpha):
    import torch.nn.functional as F
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    per = F.cross_entropy(logits, labels, reduction='none')
    weights = 1 - alpha * removed_mask.to(per.dtype)
    return (per * weights).sum() / per.numel(), per


def train(model, X, y, orders, patient_rows, alpha, *, batch=128,
          lr=.001, warmup_frac=.1, ema_decay=.99, wd=1e-5,
          observe=None, save=None, save_epochs=()):
    import copy
    import numpy as np
    import torch
    # Keep the original module layout and deterministic pool. Dropout p=0
    # implements the mentor's Identity choice without changing parameter keys.
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('BatchNorm is outside the OCT contract')
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.
    if not 0 < ema_decay < 1:
        raise ValueError('EMA decay must be in (0,1)')
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = next(model.parameters()).device
    total = sum((len(o) + batch - 1) // batch for o in orders)
    step = 0
    curve = []
    for epoch, order in enumerate(orders, 1):
        previous = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double().clone()
        model.train()
        weighted_ce = plain_ce = 0.
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            rate = learning_rate(step, total, lr, warmup_frac)
            for g in opt.param_groups:
                g['lr'] = rate
            xb = torch.as_tensor(X[idx], device=device, dtype=next(model.parameters()).dtype)
            yb = torch.as_tensor(y[idx], device=device, dtype=torch.long)
            mask = torch.as_tensor(np.isin(idx, patient_rows), device=device)
            opt.zero_grad(set_to_none=True)
            loss, per = fixed_mask_loss(model(xb), yb, mask, alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError('Training loss is non-finite')
            loss.backward()
            opt.step()
            with torch.no_grad():
                for avg, p in zip(ema.parameters(), model.parameters()):
                    avg.mul_(ema_decay).add_(p, alpha=1-ema_decay)
            weighted_ce += float(loss.detach()) * len(idx) / len(order)
            plain_ce += float(per.detach().sum()) / len(order)
            step += 1
        theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double()
        row = dict(epoch=epoch, step=step, lr=rate, online_train_ce=weighted_ce,
                   online_unweighted_ce=plain_ce,
                   relative_epoch_update=float(torch.linalg.vector_norm(theta-previous) /
                                               torch.linalg.vector_norm(previous)))
        if observe:
            row.update(observe(model, ema, epoch))
        curve.append(row)
        if save and epoch in save_epochs:
            save(model, ema, opt, epoch, step)
        if epoch == 1 or epoch % 10 == 0 or epoch == len(orders):
            print(f'epoch={epoch}/{len(orders)} lr={rate:.5g} CE={weighted_ce:.6g}', flush=True)
    return model, ema, opt, curve
