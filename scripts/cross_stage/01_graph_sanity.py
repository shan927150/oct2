#!/usr/bin/env python3
"""
01_graph_sanity.py

Verify that the two stages can actually be connected into ONE autograd graph:

    raw image  ->  shadow CNN  ->  softmax  ->  attack MLP  ->  membership CE loss

and that gradients really flow all the way back. This is the plumbing test for
Direction B. It does NOT need the real trained shadow weights (a randomly
initialised shadow is enough to test that the graph is wired), though you may
pass a real checkpoint with --shadow_ckpt.

Checks (each PASS/FAIL):
  1. prediction vector p has requires_grad = True (i.e. not detached)
  2. d loss / d(shadow params) is non-zero
  3. d loss / d(raw image) is non-zero
  4. DETACH control: feeding p.detach() into the attack model yields ZERO grad
     on shadow params (confirms the current pipeline's detach is what kills it)
  5. finite-difference agreement: autograd directional derivative along a random
     shadow-param direction matches (L(+eps) - L(-eps)) / (2 eps)

Dropout note: SmallCNN has Dropout(0.2). For a deterministic sensitivity test we
run the shadow in eval(). If you instead want the true stochastic-training
sensitivity, use --train_mode and fix the dropout RNG; do not mix the two.

Usage (repo root):
    python scripts/cross_stage/01_graph_sanity.py --n_images 8
    python scripts/cross_stage/01_graph_sanity.py --shadow_ckpt path/to/shadow.pt
"""
import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models import SmallCNN, AttackModel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("graph_sanity")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def end_to_end_loss(shadow, attack, x, m, detach=False):
    """image -> shadow -> softmax -> attack -> CE(membership). Returns (loss, p)."""
    logits = shadow(x)                      # (B, n_classes)
    p = F.softmax(logits, dim=1)            # prediction vector fed to attack model
    p_in = p.detach() if detach else p
    out = attack(p_in)                      # (B, 2) -> [out, in]
    loss = F.cross_entropy(out, m)
    return loss, p


def flat_grad(loss, params, retain=True):
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        flat.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--in_channels", type=int, default=1)   # OCT is grayscale
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--attack_hidden", type=int, default=64)
    ap.add_argument("--shadow_ckpt", default=None,
                    help="optional real shadow state_dict; else random init")
    ap.add_argument("--attack_ckpt", default=None,
                    help="optional real attack state_dict; else random init")
    ap.add_argument("--train_mode", action="store_true",
                    help="run shadow in train() with fixed dropout RNG (stochastic)")
    ap.add_argument("--fd_eps", type=float, default=1e-4)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="relative tolerance for finite-difference agreement")
    ap.add_argument("--float32", action="store_true",
                    help="use float32 (default float64 for a clean gradient-check)")
    ap.add_argument("--random_fd_dir", action="store_true",
                    help="probe a random direction instead of the gradient direction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dtype = torch.float32 if args.float32 else torch.float64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # models
    shadow = SmallCNN(args.in_channels, 128, args.n_classes).to(DEVICE).to(dtype)
    attack = AttackModel(args.n_classes, args.attack_hidden).to(DEVICE).to(dtype)
    if args.shadow_ckpt:
        sd = torch.load(args.shadow_ckpt, map_location=DEVICE)
        shadow.load_state_dict(sd.get("state_dict", sd))
        logger.info(f"loaded shadow ckpt {args.shadow_ckpt}")
    if args.attack_ckpt:
        sd = torch.load(args.attack_ckpt, map_location=DEVICE)
        attack.load_state_dict(sd.get("state_dict", sd))
        logger.info(f"loaded attack ckpt {args.attack_ckpt}")

    if args.train_mode:
        shadow.train(); attack.train()
        torch.manual_seed(args.seed)   # fix dropout draw for reproducibility
        logger.info("shadow in TRAIN mode (stochastic dropout, RNG fixed)")
    else:
        shadow.eval(); attack.eval()
        logger.info("shadow in EVAL mode (deterministic)")

    # synthetic-but-shaped inputs (values irrelevant to plumbing)
    x = torch.randn(args.n_images, args.in_channels, args.img_size, args.img_size,
                    device=DEVICE, dtype=dtype, requires_grad=True)
    m = torch.randint(0, 2, (args.n_images,), device=DEVICE)

    results = {}
    probe_seed = args.seed + 10000

    # --- connected graph ---
    # In train mode the analytic gradient and both finite-difference evaluations
    # must use the identical dropout mask.
    if args.train_mode:
        torch.manual_seed(probe_seed)
    loss, p = end_to_end_loss(shadow, attack, x, m, detach=False)
    results["1_p_requires_grad"] = bool(p.requires_grad)

    shadow_params = [q for q in shadow.parameters() if q.requires_grad]
    g_shadow = flat_grad(loss, shadow_params, retain=True)
    results["2_grad_shadow_nonzero"] = float(g_shadow.norm()) > 0
    logger.info(f"  ||d loss / d shadow_params|| = {float(g_shadow.norm()):.4e}")

    g_x = torch.autograd.grad(loss, x, retain_graph=True)[0]
    results["3_grad_image_nonzero"] = float(g_x.norm()) > 0
    logger.info(f"  ||d loss / d image||         = {float(g_x.norm()):.4e}")

    # --- detach control (should zero out shadow grad) ---
    if args.train_mode:
        torch.manual_seed(probe_seed)
    loss_d, _ = end_to_end_loss(shadow, attack, x, m, detach=True)
    g_shadow_d = flat_grad(loss_d, shadow_params, retain=False)
    results["4_detach_zeroes_shadow_grad"] = float(g_shadow_d.norm()) == 0.0
    logger.info(f"  ||shadow grad under detach|| = {float(g_shadow_d.norm()):.4e} "
                f"(expected 0)")

    # --- finite-difference agreement along a shadow-param direction ---
    # Default: probe the gradient direction itself, so dd = ||g|| is large and
    # well-conditioned. A random direction gives a tiny dd that drowns in
    # round-off (which is why we also default to float64).
    if args.random_fd_dir:
        torch.manual_seed(args.seed + 123)
        direction = torch.randn_like(g_shadow)
    else:
        direction = g_shadow.clone()
    direction = direction / direction.norm()
    analytic_dd = float((g_shadow * direction).sum())

    fd = _fd_directional(shadow, attack, x, m, shadow_params, direction,
                         args.fd_eps, args.train_mode, probe_seed)
    rel = abs(analytic_dd - fd) / (abs(analytic_dd) + 1e-12)
    results["5_finite_diff_agrees"] = rel < args.tol
    logger.info(f"  directional deriv: autograd={analytic_dd:.6e}  "
                f"finite-diff={fd:.6e}  rel_err={rel:.2e} (tol={args.tol})")

    # --- report ---
    logger.info("=" * 60)
    logger.info("GRAPH SANITY RESULTS")
    all_pass = True
    for k in sorted(results):
        status = "PASS" if results[k] else "FAIL"
        all_pass = all_pass and results[k]
        logger.info(f"  [{status}] {k}")
    logger.info("=" * 60)
    logger.info(f"OVERALL: {'PASS — graph is connected, ready for 02' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(2)


def _fd_directional(shadow, attack, x, m, params, direction, eps, train_mode, seed):
    """Central finite difference of loss along `direction` in flat param space."""
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    orig = [p.detach().clone() for p in params]

    def set_offset(sign):
        off = 0
        with torch.no_grad():
            for p, sh, n, o in zip(params, shapes, numels, orig):
                d = direction[off:off + n].reshape(sh)
                p.copy_(o + sign * eps * d)
                off += n

    def loss_at():
        if train_mode:
            torch.manual_seed(seed)   # identical dropout draw at both eval points
        with torch.no_grad():
            l, _ = end_to_end_loss(shadow, attack, x, m, detach=False)
        return float(l)

    set_offset(+1.0); lp = loss_at()
    set_offset(-1.0); lm = loss_at()
    with torch.no_grad():                       # restore
        for p, o in zip(params, orig):
            p.copy_(o)
    return (lp - lm) / (2 * eps)


if __name__ == "__main__":
    main()
