#!/usr/bin/env python3
"""
02_pair_separation_test.py  —  the go/no-go viability test for Direction B.

Question: the 804 pairs found earlier have near-identical prediction-level TracIn
scores but clearly different LOO delta_loss. The attack-only score cannot tell
them apart. Can a FROZEN cross-stage score, which lets gradients flow back
through the shadow model to the raw image, separate them in the direction that
matches the LOO effect?

Frozen cross-stage score for attack-train row i (design doc, Pilot 2):

    C_i = < d J_i / d theta_s ,  g1_i >

    x_i   = raw image that produced row i
    p_i   = softmax(shadow_s(x_i))                 (shadow_s frozen)
    J_i   = CE(attack_c(p_i), m_i)                 downstream membership loss
    g1_i  = d/d theta_s  CE(shadow_s(x_i), y_i)    shadow classification update dir (SGD)

Interpretation: does the shadow's own classification update on image i point in a
direction that also makes i's downstream membership easier to detect?

For each pair (A, B):  dC = C_A - C_B,  dL = dloss_A - dloss_B.
We report:
  - pair concordance      mean 1[sign(dC) == sign(dL)]        (0.5 = no signal)
  - within-cell Spearman  corr(C, dloss) over rows in pairs
  - partial Spearman controlling p_true
  - incremental R^2 of C over baseline [membership, shadow_id, p_true]
This is a VIABILITY test, not a causal proof: the original single-point LOO is
itself below the retraining noise floor, so treat a positive result as
"worth continuing", not "validated".

GATE (design doc): if concordance ~ 0.5 AND within-cell corr ~ 0 AND incremental
R^2 ~ 0, pause the pivot.

IMPORTANT — shadow checkpoints. The base pipeline does NOT save shadow weights,
so to be exact you must regenerate attack data with shadow-checkpoint saving and
recompute scores/LOO/pairs on that run. Pass those with --shadow_ckpt_dir.
As an APPROXIMATE fallback, --build_shadow_ckpts retrains the needed shadows from
the split (won't bit-reproduce the original vectors; use only for a rough signal).

Usage (repo root):
    python scripts/cross_stage/02_pair_separation_test.py \
        --artifacts   ./results/cross_stage/artifacts \
        --pairs_csv   ./results/oct_cluster_trend_sgd/same_score_different_delta_pairs.csv \
        --scores_dir  ./results/oct_mia_tracin_sgd/tracin \
        --attack_ckpt_dir ./results/oct_mia_tracin_sgd/tracin \
        --attack_ckpt_pattern "sgd_checkpoints_class{c}.pt" \
        --shadow_ckpt_dir ./results/cross_stage/shadow_ckpts \
        --out_dir     ./results/cross_stage/pair_sep
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
import common  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pair_sep")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# scoring core (unit-testable: no file IO)
# ---------------------------------------------------------------------------

def frozen_cross_stage_score(shadow, attack, x_i, y_i, m_i):
    """C_i = <d J_i/d theta_s, d/d theta_s CE(shadow(x_i), y_i)> for a single image.

    x_i: (1, C, H, W) tensor. y_i, m_i: python ints.
    Returns (C_i, p_i_numpy).
    """
    shadow.eval(); attack.eval()
    params = [q for q in shadow.parameters() if q.requires_grad]

    logits = shadow(x_i)                              # (1, n_cls)
    # downstream membership loss J_i (through frozen attack)
    p = F.softmax(logits, dim=1)
    j = F.cross_entropy(attack(p), torch.tensor([m_i], device=x_i.device))
    g_down = torch.autograd.grad(j, params, retain_graph=True, allow_unused=True)
    # shadow classification update direction g1_i (SGD raw gradient)
    cls = F.cross_entropy(logits, torch.tensor([y_i], device=x_i.device))
    g_cls = torch.autograd.grad(cls, params, allow_unused=True)

    dot = 0.0
    for a, b, prm in zip(g_down, g_cls, params):
        a = a if a is not None else torch.zeros_like(prm)
        b = b if b is not None else torch.zeros_like(prm)
        dot += float((a * b).sum())
    return dot, F.softmax(logits, dim=1).detach().cpu().numpy().ravel()


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    """Tie-aware Spearman correlation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return _pearson(rankdata(a, method="average"), rankdata(b, method="average"))


def partial_spearman(x, y, z):
    """Partial Spearman via Pearson correlation of rank residuals."""
    x = rankdata(np.asarray(x, float), method="average")
    y = rankdata(np.asarray(y, float), method="average")
    z = rankdata(np.asarray(z, float), method="average")

    def resid(t):
        A = np.column_stack([z, np.ones_like(z)])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        return t - A @ coef

    return _pearson(resid(x), resid(y))


def _cell_key(c, m, s):
    return f"c{int(c)}_m{int(m)}_s{int(s)}"


def pooled_within_cell_spearman(C, dloss, oct_class, membership, shadow_id,
                                p_true=None, min_cell_n=5):
    """Pooled rank association after ranking and centering inside each cell.

    Cells are OCT class x membership x shadow.  This is the quantity the design
    calls *within-cell Spearman*; a global correlation is not an acceptable
    substitute because membership/shadow bands can create an artificial signal.
    If ``p_true`` is supplied, rank-residualize C and dloss against p_true inside
    each cell before pooling.
    """
    C = np.asarray(C, float); dloss = np.asarray(dloss, float)
    oct_class = np.asarray(oct_class, int)
    membership = np.asarray(membership, int)
    shadow_id = np.asarray(shadow_id, int)
    p_true_arr = None if p_true is None else np.asarray(p_true, float)

    pooled_x, pooled_y = [], []
    per_cell = {}
    for c, m, sid in sorted(set(zip(oct_class, membership, shadow_id))):
        mask = (oct_class == c) & (membership == m) & (shadow_id == sid)
        n = int(mask.sum())
        key = _cell_key(c, m, sid)
        if n < min_cell_n or np.std(C[mask]) == 0 or np.std(dloss[mask]) == 0:
            per_cell[key] = {"n": n, "rho": None}
            continue
        rx = rankdata(C[mask], method="average")
        ry = rankdata(dloss[mask], method="average")
        if p_true_arr is not None and np.std(p_true_arr[mask]) > 0:
            rz = rankdata(p_true_arr[mask], method="average")
            A = np.column_stack([rz, np.ones(n)])
            rx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
            ry = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        else:
            rx = rx - rx.mean(); ry = ry - ry.mean()
        rho = _pearson(rx, ry)
        per_cell[key] = {"n": n, "rho": None if np.isnan(rho) else float(rho)}
        pooled_x.extend(rx.tolist()); pooled_y.extend(ry.tolist())
    pooled = _pearson(pooled_x, pooled_y) if len(pooled_x) >= 3 else float("nan")
    return pooled, per_cell


def _one_hot(labels):
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    if len(uniq) <= 1:
        return np.empty((len(labels), 0))
    # drop one reference level to avoid a redundant full-rank encoding
    return np.column_stack([(labels == u).astype(float) for u in uniq[1:]])


def incremental_r2(C, dloss, oct_class, membership, shadow_id, p_true):
    """Incremental R^2 of C over cell fixed effects plus p_true.

    The baseline uses categorical OCT-class x membership x shadow cells, not a
    numeric shadow-id slope.  Treating shadow IDs as 0,1,2,... would impose a
    meaningless ordering and can distort the gate.
    """
    y = np.asarray(dloss, float)
    n = len(y)
    cells = np.array([_cell_key(c, m, s)
                      for c, m, s in zip(oct_class, membership, shadow_id)], dtype=object)
    base = np.column_stack([
        np.ones(n),
        _one_hot(cells),
        np.asarray(p_true, float),
    ])
    full = np.column_stack([base, np.asarray(C, float)])

    def r2(A):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
        return 1 - ss_res / ss_tot

    rb, rf = r2(base), r2(full)
    return rb, rf, rf - rb


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------

def _extract_state_dict(obj):
    if isinstance(obj, list):
        if not obj:
            raise ValueError("empty checkpoint list")
        obj = obj[-1]
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError(f"unrecognized checkpoint structure: {type(obj)}")


def load_attack_models(attack_ckpt_dir, pattern, classes, n_in, hidden):
    from models import AttackModel
    out = {}
    for c in classes:
        path = Path(attack_ckpt_dir) / pattern.format(c=c)
        if not path.exists():
            logger.warning(f"attack ckpt missing for class {c}: {path}")
            continue
        obj = torch.load(path, map_location=DEVICE)
        state = _extract_state_dict(obj)
        m = AttackModel(n_in, hidden).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        out[c] = m
    return out


def load_or_build_shadows(needed_ids, args, split):
    from models import SmallCNN
    shadows = {}
    if args.shadow_ckpt_dir:
        for sid in needed_ids:
            path = Path(args.shadow_ckpt_dir) / f"shadow_{sid}.pt"
            if path.exists():
                obj = torch.load(path, map_location=DEVICE)
                state = _extract_state_dict(obj)
                m = SmallCNN(args.in_channels, args.shadow_hidden, args.n_classes).to(DEVICE)
                m.load_state_dict(state); m.eval()
                shadows[sid] = m
        missing = [sid for sid in needed_ids if sid not in shadows]
        if not missing:
            return shadows, {sid: "checkpoint" for sid in shadows}
        logger.warning(f"missing shadow ckpts for {missing}")
        if not args.build_shadow_ckpts:
            _emit_patch_and_exit(missing)
    else:
        missing = list(needed_ids)
    if args.build_shadow_ckpts:
        rebuilt = _build_shadows(missing, args, split)
        shadows.update(rebuilt)
        source = {sid: ("rebuilt_approx" if sid in rebuilt else "checkpoint")
                  for sid in shadows}
        return shadows, source
    _emit_patch_and_exit(needed_ids)


def _build_shadows(needed_ids, args, split):
    """APPROXIMATE: retrain needed shadows from split via repo's train_model."""
    logger.warning("=" * 60)
    logger.warning("BUILDING SHADOWS BY RETRAINING (approximate; will NOT "
                   "bit-reproduce the original prediction vectors / 804 pairs).")
    logger.warning("=" * 60)
    import config as cfgmod
    from data import load_dataset
    from split import build_dataset_from_indices
    from models import SmallCNN, train_model

    cfg = getattr(cfgmod, f"preset_{args.preset}")()
    overrides = json.loads(args.config_overrides_json) if args.config_overrides_json else {}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    X, y, groups = load_dataset(cfg)
    out = {}
    for sid in needed_ids:
        s = split["shadow_models"][sid]
        data = build_dataset_from_indices(X, y, s["train_idx"], s["test_idx"])
        m = SmallCNN(args.in_channels, args.shadow_hidden, args.n_classes).to(DEVICE)
        m, tr, te = train_model(
            m, data, cfg.get_shadow_epochs(), cfg.get_shadow_lr(),
            cfg.get_shadow_batch_size(), cfg.get_shadow_l2(), verbose=False,
            label=f"rebuild-shadow-{sid}", optimizer_type=cfg.optimizer_type,
            lr_decay=cfg.lr_decay)
        m.eval(); out[sid] = m
        if args.shadow_ckpt_dir:
            Path(args.shadow_ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": m.state_dict()},
                       Path(args.shadow_ckpt_dir) / f"shadow_{sid}.pt")
        logger.info(f"rebuilt shadow {sid}: train_acc={tr:.3f} test_acc={te:.3f}")
    return out


def _emit_patch_and_exit(ids):
    logger.error(
        "No shadow checkpoints available for shadows %s.\n"
        "For an EXACT test, add checkpoint saving to the shadow loop and "
        "regenerate. Minimal patch in attack.py::_train_shadows, inside the "
        "for-loop after training `model`:\n\n"
        "    import torch, os\n"
        "    os.makedirs('results/cross_stage/shadow_ckpts', exist_ok=True)\n"
        "    torch.save({'state_dict': model.state_dict()},\n"
        "               f'results/cross_stage/shadow_ckpts/shadow_{i}.pt')\n\n"
        "then re-run run_tracin.py (delete the cached attack_data.npz first so "
        "vectors, scores, LOO and the 804 pairs all come from the checkpointed "
        "shadows). Or re-run this script with --build_shadow_ckpts for an "
        "approximate signal.", ids)
    sys.exit(3)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def read_pairs(pairs_csv):
    rows = []
    with open(pairs_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="dir from 00 (class_local_index.csv)")
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--scores_dir", required=True, help="holds scores_class{c}.npz (for p_true)")
    ap.add_argument("--attack_ckpt_dir", required=True)
    ap.add_argument("--attack_ckpt_pattern", default="sgd_checkpoints_class{c}.pt")
    ap.add_argument("--shadow_ckpt_dir", default=None)
    ap.add_argument("--build_shadow_ckpts", action="store_true")
    ap.add_argument("--preset", default="oct")
    ap.add_argument("--split_path", default=None,
                    help="explicit split JSON; otherwise use 00 manifest then auto-search")
    ap.add_argument(
        "--config_overrides_json", default=None,
        help='optional JSON object applied to preset before dataset loading/rebuild')
    ap.add_argument("--allow_unchecked_artifacts", action="store_true",
                    help="allow 02 to use artifacts without real class/patient validation")
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--shadow_hidden", type=int, default=128)
    ap.add_argument("--attack_hidden", type=int, default=64)
    ap.add_argument("--max_pairs", type=int, default=200, help="cap for the pilot")
    ap.add_argument("--out_dir", default="./results/cross_stage/pair_sep")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.artifacts) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}; run 00 first")
    manifest = json.loads(manifest_path.read_text())
    validation = manifest.get("validation", {})
    fully_checked = (manifest.get("verdict_pass") is True
                     and validation.get("class_checked") is True
                     and validation.get("patient_attached") is True)
    if not fully_checked and not args.allow_unchecked_artifacts:
        raise RuntimeError(
            "Artifacts are not fully validated against the real dataset "
            "(class_checked/patient_attached). Re-run 00 with --load_dataset and "
            "the exact config overrides, or pass --allow_unchecked_artifacts only "
            "for plumbing tests.")

    # provenance lookup (oct_class, per_class_local) -> row
    lookup = {}
    with open(Path(args.artifacts) / "class_local_index.csv", newline="") as f:
        for r in csv.DictReader(f):
            lookup[(int(r["oct_class"]), int(r["per_class_local"]))] = r
    logger.info(f"loaded {len(lookup)} provenance entries")

    pairs = read_pairs(args.pairs_csv)
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]           # csv is pre-sorted by largest gap
    logger.info(f"testing {len(pairs)} pairs")

    # figure out which shadows we need, load models
    needed_rows, needed_shadows = {}, set()
    for pr in pairs:
        c = int(pr["cls"] if "cls" in pr else pr["class"])
        for side in ("A", "B"):
            local = int(pr[f"train_idx_{side}"])
            key = (c, local)
            if key not in lookup:
                continue
            prov = lookup[key]
            needed_rows[key] = prov
            needed_shadows.add(int(prov["shadow_id"]))
    logger.info(f"unique rows={len(needed_rows)}, shadows needed={sorted(needed_shadows)}")

    # load dataset images once
    import config as cfgmod
    from data import load_dataset
    cfg = getattr(cfgmod, f"preset_{args.preset}")()
    overrides = json.loads(args.config_overrides_json) if args.config_overrides_json else {}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    X, y, groups = load_dataset(cfg)
    X = np.asarray(X)

    split = _find_split(args, manifest)   # returns the loaded split dict
    shadows, shadow_sources = load_or_build_shadows(sorted(needed_shadows), args, split)
    attacks = load_attack_models(args.attack_ckpt_dir, args.attack_ckpt_pattern,
                                 sorted({int(pr.get("cls", pr.get("class"))) for pr in pairs}),
                                 args.n_classes, args.attack_hidden)

    # p_true per class from scores npz (train_x stored as the per-class subset)
    p_true_cache = {}
    for c in {int(pr.get("cls", pr.get("class"))) for pr in pairs}:
        sp = Path(args.scores_dir) / f"scores_class{c}.npz"
        if sp.exists() and "train_x" in np.load(sp).files:
            p_true_cache[c] = np.load(sp)["train_x"][:, c]

    # score each unique row
    row_C, row_ptrue = {}, {}
    for key, prov in needed_rows.items():
        c, local = key
        sid = int(prov["shadow_id"])
        raw = int(prov["raw_index"])
        m_i = int(prov["membership"])
        y_i = int(prov["oct_class_raw"]) if int(prov["oct_class_raw"]) >= 0 else c
        if sid not in shadows or c not in attacks:
            continue
        x_i = torch.tensor(X[raw:raw + 1], dtype=torch.float32, device=DEVICE)
        C_i, p_i = frozen_cross_stage_score(shadows[sid], attacks[c], x_i, y_i, m_i)
        row_C[key] = C_i
        row_ptrue[key] = float(p_i[c]) if c not in p_true_cache else float(p_true_cache[c][local])

    # assemble pair-level + row-level stats
    dC, dL, concord = [], [], []
    rowvals = {"C": [], "dloss": [], "class": [], "membership": [],
               "shadow": [], "p_true": []}
    seen = set()
    pair_records = []
    for pr in pairs:
        c = int(pr.get("cls", pr.get("class")))
        kA, kB = (c, int(pr["train_idx_A"])), (c, int(pr["train_idx_B"]))
        if kA not in row_C or kB not in row_C:
            continue
        dloss_A = float(pr["dloss_A"]); dloss_B = float(pr["dloss_B"])
        dc = row_C[kA] - row_C[kB]
        dl = dloss_A - dloss_B
        dC.append(dc); dL.append(dl)
        if abs(dc) > 1e-12 and abs(dl) > 1e-12:
            concord.append(1.0 if np.sign(dc) == np.sign(dl) else 0.0)
        pair_records.append(dict(cls=c, A=kA[1], B=kB[1], C_A=row_C[kA], C_B=row_C[kB],
                                 dC=dc, dloss_A=dloss_A, dloss_B=dloss_B, dL=dl,
                                 concordant=int(np.sign(dc) == np.sign(dl))))
        for k, dloss in ((kA, dloss_A), (kB, dloss_B)):
            if k in seen:
                continue
            seen.add(k)
            prov = needed_rows[k]
            rowvals["C"].append(row_C[k]); rowvals["dloss"].append(dloss)
            rowvals["class"].append(int(k[0]))
            rowvals["membership"].append(int(prov["membership"]))
            rowvals["shadow"].append(int(prov["shadow_id"]))
            rowvals["p_true"].append(row_ptrue[k])

    n_pairs = len(concord)
    concordance = float(np.mean(concord)) if n_pairs else float("nan")
    global_sp = spearman(rowvals["C"], rowvals["dloss"])
    within_sp, per_cell_sp = pooled_within_cell_spearman(
        rowvals["C"], rowvals["dloss"], rowvals["class"], rowvals["membership"],
        rowvals["shadow"], p_true=None)
    partial_sp, per_cell_partial = pooled_within_cell_spearman(
        rowvals["C"], rowvals["dloss"], rowvals["class"], rowvals["membership"],
        rowvals["shadow"], p_true=rowvals["p_true"])
    rb, rf, dr2 = incremental_r2(
        rowvals["C"], rowvals["dloss"], rowvals["class"], rowvals["membership"],
        rowvals["shadow"], rowvals["p_true"]) \
        if len(rowvals["C"]) >= 6 else (float("nan"),) * 3

    summary = {
        "n_pairs_tested": n_pairs,
        "n_non_tie_pairs_for_concordance": len(concord),
        "n_unique_rows": len(seen),
        "pair_concordance": concordance,
        "global_spearman_C_vs_dloss_diagnostic": global_sp,
        "pooled_within_cell_spearman_C_vs_dloss": within_sp,
        "pooled_within_cell_partial_spearman_controlling_p_true": partial_sp,
        "per_cell_spearman": per_cell_sp,
        "per_cell_partial_spearman_controlling_p_true": per_cell_partial,
        "r2_baseline_membership_shadow_ptrue": rb,
        "r2_baseline_plus_C": rf,
        "incremental_r2_of_C": dr2,
        "shadow_sources": {str(k): v for k, v in shadow_sources.items()},
        "stage1_feature": "raw_classification_gradient_proxy",
        "stage1_optimizer_caveat": (
            "This pilot does not use Adam/SGD optimizer state. If the shadow models "
            "were trained with Adam, C_i is a raw-gradient proxy, not the exact update direction."),
    }

    # gate
    gate_fail = (n_pairs > 0 and abs(concordance - 0.5) < 0.05
                 and (np.isnan(within_sp) or abs(within_sp) < 0.05)
                 and (np.isnan(dr2) or dr2 < 0.01))
    summary["gate_verdict"] = "PAUSE_PIVOT" if gate_fail else "SIGNAL_PRESENT_CONTINUE"

    with open(out / "pair_separation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "pair_records.csv", "w", newline="") as f:
        if pair_records:
            w = csv.DictWriter(f, fieldnames=list(pair_records[0].keys()))
            w.writeheader(); w.writerows(pair_records)

    logger.info("=" * 60)
    logger.info("PAIR-SEPARATION VIABILITY RESULT")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    if any(v == "rebuilt_approx" for v in shadow_sources.values()):
        logger.warning("NOTE: one or more shadows were retrained (approximate). "
                       "Confirm any positive signal with exact shadow checkpoints "
                       "before trusting the number.")


def _find_split(args, manifest):
    if args.split_path:
        return common.load_split(args.split_path)
    manifest_split = manifest.get("split_path")
    if manifest_split and Path(manifest_split).exists():
        return common.load_split(manifest_split)
    # Fallback search for moved result trees.
    for cand in [args.scores_dir, str(Path(args.scores_dir).parent),
                 str(Path(args.scores_dir).parents[1])]:
        try:
            return common.load_split(common.resolve_split_path(cand, None))
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        "Could not locate split JSON. Pass --split_path explicitly. The path "
        "stored by 00 may no longer exist if artifacts were moved.")


if __name__ == "__main__":
    main()
