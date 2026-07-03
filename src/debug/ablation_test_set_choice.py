#!/usr/bin/env python
"""
ablation_test_set_choice.py  (Ablation 1 - cheap, reuses saved scores matrix)

Question: does the cross-influence score (and its band structure) mainly reflect
alignment with the MEMBER-test gradient direction? Re-aggregate the SAVED scores
matrix over different test-row subsets (no gradient recompute) and compare.

Subsets:
  A. member test only      (test_y == 1)         <- the current LOO definition
  B. non-member test only  (test_y == 0)
  C. all test
  D. correctly attacked member   (test_y==1 and attack predicts member)
  E. incorrectly attacked member (test_y==1 and attack predicts non-member)
For D/E we need the attack model's predictions on the test set: loaded from the
final checkpoint in sgd_checkpoints_class{c}.pt (fallback: retrain).

Outputs per class: Spearman of each subset's train influence profile vs the member
profile and vs train membership, plus the band centroid shift.

  python ablation_test_set_choice.py \
    --scores_dir results/oct_mia_tracin_sgd/tracin \
    --checkpoint_dir results/oct_mia_tracin_sgd/tracin \
    --output_dir results/oct_debug_sgd/test_set_choice --classes 0 1 2 3
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr

from models import build_attack_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _get(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def attack_predict(state_dict, X):
    m = build_attack_model("nn", X.shape[1], 64).to(DEVICE)
    m.load_state_dict(state_dict); m.eval()
    with torch.no_grad():
        out = m(torch.tensor(X, dtype=torch.float32, device=DEVICE))
    return out.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for c in args.classes:
        sp = os.path.join(args.scores_dir, f"scores_class{c}.npz")
        if not os.path.exists(sp):
            print(f"skip class {c}: {sp} missing"); continue
        npz = np.load(sp, allow_pickle=True)
        scores = np.asarray(_get(npz, "scores", "scores_raw"), float)   # (N_test, N_train)
        train_y = np.asarray(_get(npz, "train_y", "y_train", "attack_train_y"), int)
        test_x = np.asarray(_get(npz, "test_x", "X_test", "attack_test_x"), np.float32)
        test_y = np.asarray(_get(npz, "test_y", "y_test", "attack_test_y"), int)
        name = CLASS_NAMES[c] if c < 4 else str(c)

        # attack predictions on test (for correct/incorrect split)
        pred = None
        ckpt_path = os.path.join(args.checkpoint_dir or args.scores_dir, f"sgd_checkpoints_class{c}.pt")
        if os.path.exists(ckpt_path) and test_x is not None:
            obj = torch.load(ckpt_path, map_location="cpu")
            if isinstance(obj, dict) and "checkpoints" in obj:
                obj = obj["checkpoints"]
            last = obj[-1]
            sd = last["state_dict"] if isinstance(last, dict) and "state_dict" in last else last
            pred = attack_predict(sd, test_x)

        subsets = {
            "member": test_y == 1,
            "nonmember": test_y == 0,
            "all": np.ones(len(test_y), bool),
        }
        if pred is not None:
            subsets["correct_member"] = (test_y == 1) & (pred == 1)
            subsets["incorrect_member"] = (test_y == 1) & (pred == 0)

        profiles = {}
        for tag, mask in subsets.items():
            if mask.sum() == 0:
                continue
            profiles[tag] = scores[mask].mean(axis=0)   # (N_train,)

        ref = profiles["member"]
        print(f"\n=== Class {c} ({name}) ===")
        print(f"  {'subset':>16} {'n_test':>7} {'rho_vs_member':>14} {'rho_vs_membership':>18} {'mean_pos':>9} {'mean_neg':>9}")
        for tag, prof in profiles.items():
            n_test = int(subsets[tag].sum())
            rho_ref = float(spearmanr(prof, ref)[0])
            rho_mem = float(spearmanr(prof, train_y)[0])
            mean_pos = float(prof[train_y == 1].mean())
            mean_neg = float(prof[train_y == 0].mean())
            print(f"  {tag:>16} {n_test:>7} {rho_ref:>14.3f} {rho_mem:>18.3f} {mean_pos:>9.3f} {mean_neg:>9.3f}")
            rows.append(dict(cls=c, class_name=name, subset=tag, n_test=n_test,
                             spearman_vs_member_profile=rho_ref, spearman_vs_train_membership=rho_mem,
                             mean_influence_on_member_train=mean_pos,
                             mean_influence_on_nonmember_train=mean_neg))

        # scatter: member-profile vs nonmember-profile influence (per train point)
        if "nonmember" in profiles:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(profiles["member"], profiles["nonmember"], c=train_y, cmap="coolwarm", s=8, alpha=0.4)
            lim = [min(profiles["member"].min(), profiles["nonmember"].min()),
                   max(profiles["member"].max(), profiles["nonmember"].max())]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel("influence from member-test"); ax.set_ylabel("influence from non-member-test")
            ax.set_title(f"Class {c} ({name}): train influence under two test subsets")
            fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, f"test_set_choice_class{c}.png"), dpi=120)
            plt.close(fig)

    with open(os.path.join(args.output_dir, "test_set_choice_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved test_set_choice_summary.csv + per-class scatter")
    print("Read: if every subset's profile has high rho_vs_member and high rho_vs_membership, "
          "the score mostly encodes member-direction alignment (problem D), independent of which test points are used.")


if __name__ == "__main__":
    main()
