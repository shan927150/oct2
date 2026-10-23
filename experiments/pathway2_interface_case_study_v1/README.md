# Interface LDS case study

CPU analysis of the completed interface run. Uses existing predictions, retrained probabilities and attack truth. No models are trained. No new permutation test or bootstrap is run.

`case_study.py` is the reviewed `case_study(2).py`, copied without changes. The whole-path score tau ranks all training images. Own-row scores and their correlation use only rows active in the DME/DRUSEN attack objective. The deleted-minus-retained contrast is based on the existing half-data subsets, not LOO.

Keep this diagnostic checkout separate from `/u/yli103/oct2-interface-lds-v1r1`. The `--repo` argument must point to that original checkout at commit `449e5ff03bedd6dd560d39c0ac585fd3a7522a19`, whose path and commit are recorded in the completed run. Do not pull this diagnostic branch into the frozen checkout.

## Submit on Delta

From a Delta login node, clone the diagnostic branch once:

```bash
git clone --branch exp/pathway2-interface-case-study-v1 --single-branch \
  https://github.com/shan927150/oct2.git ~/oct2-interface-case-study
```

Submit one CPU job that runs top-20 and top-100 sequentially:

```bash
pkg="$HOME/oct2-interface-case-study/experiments/pathway2_interface_case_study_v1"
out=/u/yli103/oct2-pathway2-runs/A/interface_lds_v1r1_20261022T230209460267514Z
mkdir -p "$out/case_study"
sbatch --output="$out/case_study/slurm-%j.log" \
  "$pkg/case_study.slurm" "$pkg" /u/yli103/oct2-interface-lds-v1r1 "$out"
```

## Read results

Outputs are under `OUT/case_study/gamma0.03_k20/` and `OUT/case_study/gamma0.03_k100/`. Each contains:

- `images.csv`: raw image IDs, features, own-row contrasts, tau and group membership.
- `groups.csv`: top/bottom group summaries.
- `p_true_bins.csv`: true-class confidence bins, effect sizes and correlations.
- `subsets.csv`: row1, row3, attack truth and their ranks over 32 subsets.
- `probability_bounds.csv`: per-seed out-of-bounds vector fractions and maximum violations, with tolerance 1e-12.
- `deleted_row_contributions.csv`: each deleted image's contribution to the row3-minus-row1 score difference.

The Slurm log preserves the printed metrics. At gamma 0.03, compare with the existing report before interpreting cases:

| Metric | Report reference |
| --- | ---: |
| `lds_row1` | 0.261730205 |
| `lds_row3` | 0.729838710 |
| `own_effect_spearman_true_class` | 0.4665218 |

Both values of top-k must give the same three metrics, because top-k changes only the selected image groups. The script prints three significant digits. `row1_vs_tau_identity_max_rel_err` checks numerical consistency of the saved predictions and tau; it does not compare against retraining. Image contributions add to score differences, not to correlation improvements.

## Validation before submission

Syntax and CLI checks passed. Synthetic checks passed for direct row3 replacement, score decomposition, active-row selection, whole-path selection, subset ranks, probability bounds, matched seed panels and separate top-k output directories. The actual Delta arrays have not yet been rerun with this script.
