# P0 main comparison: seeds 42, 43, 44

Validation only. All raw splits remain fixed at split_seed=42.
exp1: five folds x three model seeds; average seeds within each fold, then report mean +/- sample std across five folds.
exp2: mean +/- sample std across three model seeds on one chronological split.
These are descriptive statistics; folds and seeds are not independent test replicates.

| Tier | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| V14 3-sigma | 0.5970 +/- 0.0340 | 0.1767 +/- 0.0087 | 0.4841 +/- 0.0000 | 0.1020 +/- 0.0000 |
| IF six features | 0.6110 +/- 0.0348 | 0.6573 +/- 0.0209 | 0.4489 +/- 0.1087 | 0.5388 +/- 0.0631 |
| Logistic regression | 0.7477 +/- 0.0559 | 0.8190 +/- 0.0145 | 0.7727 +/- 0.0000 | 0.7921 +/- 0.0000 |

T1 and lbfgs logistic regression are deterministic here; their within-fold seed std is zero. Their exp1 cross-fold std is not zero.
The IF remains normal-only, with fixed V14/V10/V12/V4/V11/V17, 200 trees, max_samples=256.
Within-fold seed variation is separately recorded in p0_within_fold_seed_summary.csv.
T3 P1-P6 sampling seeds 42/43/44 have now been completed separately; see T3_SAMPLING.md.
Contamination already has three seeds. Final test evaluation and significance analysis are now documented separately in FINAL_TEST.md.
