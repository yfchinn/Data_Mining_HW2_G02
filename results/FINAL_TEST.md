# Final frozen P0 test results

Configuration was frozen before scoring. Models and validation thresholds were loaded from saved artifacts; no retraining or test threshold tuning.
T1: fixed V14, 3sigma. T2: normal-only IF on V14/V10/V12/V4/V11/V17, 200 trees, 256 samples/tree. T3: all-feature LR, C=1, balanced, P0 (no resampling).
T3 P0 was retained because validation AP was effectively tied with SMOTE; no test-based model selection was performed.

exp1: five folds x seeds 42/43/44; average seeds within each fold, then cross-fold mean +/- sample std. All folds share the same test set.
exp2: seeds 42/43/44 on the same chronological test split; cross-seed mean +/- sample std.

| Experiment | Tier | Test AP | ROC-AUC | F1 | Training seconds | Score ms/sample | Preprocess+score ms/sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp1 | tier1_three_sigma | 0.5883 +/- 0.0001 | 0.9563 +/- 0.0000 | 0.1801 +/- 0.0004 | 0.0132 +/- 0.0053 | 0.000142 +/- 0.000041 | 0.000189 +/- 0.000053 |
| exp1 | tier2_isolation_forest | 0.5861 +/- 0.0154 | 0.9637 +/- 0.0006 | 0.6237 +/- 0.0156 | 0.8200 +/- 0.4185 | 0.007523 +/- 0.000936 | 0.007561 +/- 0.000941 |
| exp1 | tier3_logistic_regression | 0.7175 +/- 0.0092 | 0.9707 +/- 0.0036 | 0.8119 +/- 0.0092 | 0.9710 +/- 0.4176 | 0.000341 +/- 0.000082 | 0.000387 +/- 0.000092 |
| exp2 | tier1_three_sigma | 0.6146 +/- 0.0000 | 0.9527 +/- 0.0000 | 0.1506 +/- 0.0000 | 0.0104 +/- 0.0013 | 0.000086 +/- 0.000007 | 0.000122 +/- 0.000011 |
| exp2 | tier2_isolation_forest | 0.4592 +/- 0.0910 | 0.9750 +/- 0.0017 | 0.5355 +/- 0.0354 | 0.5358 +/- 0.0463 | 0.006476 +/- 0.000181 | 0.006509 +/- 0.000183 |
| exp2 | tier3_logistic_regression | 0.7429 +/- 0.0000 | 0.9819 +/- 0.0000 | 0.7812 +/- 0.0000 | 0.5332 +/- 0.0107 | 0.000231 +/- 0.000013 | 0.000265 +/- 0.000016 |

## Significance: IF versus strongest supervised reference LR

1000 paired class-stratified bootstrap replicates, seed 20261102. Same sampled transaction multiplicities for all fixed trained models; statistic is the difference of mean per-run AP (IF minus LR).
Tied scores are handled exactly; weighted AP was tested against sklearn. LR identical seed predictions are deduplicated with equal fold weighting.

| Experiment | AP gap (IF - LR) | Percentile 95% CI | Excludes zero |
| --- | --- | --- | --- |
| exp1 | -0.1314 | [-0.1854, -0.0752] | True |
| exp2 | -0.2837 | [-0.3599, -0.2046] | True |

Both intervals lie below zero: LR has higher test AP under each frozen protocol. This is a conditional result on these models and these test populations, not a universal algorithm ranking.
Bootstrap assumes independent transactions and conditions on observed class counts. Card IDs are unavailable for clustered resampling; temporal/card dependence can make these intervals optimistic. No correction for comparing two separate protocols was applied.

## Runtime and reproducibility

Training times are the recorded original model-fit times. Inference was timed in this test run, as a single batch; ms/sample is batch-amortized CPU latency, not single-request online latency. Preprocess+score includes scaler transformation, excludes CSV IO and model loading.
Per-run metrics contain confusion counts, precision/recall/FPR, thresholds and timing. Saved test predictions are in outputs/final_test/<experiment>/fold_<n>/.
The frozen protocol and input SHA256 are saved in final_test_protocol.json. Evaluation refuses to overwrite an existing outputs/final_test directory.
No deployment, interpretation, or error-case study was performed, per the agreed experiment-only scope.
