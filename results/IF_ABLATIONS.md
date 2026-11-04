# IF controlled ablations (validation only)

All models use six fixed features except the feature-set ablation. No test metrics were computed.

## Training contamination

Fixed normal rows, normal-only scaler, 200 trees, 256 samples/tree. Three seeds per fold.
Exp1 aggregates 5 folds x 3 seeds (correlated runs); exp2 aggregates 3 seeds. Std is descriptive, not a significance test.
Fraud subsets are nested within each fold/seed. Inclusion counts are bounded by the available fraud population.
C0 is re-fit with a stable normal-row order; exp1 C0 can differ from earlier P0 due to changed row order.

| Experiment | Mode | Fraud count | Contamination % | AP mean +/- std | Delta AP vs paired C0 | F1 mean |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 | C0 | 0 | 0.0000 | 0.6148 +/- 0.0548 | +0.0000 | 0.6692 |
| exp1 | C1 | 26 | 0.0143 | 0.6223 +/- 0.0482 | +0.0075 | 0.6720 |
| exp1 | C2 | 78-79 | 0.0430 | 0.6330 +/- 0.0462 | +0.0182 | 0.6807 |
| exp1 | C3 | 157-158 | 0.0863 | 0.6461 +/- 0.0429 | +0.0313 | 0.6864 |
| exp1 | C4 | 315-316 | 0.1729 | 0.6681 +/- 0.0383 | +0.0533 | 0.7062 |
| exp2 | C0 | 0 | 0.0000 | 0.4489 +/- 0.1087 | +0.0000 | 0.5388 |
| exp2 | C1 | 30 | 0.0176 | 0.4532 +/- 0.1160 | +0.0044 | 0.5407 |
| exp2 | C2 | 90 | 0.0528 | 0.5003 +/- 0.1021 | +0.0514 | 0.5671 |
| exp2 | C3 | 180 | 0.1054 | 0.5695 +/- 0.0675 | +0.1206 | 0.5997 |
| exp2 | C4 | 360 | 0.2107 | 0.6057 +/- 0.0288 | +0.1568 | 0.6144 |

Low-level contamination improved mean validation AP in both protocols. This is an empirical observation, not evidence that higher contamination is beneficial. The study spans only about 0-0.21% contamination, not 1/5/10%.
Adding rows also changes the training size and random subsamples. The study measures the effect of adding fraud while holding all normal rows fixed; it does not isolate fraud identity from training-size effects.
Model parameters are fixed, but the threshold is re-selected by the same validation-best-F1 rule in each condition. AP is threshold-independent.
Main comparisons remain normal-only, in accordance with the chosen assignment protocol.

## Feature-set ablation

Fixed 200 trees and max_samples=256. Compare all 30 features to the fixed six-feature set; no KS recomputation.
Exp1 uses fold 0 only, with 3 seeds; exp2 uses its chronological split with 3 seeds.

| Experiment | All-feature AP | Six-feature AP | Paired delta AP mean +/- std | Baseline seed std |
| --- | --- | --- | --- | --- |
| exp1 | 0.1335 | 0.6554 | +0.5219 +/- 0.0508 | 0.0146 |
| exp2 | 0.0190 | 0.4489 | +0.4299 +/- 0.1073 | 0.0021 |

Mean feature-set gains exceed baseline seed std in both splits, but validation-guided model selection is optimistic and does not establish independent test superiority.
These two ablations cover feature design and anomaly-detection-specific training contamination. Max-samples diagnostics and T3 sampling studies are available separately.
