# T3 sampling: three seeds

Sampling/model seeds: 42, 43, 44. Raw split seed: fixed 42.
exp1: five folds; average seeds within each fold, then report cross-fold mean +/- sample std.
exp2: one chronological split; report cross-seed mean +/- sample std.
LR C=1, class_weight=balanced, SMOTE fraud/normal=0.01. Validation only; thresholds selected on validation.
Total: 225 exp1 fits and 45 exp2 fits (including P0).

| Mode | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| P0 | 0.7477 +/- 0.0559 | 0.8190 +/- 0.0145 | 0.7727 +/- 0.0000 | 0.7921 +/- 0.0000 |
| P1-1 | 0.6456 +/- 0.0438 | 0.7136 +/- 0.0504 | 0.3815 +/- 0.0553 | 0.5454 +/- 0.0191 |
| P1-5 | 0.7169 +/- 0.0620 | 0.7865 +/- 0.0250 | 0.5973 +/- 0.2836 | 0.6918 +/- 0.2011 |
| P1-10 | 0.7380 +/- 0.0555 | 0.8065 +/- 0.0172 | 0.7110 +/- 0.0485 | 0.7653 +/- 0.0519 |
| P2-1 | 0.6200 +/- 0.0978 | 0.6936 +/- 0.0701 | 0.2511 +/- 0.0737 | 0.4544 +/- 0.0463 |
| P2-5 | 0.7148 +/- 0.0588 | 0.7913 +/- 0.0197 | 0.5439 +/- 0.1388 | 0.6699 +/- 0.0604 |
| P2-10 | 0.7309 +/- 0.0584 | 0.7978 +/- 0.0195 | 0.7504 +/- 0.0185 | 0.8016 +/- 0.0179 |
| P3-1 | 0.6504 +/- 0.0488 | 0.7123 +/- 0.0203 | 0.2341 +/- 0.0822 | 0.4439 +/- 0.0789 |
| P3-5 | 0.7236 +/- 0.0556 | 0.7908 +/- 0.0098 | 0.5551 +/- 0.2183 | 0.6629 +/- 0.1371 |
| P3-10 | 0.7334 +/- 0.0624 | 0.7971 +/- 0.0168 | 0.7408 +/- 0.0358 | 0.7743 +/- 0.0462 |
| P4-1 | 0.6620 +/- 0.0487 | 0.7239 +/- 0.0289 | 0.2325 +/- 0.0629 | 0.4651 +/- 0.0684 |
| P4-5 | 0.7161 +/- 0.0664 | 0.7875 +/- 0.0260 | 0.7466 +/- 0.0439 | 0.7998 +/- 0.0351 |
| P4-10 | 0.7325 +/- 0.0580 | 0.8030 +/- 0.0189 | 0.7520 +/- 0.0146 | 0.8027 +/- 0.0159 |
| P5 | 0.7465 +/- 0.0510 | 0.8213 +/- 0.0166 | 0.7726 +/- 0.0034 | 0.8121 +/- 0.0041 |
| P6 | 0.7228 +/- 0.0387 | 0.8114 +/- 0.0147 | 0.6954 +/- 0.0366 | 0.7965 +/- 0.0202 |

P0 and P5 (SMOTE) have very similar AP. P5 has a higher validation F1 mean in both protocols, but this does not establish statistical significance or test superiority.
Several undersampling settings have large seed sensitivity, especially exp2 P1-5/P3-5. Single-seed rankings were unreliable.
Within-fold seed variation is saved separately in *_within_fold_seeds.csv. Do not treat 15 correlated fold/seed runs as independent significance-test replicates.
K-Medoids uses the existing candidate-based approximation. Every new seed was checked against the original seed42 raw split IDs. No test scoring was performed.
