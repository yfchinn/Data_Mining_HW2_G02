# T3 sampling validation results

Fixed LR: C=1, class_weight=balanced; sampling seed=42; SMOTE fraud/normal=0.01.
exp1: five validation folds; exp2: one chronological validation split.
Thresholds maximize F1 on the same validation set. No test evaluation. No contamination.

| Mode | exp1 AP mean | exp1 F1 mean | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| P0 | 0.7477 | 0.8190 | 0.7727 | 0.7921 |
| P1-1 | 0.6355 | 0.7058 | 0.4387 | 0.5455 |
| P1-10 | 0.7405 | 0.8043 | 0.7650 | 0.8200 |
| P1-5 | 0.7291 | 0.8059 | 0.2700 | 0.4596 |
| P2-1 | 0.6204 | 0.6882 | 0.3274 | 0.4910 |
| P2-10 | 0.7255 | 0.7937 | 0.7487 | 0.8119 |
| P2-5 | 0.7224 | 0.7903 | 0.5294 | 0.6535 |
| P3-1 | 0.6396 | 0.6880 | 0.3159 | 0.5203 |
| P3-10 | 0.7413 | 0.8051 | 0.7591 | 0.8119 |
| P3-5 | 0.7226 | 0.7887 | 0.7595 | 0.7921 |
| P4-1 | 0.6584 | 0.7155 | 0.3019 | 0.5414 |
| P4-10 | 0.7378 | 0.8036 | 0.7511 | 0.8119 |
| P4-5 | 0.7028 | 0.7756 | 0.6959 | 0.7593 |
| P5 | 0.7475 | 0.8195 | 0.7696 | 0.8119 |
| P6 | 0.7193 | 0.8106 | 0.7209 | 0.8119 |

P0 has the highest mean AP in exp1; P5 has a slightly higher mean F1 (0.8195 versus 0.8190). P0 has the highest AP in exp2; P1-10 has the highest F1. These validation results do not establish statistical significance or test superiority. Sampling-seed variation has not been measured.

K-Medoids uses the existing candidate-based approximation. Detailed sample counts and fold metrics are in the accompanying CSV files.

