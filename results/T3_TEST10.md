# T3 test-10% split exploration (validation only)

2026-11-03. Fixed balanced LR, C=1, sampling/model seeds 42/43/44; raw split seed=42; SMOTE fraud/normal=0.01.
exp1: 72/18/10 approximately, using 5 stratified development folds. exp2: chronological 70/20/10.
225 exp1 fits and 45 exp2 fits, including P0 controls. New test sets were NOT scored.
This is post-test exploration. Validation populations change and overlap data previously used in the original test; improvements cannot be attributed solely to more training rows or treated as fresh independent confirmation.

## Split counts (before sampling, fold 0)

| Experiment | Train rows | Validation rows | Test rows | Train fraud | Validation fraud | Test fraud |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 | 205060 | 51266 | 28481 | 354 | 89 | 49 |
| exp2 | 199364 | 56962 | 28481 | 384 | 86 | 22 |

## Test 10%: validation metrics

exp1: mean seeds within each fold, then cross-fold mean/std. exp2: cross-seed mean/std.

| Mode | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| P0 | 0.7458 +/- 0.0299 | 0.8215 +/- 0.0317 | 0.8244 +/- 0.0000 | 0.8258 +/- 0.0000 |
| P1-1 | 0.6155 +/- 0.1082 | 0.6867 +/- 0.0970 | 0.5901 +/- 0.2722 | 0.6726 +/- 0.1651 |
| P1-5 | 0.7334 +/- 0.0442 | 0.8025 +/- 0.0430 | 0.7422 +/- 0.1500 | 0.7739 +/- 0.1200 |
| P1-10 | 0.7382 +/- 0.0313 | 0.8138 +/- 0.0290 | 0.8191 +/- 0.0145 | 0.8325 +/- 0.0163 |
| P2-1 | 0.6053 +/- 0.0684 | 0.6753 +/- 0.0630 | 0.4878 +/- 0.0907 | 0.6376 +/- 0.0456 |
| P2-5 | 0.7158 +/- 0.0501 | 0.7948 +/- 0.0561 | 0.7367 +/- 0.0490 | 0.7621 +/- 0.0391 |
| P2-10 | 0.7234 +/- 0.0412 | 0.8030 +/- 0.0449 | 0.7978 +/- 0.0128 | 0.8085 +/- 0.0107 |
| P3-1 | 0.6969 +/- 0.0595 | 0.7771 +/- 0.0621 | 0.6440 +/- 0.1279 | 0.7112 +/- 0.0564 |
| P3-5 | 0.7357 +/- 0.0311 | 0.8141 +/- 0.0316 | 0.8131 +/- 0.0236 | 0.8084 +/- 0.0390 |
| P3-10 | 0.7438 +/- 0.0338 | 0.8144 +/- 0.0288 | 0.7994 +/- 0.0482 | 0.8102 +/- 0.0565 |
| P4-1 | 0.6853 +/- 0.0594 | 0.7594 +/- 0.0450 | 0.5091 +/- 0.1423 | 0.6673 +/- 0.0749 |
| P4-5 | 0.7300 +/- 0.0364 | 0.8071 +/- 0.0385 | 0.7313 +/- 0.0784 | 0.7625 +/- 0.0452 |
| P4-10 | 0.7351 +/- 0.0299 | 0.8135 +/- 0.0340 | 0.8178 +/- 0.0048 | 0.8306 +/- 0.0064 |
| P5 | 0.7490 +/- 0.0311 | 0.8217 +/- 0.0319 | 0.8273 +/- 0.0076 | 0.8390 +/- 0.0125 |
| P6 | 0.7327 +/- 0.0277 | 0.8178 +/- 0.0279 | 0.8053 +/- 0.0219 | 0.8423 +/- 0.0031 |

## Original test20 vs new test10: descriptive validation comparison

| Mode | exp1 AP test20 -> test10 | exp2 AP test20 -> test10 |
| --- | --- | --- |
| P0 | 0.7477 -> 0.7458 | 0.7727 -> 0.8244 |
| P1-1 | 0.6456 -> 0.6155 | 0.3815 -> 0.5901 |
| P1-5 | 0.7169 -> 0.7334 | 0.5973 -> 0.7422 |
| P1-10 | 0.7380 -> 0.7382 | 0.7110 -> 0.8191 |
| P2-1 | 0.6200 -> 0.6053 | 0.2511 -> 0.4878 |
| P2-5 | 0.7148 -> 0.7158 | 0.5439 -> 0.7367 |
| P2-10 | 0.7309 -> 0.7234 | 0.7504 -> 0.7978 |
| P3-1 | 0.6504 -> 0.6969 | 0.2341 -> 0.6440 |
| P3-5 | 0.7236 -> 0.7357 | 0.5551 -> 0.8131 |
| P3-10 | 0.7334 -> 0.7438 | 0.7408 -> 0.7994 |
| P4-1 | 0.6620 -> 0.6853 | 0.2325 -> 0.5091 |
| P4-5 | 0.7161 -> 0.7300 | 0.7466 -> 0.7313 |
| P4-10 | 0.7325 -> 0.7351 | 0.7520 -> 0.8178 |
| P5 | 0.7465 -> 0.7490 | 0.7726 -> 0.8273 |
| P6 | 0.7228 -> 0.7327 | 0.6954 -> 0.8053 |

These comparisons use DIFFERENT validation populations; no paired significance claim is made. 1:1 sampling remains weaker and some settings remain seed-sensitive.
All thresholds are selected on validation; this is not a new final-test report. Original test20 models, metrics, and FINAL_TEST.md are preserved.
