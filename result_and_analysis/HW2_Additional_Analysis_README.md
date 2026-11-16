# HW2 §7–§8: Colab replication and reviewer follow-ups

This experiment supplements **the frozen P0 results** in `results/FINAL_TEST.md`. The main paper's test metrics must **not** be silently replaced by these retrospective, post-feedback explorations. New threshold, feature-selection, and scoring decisions require a **new untouched forward holdout** for an unbiased fresh test estimate.

## Run in Colab

1. Open `result_and_analysis/HW2_Additional_Analysis_Colab.ipynb` from branch `analysis/reviewer-fixes-20261116` on GitHub with Google Colab.
2. Execute the cells sequentially. The notebook clones the repo, obtains the dataset with Git LFS, checks its exact SHA-256, installs a compatible scikit-learn environment, then executes `result_and_analysis/extended_analysis.py`.
3. Read the output tables in `outputs/extended_analysis/`. Download them as ZIP; generated outputs are ignored by Git and not automatically pushed.
4. If Git LFS does not download the CSV, obtain the official `creditcard.csv` from the ULB Kaggle data page and upload to `data/creditcard.csv`. **Check the SHA-256 before running**. If the hash differs, do not cite numbers as reproductions of the report.

From the repository root, the full run is:

```bash
python result_and_analysis/extended_analysis.py \
  --csv data/creditcard.csv \
  --out outputs/extended_analysis \
  --experiments exp1 exp2 \
  --seeds 42 43 44 \
  --exp1-folds 5 \
  --bootstrap-repeats 1000 \
  --capacity-rate 0.001 \
  --contam-total 0 \
  --contam-max 360
```

Dependencies: Python 3.11/3.12, NumPy, pandas, SciPy, scikit-learn. This self-contained extension does *not* require `scikit-learn-extra`; it reproduces P0 split logic directly from `data_preparation(exp1).py` and `data_preparation(exp2).py`.

## Analyses and output files

| Output file | Purpose | Evaluation partition |
| --- | --- | --- |
| `comparisons.csv` | original six-feature IF, train-fold-only K–S top-six IF, and V14 three-sigma AP/F1/etc | validation and *retrospective* test |
| `fold_selected_features.csv` | selected feature and training K–S statistic per fold | training only |
| `IF_vs_T1_paired_bootstrap.csv` | paired, class-stratified transaction bootstrap of mean AP gap on shared test rows | previously used P0 test (retrospective) |
| `size_matched_contamination_validation.csv` | C0→C4, **fixed number of training rows** and 200 trees / 256 samples | validation only |
| `size_matched_contamination_summary.csv` | mean, SD of contamination AP and *paired* AP change | validation only |
| `capacity_threshold_validation.csv` | F1-optimal threshold against validation-allowed alerts at assumed 0.1% rate | validation only |
| `threshold_sensitivity_validation.csv` | IF validation FPR / recall / alert changes at threshold shifts −0.05, −0.025, 0, +0.025, +0.05 | validation only |
| `hyperparameter_sweep_validation.csv` | 100/200/400 trees, 256/512 max_samples, 3 seeds; fold 0 | validation only |
| `case_feature_perturbation.csv` | original IF score versus replacing one input with **normal training median**; decision-path feature split counts | fixed previously inspected Exp1 error cases |
| `case_attribution_summary.csv` | mean and SD of score reductions for each of 3 rows | descriptive, retrospective |
| `protocol.json` | dataset hash and methodological caveats | metadata |

## Methodology / critical interpretations

- **Original P0 sample identities:** Exp1: one stratified 20% test holdout (`random_state=42`), five stratified CV folds of remaining 80%. Exp2: stable sort by Time followed by 60/20/20 integer boundaries. These match repository scripts. Same test IDs in all 15 Exp1 fold/seed evaluations.
- **Fold-safe top-six selection:** `ks_2samp` computes class separation ONLY within each current training fold. Existing six-feature IF is retained as a fixed reference. Selection on training fold labels is allowed for semi-supervised feature design but it is **not label-free feature selection**. An outer untouched test is required to fairly evaluate the *selected pipeline*, not cherry-picked post-test results.
- **Constant-size contamination:** the total number of *training* rows is held fixed within each fold. Fraud samples progressively replace normal samples; IF sees a changing class composition while 200 trees, `max_samples=256`, scaler-irrelevant V features, validation population, and seed are held fixed. Default `--contam-total 0` means use the entire available normal training population as the fixed training size; C0 is all normal rows, C4 replaces up to 315/316 Exp1 or 360 Exp2 normals. This controls the original sample-count confound. Stochastic tree subsample *membership* necessarily changes when training rows change; do not claim identical trees or independent runs.
- **Capacity threshold:** 0.001 means an **assumed 0.1% validation alert-rate ceiling**, not observed bank capacity. Threshold is chosen on validation scores only, with tie handling; outputs report validation FP/FN and alert counts. The actual business budget is unknown.
- **Interpretability:** perturbation uses a *normal training median* replacement in the trained IF, with positive `score_reduction` meaning the replacement lowered the anomaly score. Path split counts indicate tree traversal frequency, NOT causal contributions. Feature perturbations break correlation dependencies of anonymized PCA features and cannot establish financial reasons or causal responsibility. No SHAP claims are made.
- **Bootstrap:** pairs on the same transactions and class counts; treats repeated model scores as repeated *models*, not repeated independent test sets. Without card IDs, CI may be overly narrow.
- **Retrospective test caution:** The original test set has already been investigated extensively. Do not call novel results from that test *untouched* or use them to select the preferred hyperparameters/threshold and then report a new unbiased test metric. Further independently collected chronological data would be required.
