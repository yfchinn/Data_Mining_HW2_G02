# HW2 Results、Analysis

本資料夾整理報告第 7 章（Results）與第 8 章（Analysis）使用的主要實驗數據來源、數字定義、錯誤案例分析方式，以及 `error_analysis.py`。


> **實驗範圍提醒**
>
> 本文件的正式模型比較以原始 P0 protocol 的 frozen test evaluation 為準。新版 Exp1 P1–P4 先採樣再切分的探索結果，資料分布與正式 P0 不同，不混入正式比較表。
> IF feature-set ablation 與 contamination ablation 是 validation-only 實驗，不可當成 test performance。

---

## 1. 資料集與實驗設定

資料集為 `data/creditcard.csv`，共 284,807 筆交易、492 筆 fraud、30 個輸入特徵。`Class=0` 代表正常交易，`Class=1` 代表 fraud。

主要資料來源：

- `README.md`：資料前處理、共用 split interface、三個 tier 的模型定義與參數。
- `results/EXPERIMENT_SUMMARY_ZH.md` 第 2–3 節：資料集統計、Exp1/Exp2 切分、模型設定與門檻。
- `results/FINAL_TEST.md`：正式凍結後的 test 結果與 bootstrap 顯著性檢定。

### Evaluation protocols

| Protocol | Split | Repeated runs |
|---|---|---|
| Exp1 | 固定 20% stratified test；其餘 80% 做 stratified 5-fold | 5 folds × seeds 42/43/44；每個 fold 的 3 seeds 先平均，再計算 5 folds 的 mean ± sample std |
| Exp2 | 依 `Time` 排序，60%/20%/20% chronological split | 同一 split 上 seeds 42/43/44，報告 mean ± sample std |

Exp1 的五個 fold 共用同一份 test set。因此 fold/seed 的標準差主要描述不同訓練結果的變異，不是五份獨立 test populations 的變異。

### Three tiers

| Tier | Method | Training population | Features / decision |
|---|---|---|---|
| T1 | V14 three-sigma | Normal-only | 固定使用 V14；`abs((V14-mean)/std)`，門檻固定為 3 |
| T2 | Isolation Forest | Normal-only | V14, V10, V12, V4, V11, V17；200 trees、`max_samples=256`；validation 最大 F1 選 threshold |
| T3 | Logistic Regression | Normal + fraud | 全部 30 個特徵；`C=1`、L2、`class_weight=balanced`、`lbfgs`、`max_iter=2000`；validation 最大 F1 選 threshold |

三個模型的異常分數方向均統一為「分數越大越異常」，T2/T3 使用 `score >= threshold` 判定 anomaly。T2 不使用 sklearn Isolation Forest 內建 `predict`/offset 作為最後決策。

---

## 2. 第 7 章 Results：正文數字與來源

### 2.1 正式 test 主比較

**數字來源：`results/FINAL_TEST.md` 的主結果表。**

正文使用 AP、ROC-AUC、F1。AP 是 Average Precision，不是 accuracy，也不是梯形法 PR-AUC。表格中的數值為 mean ± sample standard deviation。

| Protocol | Tier | AP | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Exp1 | T1: V14 3-sigma | 0.5883 ± 0.0001 | 0.9563 ± 0.0000 | 0.1801 ± 0.0004 |
| Exp1 | T2: IF | 0.5861 ± 0.0154 | 0.9637 ± 0.0006 | 0.6237 ± 0.0156 |
| Exp1 | T3: LR | 0.7175 ± 0.0092 | 0.9707 ± 0.0036 | 0.8119 ± 0.0092 |
| Exp2 | T1: V14 3-sigma | 0.6146 ± 0.0000 | 0.9527 ± 0.0000 | 0.1506 ± 0.0000 |
| Exp2 | T2: IF | 0.4592 ± 0.0910 | 0.9750 ± 0.0017 | 0.5355 ± 0.0354 |
| Exp2 | T3: LR | 0.7429 ± 0.0000 | 0.9819 ± 0.0000 | 0.7812 ± 0.0000 |

對應報告敘述：

- LR 在兩種 protocol 的 AP 與 F1 都最高。
- IF 的 F1 明顯高於固定 V14 3-sigma；但 AP 在 Exp1 與 3-sigma 接近，在 Exp2 則低於 3-sigma。
- AP 衡量 anomaly ranking；F1 是 validation 選定 threshold 後的分類結果，兩者不必然同方向變動。

### 2.2 Statistical significance

**數字來源：`results/FINAL_TEST.md` 的 “Significance: IF versus strongest supervised reference LR”。**

使用 1,000 次 paired class-stratified transaction bootstrap，統計量為 `AP_IF - AP_LR`。

| Protocol | AP gap (IF − LR) | 95% percentile CI |
|---|---:|---:|
| Exp1 | -0.1314 | [-0.1854, -0.0752] |
| Exp2 | -0.2837 | [-0.3599, -0.2046] |

兩個區間都低於 0，代表在這兩個 frozen test protocols 上，LR 的 AP 高於 IF。這是針對目前固定模型與 test population 的條件式結果，不應寫成 LR 對所有資料集都必然較好。

Bootstrap 以單筆交易為抽樣單位，且沒有 card ID 可供群集抽樣；若交易之間存在時間或卡片層級相依，CI 可能偏樂觀。此限制亦記載於 `results/FINAL_TEST.md`。

### 2.3 Runtime

**數字來源：`results/FINAL_TEST.md` 的主結果表。**

下表列出 training seconds 與 preprocessing-plus-scoring latency。Latency 是一次 batch 的總時間攤提到每筆樣本，不是單筆 online request latency。

| Protocol | Tier | Training (s) | Score (ms/sample) | Preprocess + score (ms/sample) |
|---|---|---:|---:|---:|
| Exp1 | T1 | 0.0132 ± 0.0053 | 0.000142 ± 0.000041 | 0.000189 ± 0.000053 |
| Exp1 | T2 | 0.8200 ± 0.4185 | 0.007523 ± 0.000936 | 0.007561 ± 0.000941 |
| Exp1 | T3 | 0.9710 ± 0.4176 | 0.000341 ± 0.000082 | 0.000387 ± 0.000092 |
| Exp2 | T1 | 0.0104 ± 0.0013 | 0.000086 ± 0.000007 | 0.000122 ± 0.000011 |
| Exp2 | T2 | 0.5358 ± 0.0463 | 0.006476 ± 0.000181 | 0.006509 ± 0.000183 |
| Exp2 | T3 | 0.5332 ± 0.0107 | 0.000231 ± 0.000013 | 0.000265 ± 0.000016 |

### 2.4 Ablation：IF feature-set

**數字來源：`results/IF_ABLATIONS.md` 的 “Feature-set ablation”。**

Validation only；固定 200 trees、`max_samples=256`，比較 all 30 features 與固定 6 features。

| Protocol | All-feature AP | Six-feature AP | Paired ΔAP |
|---|---:|---:|---:|
| Exp1 | 0.1335 ± 0.0146 | 0.6554 | +0.5219 ± 0.0508 |
| Exp2 | 0.0190 ± 0.0021 | 0.4489 | +0.4299 ± 0.1073 |

**重要：目前 repository 的 `results/IF_ABLATIONS.md` 註明 Exp1 feature-set ablation 使用 fold 0 的 3 seeds，並非完整 5 folds；Exp2 使用 3 seeds。** 若另有更新後的 all-fold feature ablation，應先更新該結果檔與本表，再將它標成 all-fold。不要把 error analysis 的 all-fold 設定誤認為 feature-set ablation 也已經更新。

### 2.5 Ablation：IF training contamination

**數字來源：`results/IF_ABLATIONS.md` 的 “Training contamination”。**

Validation only；固定所有 normal training rows、normal-only scaler、六個 IF features、200 trees、`max_samples=256`，再加入巢狀 fraud subsets。

| Protocol | Mode | Fraud rows added | Contamination | AP | ΔAP vs C0 | F1 |
|---|---|---:|---:|---:|---:|---:|
| Exp1 | C0 | 0 | 0.0000% | 0.6148 ± 0.0548 | +0.0000 | 0.6692 |
| Exp1 | C1 | 26 | 0.0143% | 0.6223 ± 0.0482 | +0.0075 | 0.6720 |
| Exp1 | C2 | 78–79 | 0.0430% | 0.6330 ± 0.0462 | +0.0182 | 0.6807 |
| Exp1 | C3 | 157–158 | 0.0863% | 0.6461 ± 0.0429 | +0.0313 | 0.6864 |
| Exp1 | C4 | 315–316 | 0.1729% | 0.6681 ± 0.0383 | +0.0533 | 0.7062 |
| Exp2 | C0 | 0 | 0.0000% | 0.4489 ± 0.1087 | +0.0000 | 0.5388 |
| Exp2 | C1 | 30 | 0.0176% | 0.4532 ± 0.1160 | +0.0044 | 0.5407 |
| Exp2 | C2 | 90 | 0.0528% | 0.5003 ± 0.1021 | +0.0514 | 0.5671 |
| Exp2 | C3 | 180 | 0.1054% | 0.5695 ± 0.0675 | +0.1206 | 0.5997 |
| Exp2 | C4 | 360 | 0.2107% | 0.6057 ± 0.0288 | +0.1568 | 0.6144 |

這項實驗只測試約 0–0.21% 的低污染量；不能推論污染越多越好。加入 fraud 同時增加 training sample size，也改變樹的抽樣內容，因此不是完全隔離污染身分與訓練樣本數的實驗。

---

## 3. 第 8 章 Analysis：錯誤案例數據與來源

### 3.1 Error case analysis 的資料來源

錯誤案例不是從 `results/*.md` 的彙總表直接取得，而是由以下原始資料對應而來：

1. 原始特徵：`data/creditcard.csv`。
2. 正式 test predictions：`outputs/final_test/exp1/fold_<n>/tier2_isolation_forest_seed_<seed>.csv`。
3. 預測檔欄位：`row_id`, `y_true`, `score`, `threshold`, `y_pred`。
4. `error_analysis.py` 以 `row_id` 將預測列對回原始 CSV 的 row position，並檢查 `Class == y_true`，通過後才輸出特徵。

`outputs/final_test/` 可能是本機實驗產物，不一定已追蹤在 Git。若 branch 不包含原始 predictions，執行錯誤分析前需先有這些輸出檔。

### 3.2 All-fold error analysis summary

本次分析使用 Exp1 全部 5 folds × 3 seeds，共 15 組 T2 predictions。Exp1 的 15 組模型使用同一份 test set，因此以下的穩定錯誤表示同一交易在全部 15 組模型中都被誤判。

| 統計項目 | 結果 |
|---|---:|
| Unique transactions misclassified in at least one run | 143 |
| Consistent FP (all 15 runs) | 17 |
| Consistent FN (all 15 runs) | 21 |
| Consistent errors total | 38 |
| Mean FP per fold-seed run | 49.13 |
| Mean FN per fold-seed run | 31.40 |

其餘 105 筆候選交易至少在部分模型中誤判，但不是在全部 15 組模型中都出錯。平均 FP/FN 數量是每組模型的錯誤數平均，不是 unique transaction count。

### 3.3 Representative cases

下表的 score、threshold 與 margin 是同一 row ID 在 15 組模型預測中的平均值。定義：

`anomaly_margin = score - threshold`

- 正值：score 位於 anomaly 一側，預測為 fraud。
- 負值：score 位於 normal 一側，預測為 normal。
- `|margin|` 越小，越接近該次模型的 threshold。

| row_id | Type | Time | Amount | Mean score | Mean threshold | Mean margin |
|---:|---|---:|---:|---:|---:|---:|
| 57615 | FN | 47982 | 0.76 | 0.703134 | 0.714324 | -0.011190 |
| 164455 | FP | 116728 | 0.77 | 0.728830 | 0.714324 | +0.014506 |
| 13796 | FP | 24462 | 1.51 | 0.731774 | 0.714324 | +0.017450 |

以上三筆都是 15/15 次持續誤判的案例。六個 IF 輸入特徵的原始值如下：

| row_id | V14 | V10 | V12 | V4 | V11 | V17 |
|---:|---:|---:|---:|---:|---:|---:|
| 57615 (FN) | -4.3298 | -3.9424 | -3.3943 | 1.4925 | 2.4017 | -4.4802 |
| 164455 (FP) | -7.1526 | -1.7253 | -1.8237 | 4.4456 | 3.0099 | 6.1404 |
| 13796 (FP) | -3.6372 | 5.9963 | -2.7525 | 0.0110 | 4.1089 | 3.4181 |

這些特徵值可用來描述模型所接收的輸入輪廓，但 V1–V28 是轉換後的匿名化特徵，不能直接對應到具體交易行為。若要聲稱某個特徵「造成」IF 將樣本判為異常，仍需 SHAP 或其他模型歸因結果；僅憑原始特徵值不能宣稱已完成 SHAP explanation。

### 3.4 Interpretation notes

- FN 57615 是一筆實際 fraud，Amount 為 0.76。其平均 score 為 0.703134，低於平均 threshold 0.714324，margin 為 -0.011190；雖然非常接近決策邊界，但在 15 組模型中都未跨過 threshold。
- FP 164455 是正常交易，Amount 為 0.77。平均 score 高於 threshold 0.014506，且 15 組模型都將它判為 anomaly。
- FP 13796 也是正常交易，Amount 為 1.51，平均 margin 為 +0.017450，同樣在 15 組模型中持續被誤判。
- 穩定 FP 的 Amount 中位數為 89.99；穩定 FN 的 Amount 中位數為 3.79。這只是 17 筆與 21 筆穩定錯誤樣本的描述統計，樣本數很少，不能據此建立金額與誤判的因果關係。
- 由於 IF 是以六個轉換後特徵的聯合分布進行 isolation，錯誤可能來自多維組合，而非單一欄位或交易金額。此處應描述為可能的模型行為，不應當成已驗證的因果解釋。

### 3.5 Practical implication

Exp1 test results 中，IF precision 為 0.5855、recall 為 0.6796；LR precision 為 0.8148、recall 為 0.8102（來源：`results/EXPERIMENT_SUMMARY_ZH.md` 第 4 節）。IF 的誤報會增加人工覆核量，漏報則代表部分 fraud 未被攔截。實際 threshold 應依誤報與漏報的業務成本決定，而不是只追求單一 F1。

---

## 4. Reproduce error analysis

### 4.1 Run

```bash
python error_analysis.py
```

程式會自動搜尋 `fold_*/tier2_isolation_forest_seed_*.csv`，檢查 row ID 與標籤對齊，並確認 `score >= threshold` 與 `y_pred` 的規則一致。

### 4.2 Output files

| Output | Content |
|---|---|
| `case_candidates_exp1.csv` | 所有至少在一組 fold/seed 中出錯的 unique transactions，附原始 30 個特徵與跨 run 統計 |
| `case_details_exp1.csv` | 每筆候選交易在每個 fold/seed 的 score、threshold、prediction、error type，以及原始特徵 |
| `stable_error_cases_exp1.csv` | 在所有載入的 fold/seed runs 中都持續誤判的交易 |

本次正式分析載入 15 組 predictions，因此 stable FP/FN 分別代表 15/15 組模型都判錯。若之後改變 folds 或 seeds 數量，程式會依實際載入的 evaluation 數量重新定義穩定錯誤。

---

## 5. Repository source map

| Report item | Source file(s) | What to retrieve |
|---|---|---|
| Dataset、split、三個 tier 設定 | `README.md` | 共用資料接口、P0 模型設定、threshold 規則 |
| Exp1/Exp2 protocol 與樣本數 | `results/EXPERIMENT_SUMMARY_ZH.md` §2 | 切分方式、train/validation/test 數量 |
| 正式 test AP / ROC-AUC / F1 | `results/FINAL_TEST.md` 主表 | 第 7 章主比較表 |
| Bootstrap AP gap 與 95% CI | `results/FINAL_TEST.md` Significance | IF vs LR 的 paired bootstrap |
| Training time、inference latency | `results/FINAL_TEST.md` 主表 | 第 7 章 runtime table |
| Feature-set ablation | `results/IF_ABLATIONS.md` Feature-set ablation | all-30 vs six-feature validation AP |
| Contamination ablation | `results/IF_ABLATIONS.md` Training contamination | C0–C4 validation AP/F1 |
| Exp1 test precision/recall | `results/EXPERIMENT_SUMMARY_ZH.md` §4 | IF/LR precision、recall 與主結果 |
| FP/FN row IDs、scores、thresholds、raw features | `outputs/final_test/exp1/fold_*/tier2_isolation_forest_seed_*.csv` + `data/creditcard.csv` | 由 `error_analysis.py` 還原 |
| Error analysis derived tables | `error_case_analysis/*.csv` | 案例統計 |

