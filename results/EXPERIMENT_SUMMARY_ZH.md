# 信用卡異常偵測：實驗結果總整理

更新日期：2026-11-04。這份文件彙整原始 **test 20%** 的正式實驗，以及新版 Exp1 先採樣再切分的追加探索，方便組員寫報告與製作簡報。
第 1–8 節為原始正式實驗及其 ablations；第 9 節使用不同資料分布，不能與正式 test 直接比較。
不包含已還原的「先平衡資料再切分」及關閉 class weight 的分支。

## 1. 先看結論

- 正式 test 上，LR 的 AP、F1 在兩套切分都最高；配對 bootstrap 的 IF−LR AP 差異信賴區間皆低於 0。
- 六特徵 IF 的決策 F1 優於固定 V14 3σ，但 AP 沒有全面優於 3σ。AP 衡量排序，F1 衡量指定門檻的決策。
- T3 的 SMOTE（P5）在 validation 上維持接近 P0 的 AP，F1 稍高；差距未證明具統計顯著性。正式 test 的 T3 保留 P0。
- 強烈 1:1 undersampling 多數降低 validation 表現；採樣 seed 對部分設定有很大影響。
- 原始切分的 IF 在很低的訓練污染量下 AP 反而上升；這是敏感度實驗結果，主比較仍採 normal-only IF。
- 新版 Exp1 八組模式內，LR 的 test AP 都最高；其 test 已先 undersampling，不能將較高分數解讀為比正式實驗更好。
- 新 P1-10 的 T2 污染實驗中，C1/C2 接近 C0；C4 test AP 從 0.9163 降至 0.9034，尚未做顯著性分析。

## 2. 資料與切分

資料為 `data/creditcard.csv`，共 284,807 筆、492 筆 fraud、30 個特徵。
偵測單位是一筆交易，Class=0 為正常，Class=1 為 fraud。

| 實驗 | 切分方式 | 原始 train / validation / test |
| --- | --- | --- |
| exp1 | 固定 20% stratified holdout test；剩餘 80% 做 stratified 5-fold | 每 fold 約 64% / 16% / 20% |
| exp2 | 依 Time 排序後切分 | 60% / 20% / 20% |

| 實驗 | Fold | Partition | 筆數 | 正常 | Fraud |
| --- | --- | --- | --- | --- | --- |
| exp1 | fold 0 | train | 182,276 | 181,960 | 316 |
| exp1 | fold 0 | val | 45,569 | 45,491 | 78 |
| exp1 | fold 0 | test | 56,962 | 56,864 | 98 |
| exp1 | fold 1-4 (each) | train | 182,276 | 181,961 | 315 |
| exp1 | fold 1-4 (each) | val | 45,569 | 45,490 | 79 |
| exp1 | fold 1-4 (each) | test | 56,962 | 56,864 | 98 |
| exp2 | single split | train | 170,884 | 170,524 | 360 |
| exp2 | single split | val | 56,961 | 56,904 | 57 |
| exp2 | single split | test | 56,962 | 56,887 | 75 |

原始 split seed 固定 42；主比較與採樣的模型 seeds 為 42、43、44。
exp1 每個 fold 各訓練三個 seeds，所有 folds 共用同一個 test；exp2 僅一個 chronological split。
Undersampling、IQR、SMOTE 只改變 training，validation/test 維持原分布。

## 3. 三個模型、前處理與門檻

| Tier | 模型 | 參數估計使用的資料 | 特徵與異常分數 | 決策門檻 |
| --- | --- | --- | --- | --- |
| T1 | 3σ | 只用正常 training | 固定 V14；abs((V14−μ)/σ)，σ 使用 ddof=0 | 固定 score ≥ 3 |
| T2 | Isolation Forest | 只用正常 training | V14、V10、V12、V4、V11、V17；負的 sklearn score_samples，越高越異常 | validation 最大 F1；同分取較高門檻 |
| T3 | Logistic Regression | 正常＋fraud training | 全 30 特徵；fraud 機率 | 同上，以 validation 選門檻 |

- IF：200 trees、max_samples=256、contamination="auto"、n_jobs=1。自行選門檻，不使用 sklearn 內建 predict/offset 做決策。
- LR：C=1、L2、class_weight="balanced"、solver="lbfgs"、max_iter=2000；主比較尚未做 C/penalty 搜尋。
- Time、Amount 使用 training-fitted RobustScaler，**沒有 log1p**；V1–V28 不變。T1/T2 的 scaler 只 fit 正常 training，T3 fit 完整 training。T3 採樣後不重 fit scaler。
- 目前 T1/T2 只用 V 特徵，因此 Time/Amount 的 scaler 實際只影響 T3 的模型分數。
- V14 與六特徵由團隊既有 EDA 指定，未重算 KS。既有 `feature_analysis.py` 主流程使用完整資料，因此報告不能把這個特徵選擇步驟描述成完全只看 training；需揭露這項限制。

## 4. 主比較：train、validation、test 結果

**各區分開算指標；不把不同分數尺度的模型或不同 runs 的分數直接混在一起。**
exp1 先平均每個 fold 內三個 seeds，再報告五個 folds 的 mean ± sample std；exp2 報告三個 seeds 的 mean ± sample std。
表格中 AP、F1 帶標準差，ROC-AUC、precision、recall 列平均值；完整逐 run 統計見來源 CSV。

Train 指標是在**完整原始 training partition（含兩類）**上補算，使用已保存的模型和門檻，沒有重新訓練。
T1/T2 雖然在這裡評估 training fraud，但模型參數仍只 fit 正常 training；這不代表 fraud 曾用於 IF 訓練。
此處門檻含 validation 的資訊，所以 train 指標只是事後描述，不能視為獨立驗證。
LR 的 train 指標為 in-sample；validation F1 也因同一集合選門檻而較樂觀。正式泛化結論以 test 為主。

### exp1：分層隨機切分

| 模型 | Partition | AP | F1 | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| T1 V14 3σ | train | 0.5906 ± 0.0084 | 0.1766 ± 0.0027 | 0.9517 | 0.0988 | 0.8325 |
| T1 V14 3σ | val | 0.5970 ± 0.0340 | 0.1767 ± 0.0087 | 0.9518 | 0.0988 | 0.8326 |
| T1 V14 3σ | test | 0.5883 ± 0.0001 | 0.1801 ± 0.0004 | 0.9563 | 0.1004 | 0.8776 |
| T2 六特徵 IF | train | 0.6040 ± 0.0127 | 0.6294 ± 0.0087 | 0.9650 | 0.6423 | 0.6263 |
| T2 六特徵 IF | val | 0.6110 ± 0.0348 | 0.6573 ± 0.0209 | 0.9645 | 0.6707 | 0.6547 |
| T2 六特徵 IF | test | 0.5861 ± 0.0154 | 0.6237 ± 0.0156 | 0.9637 | 0.5855 | 0.6796 |
| T3 LR | train | 0.7502 ± 0.0175 | 0.8036 ± 0.0111 | 0.9901 | 0.8516 | 0.7614 |
| T3 LR | val | 0.7477 ± 0.0559 | 0.8190 ± 0.0145 | 0.9826 | 0.8629 | 0.7817 |
| T3 LR | test | 0.7175 ± 0.0092 | 0.8119 ± 0.0092 | 0.9707 | 0.8148 | 0.8102 |

### exp2：時間順序切分

| 模型 | Partition | AP | F1 | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| T1 V14 3σ | train | 0.6074 ± 0.0000 | 0.1940 ± 0.0000 | 0.9567 | 0.1094 | 0.8528 |
| T1 V14 3σ | val | 0.4841 ± 0.0000 | 0.1020 ± 0.0000 | 0.9411 | 0.0544 | 0.8070 |
| T1 V14 3σ | test | 0.6146 ± 0.0000 | 0.1506 ± 0.0000 | 0.9527 | 0.0830 | 0.8133 |
| T2 六特徵 IF | train | 0.6219 ± 0.0361 | 0.6395 ± 0.0312 | 0.9651 | 0.5935 | 0.6935 |
| T2 六特徵 IF | val | 0.4489 ± 0.1087 | 0.5388 ± 0.0631 | 0.9508 | 0.5372 | 0.5439 |
| T2 六特徵 IF | test | 0.4592 ± 0.0910 | 0.5355 ± 0.0354 | 0.9750 | 0.5477 | 0.5244 |
| T3 LR | train | 0.7477 ± 0.0000 | 0.8223 ± 0.0000 | 0.9869 | 0.8491 | 0.7972 |
| T3 LR | val | 0.7727 ± 0.0000 | 0.7921 ± 0.0000 | 0.9709 | 0.9091 | 0.7018 |
| T3 LR | test | 0.7429 ± 0.0000 | 0.7812 ± 0.0000 | 0.9819 | 0.9434 | 0.6667 |

T1 與 lbfgs LR 在同一切分下為確定性方法，跨 seed 標準差可能為 0；這不代表對新資料沒有不確定性。
exp1 各 run 的 test 資料相同，因此標準差主要反映訓練模型差異，不是五份獨立 test 的變異。

## 5. AP、F1 與表格的解讀

- **AP（Average Precision）**：以 fraud 為正類，根據連續分數計算 Σ(Recall 增量 × Precision)，衡量 fraud 是否排在高分位置。不是 accuracy，也不是梯形法 PR-AUC。
- **Precision**：警報中真正 fraud 的比例；**Recall**：全部 fraud 中找回的比例。
- **F1**：指定門檻下 precision 與 recall 的調和平均；AP 不依賴單一門檻，兩者排名可能不同。
- **ROC-AUC**：整體排序的另一指標；在極端不平衡下，即使很高，實際 precision/AP 仍可能低。
- **FPR**：正常交易被誤報的比例；混淆矩陣 TN/FP/FN/TP 保存在逐 run 的 metrics CSV。
- 「±」是上節定義的標準差；下節 bootstrap CI 是另一種不確定性，不能混用。

## 6. 正式 test 的時間成本與顯著性

| 實驗 | 模型 | 訓練秒數 mean ± std | 推論 ms/筆 | 前處理＋推論 ms/筆 |
| --- | --- | --- | --- | --- |
| exp1 | T1 V14 3σ | 0.0132 ± 0.0053 | 0.000142 | 0.000189 |
| exp1 | T2 六特徵 IF | 0.8200 ± 0.4185 | 0.007523 | 0.007561 |
| exp1 | T3 LR | 0.9710 ± 0.4176 | 0.000341 | 0.000387 |
| exp2 | T1 V14 3σ | 0.0104 ± 0.0013 | 0.000086 | 0.000122 |
| exp2 | T2 六特徵 IF | 0.5358 ± 0.0463 | 0.006476 | 0.006509 |
| exp2 | T3 LR | 0.5332 ± 0.0107 | 0.000231 | 0.000265 |

訓練時間只涵蓋原本保存的模型 fit；推論時間是本次 test 的單次批次執行、每筆攤提值，不是線上單筆請求延遲。
不含 CSV IO、模型載入；前處理＋推論欄位包含 scaler transform。不同時間測量會受機器負載影響。

以 1,000 次 **paired、按類別分層的交易 bootstrap** 比較 IF 與最強參考 LR。
每次對所有模型使用相同抽樣交易，維持類別筆數；統計量為平均逐 run AP 的差異。
模型和 validation 門檻保持固定，沒有把相關的 fold/seed runs 當作獨立樣本重抽。

| 實驗 | AP 差異 IF−LR | 95% CI |
| --- | --- | --- |
| exp1 | -0.1314 | [-0.1854, -0.0752] |
| exp2 | -0.2837 | [-0.3599, -0.2046] |

兩個 CI 均完全低於 0，支持 LR 在目前固定協定與 test 族群下的 AP 高於 IF。
分析假設交易彼此獨立；缺少 card IDs，無法做按卡片分群的 bootstrap。時間／卡片關聯可能使 CI 偏樂觀。
結果條件化於已訓練模型、觀察到的類別比例，不能解讀為 LR 在所有資料集都優於 IF。

## 7. T3 P0–P6：採樣比較（validation）

固定 LR 的 C=1、balanced、全特徵與切分。P1–P4 各比較 normal:fraud=1:1、5:1、10:1；P5/P6 的 SMOTE fraud/normal=0.01。

| 模式 | training 處理 |
| --- | --- |
| P0 | 不採樣、不刪離群值 |
| P1 | 正常交易隨機、不重複 undersampling；fraud 全保留 |
| P2 | 正常交易 V10 的 1.5×IQR 過濾，再 random undersampling |
| P3 | Candidate-based K-Medoids undersampling |
| P4 | 同樣 IQR 過濾，再 K-Medoids |
| P5 | SMOTE |
| P6 | 同樣 IQR 過濾，再 SMOTE |

K-Medoids 先抽約目標筆數 1.25 倍、上限 5,000 的正常候選池，使用處理後 30 特徵的歐氏距離、alternate、k-medoids++、最多 20 次迭代，保留實際 medoids；不是對全量正常資料分群。
IQR 範圍只用正常 training 計算，fraud 不因 IQR 被移除。
三個採樣 seeds 共完成 exp1 225 次、exp2 45 次訓練（皆含 P0）；exp1 folds 與 seed 的統計方式同主比較。

| Mode | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| P0 | 0.7477 ± 0.0559 | 0.8190 ± 0.0145 | 0.7727 ± 0.0000 | 0.7921 ± 0.0000 |
| P1-1 | 0.6456 ± 0.0438 | 0.7136 ± 0.0504 | 0.3815 ± 0.0553 | 0.5454 ± 0.0191 |
| P1-5 | 0.7169 ± 0.0620 | 0.7865 ± 0.0250 | 0.5973 ± 0.2836 | 0.6918 ± 0.2011 |
| P1-10 | 0.7380 ± 0.0555 | 0.8065 ± 0.0172 | 0.7110 ± 0.0485 | 0.7653 ± 0.0519 |
| P2-1 | 0.6200 ± 0.0978 | 0.6936 ± 0.0701 | 0.2511 ± 0.0737 | 0.4544 ± 0.0463 |
| P2-5 | 0.7148 ± 0.0588 | 0.7913 ± 0.0197 | 0.5439 ± 0.1388 | 0.6699 ± 0.0604 |
| P2-10 | 0.7309 ± 0.0584 | 0.7978 ± 0.0195 | 0.7504 ± 0.0185 | 0.8016 ± 0.0179 |
| P3-1 | 0.6504 ± 0.0488 | 0.7123 ± 0.0203 | 0.2341 ± 0.0822 | 0.4439 ± 0.0789 |
| P3-5 | 0.7236 ± 0.0556 | 0.7908 ± 0.0098 | 0.5551 ± 0.2183 | 0.6629 ± 0.1371 |
| P3-10 | 0.7334 ± 0.0624 | 0.7971 ± 0.0168 | 0.7408 ± 0.0358 | 0.7743 ± 0.0462 |
| P4-1 | 0.6620 ± 0.0487 | 0.7239 ± 0.0289 | 0.2325 ± 0.0629 | 0.4651 ± 0.0684 |
| P4-5 | 0.7161 ± 0.0664 | 0.7875 ± 0.0260 | 0.7466 ± 0.0439 | 0.7998 ± 0.0351 |
| P4-10 | 0.7325 ± 0.0580 | 0.8030 ± 0.0189 | 0.7520 ± 0.0146 | 0.8027 ± 0.0159 |
| P5 | 0.7465 ± 0.0510 | 0.8213 ± 0.0166 | 0.7726 ± 0.0034 | 0.8121 ± 0.0041 |
| P6 | 0.7228 ± 0.0387 | 0.8114 ± 0.0147 | 0.6954 ± 0.0366 | 0.7965 ± 0.0202 |

這些模式目前**只有 validation 結果**；沒有對 P1–P6 補跑 train/test 指標，也沒有把它們當成正式 test 主模型。
避免將缺少的 test 結果誤認為 0，或將 validation 最佳 F1 當作 test 效能。
P0 與 P5 的 AP 很接近；部分 undersampling 模式有明顯 seed 波動，不宜只挑某一次較好的結果。

## 8. IF controlled ablations（validation）

### 8.1 特徵集合與 max_samples

下表僅 exp1 fold 0 與 exp2，各三個 seeds，± 為跨 seed 標準差；與主比較 exp1 的跨 fold 標準差不同。
每個變體相對 baseline 只改一項設定。六特徵是固定既有集合，不重新挑 KS。

| 設定 | exp1 AP | exp2 AP |
| --- | --- | --- |
| 全 30 特徵，256 samples | 0.1335 ± 0.0146 | 0.0190 ± 0.0021 |
| 移除 Time | 0.1058 ± 0.0182 | 0.0202 ± 0.0021 |
| 只用 V1–V28 | 0.1454 ± 0.0312 | 0.0251 ± 0.0051 |
| V14-only | 0.6561 ± 0.0099 | 0.4655 ± 0.0406 |
| 固定六特徵 | 0.6554 ± 0.0543 | 0.4489 ± 0.1087 |
| 全特徵，1024 samples | 0.1352 ± 0.0245 | 0.0242 ± 0.0009 |
| 全特徵，4096 samples | 0.2190 ± 0.0131 | 0.0347 ± 0.0030 |

特徵集合的影響遠大於單純移除 Time 或增加 max_samples；因此主 T2 採固定六特徵。
這支持全特徵 IF 的異常排序偏向其他稀有正常模式，但不等於已確認單一特徵的因果責任。
V14-only 也很強，不能宣稱六特徵版全面較好；保留為 ablation。

### 8.2 訓練污染量 C0–C4

固定六特徵、200 trees、256 samples/tree、全部原始正常 training 及 normal-only scaler，只額外放入巢狀 fraud 子集。
此處刻意直接 fit 所有選入樣本，避免 normal-only wrapper 將污染 fraud 移除。
C0 為無污染主設定；C1–C4 僅為敏感度研究，不替換正式主比較。

| 實驗 | Mode | 混入 fraud 筆數 | 平均污染率 | AP | ΔAP vs C0 | F1 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 | C0 | 0 | 0.0000% | 0.6148 ± 0.0548 | +0.0000 | 0.6692 |
| exp1 | C1 | 26 | 0.0143% | 0.6223 ± 0.0482 | +0.0075 | 0.6720 |
| exp1 | C2 | 78–79 | 0.0430% | 0.6330 ± 0.0462 | +0.0182 | 0.6807 |
| exp1 | C3 | 157–158 | 0.0863% | 0.6461 ± 0.0429 | +0.0313 | 0.6864 |
| exp1 | C4 | 315–316 | 0.1729% | 0.6681 ± 0.0383 | +0.0533 | 0.7062 |
| exp2 | C0 | 0 | 0.0000% | 0.4489 ± 0.1087 | +0.0000 | 0.5388 |
| exp2 | C1 | 30 | 0.0176% | 0.4532 ± 0.1160 | +0.0044 | 0.5407 |
| exp2 | C2 | 90 | 0.0528% | 0.5003 ± 0.1021 | +0.0514 | 0.5671 |
| exp2 | C3 | 180 | 0.1054% | 0.5695 ± 0.0675 | +0.1206 | 0.5997 |
| exp2 | C4 | 360 | 0.2107% | 0.6057 ± 0.0288 | +0.1568 | 0.6144 |

此表 exp1 統計涵蓋 5 folds × 3 seeds，± 是這 15 個相關 runs 的描述性標準差；exp2 為 3 seeds。
ΔAP 對同 fold/seed 的本實驗 C0 計算。本實驗固定資料列順序，與較早 exp1 P0 順序不同，C0 數字不必完全相同。
最高實際污染率僅約 0.17%／0.21%，不是 1%／5%／10%。小量污染 AP 上升，不能推論更高污染也有益。
加入 fraud 也增加 training 筆數並改變各樹抽樣；此設計測的是「固定正常資料、額外加入 fraud」的效果，未單獨隔離 training size 的影響。

## 9. 新版 Exp1：P1–P4 三個 tiers 與 T2 污染實驗（追加探索）

日期：2026-11-03。直接使用組員 `data_preparation_exp1_updated_v2.get_prepared_data`，seed=42、fold_index=None。
**P1–P4 三個 tiers 比較使用外層 60/20/20 的單次完整 train/validation/test，不是內層五折 CV，也不是三個 seeds 的平均。**
另於 2026-11-04 追加 T2 P1-10 的 C0–C4 污染實驗，使用 seeds 42／43／44，結果整合於本文後段。
8 組模式 × 3 模型＝24 個已訓練模型，72 列 partition 指標。

### 模型設定

- T1：固定 V14、雙側 3σ，僅 normal training 估計 μ/σ，固定門檻 3。
- T2：固定 V14/V10/V12/V4/V11/V17，normal-only IF，200 trees、max_samples=256、seed42。
- T3：全 30 特徵 LR，C=1、L2、balanced、lbfgs、max_iter=2000，使用兩類 training。
- T2/T3 使用 validation 最大 F1 決定門檻，凍結後評估 test；沒有依 test 更改模型。
- 所有模型直接使用接口提供的資料；Time/Amount scaler fit 當次 training，兩個 normal-only 模型不使用 Time/Amount，故這兩欄的 scaler 不影響其分數。

### 模式與切分

P1：切分前 random undersampling 到 1/5/10:1。P2：切分前正常資料 V10 IQR 過濾，再做同樣 undersampling。
P3-10：切分前 random 10:1，切分後 training 用 K-Medoids 減到約 5:1。P4-10：先加 IQR，再做同樣 P3 流程。
P3/P4 目前只有 -10 兩組。K-Medoids 為既有 candidate-based 方法。

| 模式 | Train 正常 / fraud | Validation 正常 / fraud | Test 正常 / fraud |
| --- | --- | --- | --- |
| P1-1 | 295 / 295 | 98 / 99 | 99 / 98 |
| P1-5 | 1476 / 295 | 492 / 98 | 492 / 99 |
| P1-10 | 2952 / 295 | 984 / 98 | 984 / 99 |
| P2-1 | 295 / 295 | 98 / 99 | 99 / 98 |
| P2-5 | 1476 / 295 | 492 / 98 | 492 / 99 |
| P2-10 | 2952 / 295 | 984 / 98 | 984 / 99 |
| P3-10 | 1475 / 295 | 984 / 98 | 984 / 99 |
| P4-10 | 1475 / 295 | 984 / 98 | 984 / 99 |

### 結果：AP / F1

每格為 AP / F1。AP 為 Average Precision，非 accuracy；完整 ROC-AUC、precision/recall、FPR、accuracy、混淆矩陣與時間見 CSV。

#### Test

| 模式 | T1：AP / F1 | T2：AP / F1 | T3：AP / F1 |
| --- | --- | --- | --- |
| P1-1 | 0.9502 / 0.8800 | 0.9621 / 0.9011 | 0.9770 / 0.9130 |
| P1-5 | 0.9099 / 0.8601 | 0.9095 / 0.8729 | 0.9384 / 0.9271 |
| P1-10 | 0.8873 / 0.8342 | 0.9145 / 0.8889 | 0.9404 / 0.9032 |
| P2-1 | 0.9507 / 0.8876 | 0.9676 / 0.9130 | 0.9784 / 0.9061 |
| P2-5 | 0.9342 / 0.8973 | 0.9280 / 0.8557 | 0.9492 / 0.9175 |
| P2-10 | 0.9102 / 0.8691 | 0.9310 / 0.8854 | 0.9398 / 0.9158 |
| P3-10 | 0.8868 / 0.8394 | 0.9135 / 0.8783 | 0.9397 / 0.9071 |
| P4-10 | 0.9099 / 0.8830 | 0.9304 / 0.9130 | 0.9375 / 0.9032 |

#### Validation

| 模式 | T1：AP / F1 | T2：AP / F1 | T3：AP / F1 |
| --- | --- | --- | --- |
| P1-1 | 0.9687 / 0.9255 | 0.9839 / 0.9436 | 0.9826 / 0.9543 |
| P1-5 | 0.9222 / 0.8783 | 0.9221 / 0.8901 | 0.9666 / 0.9189 |
| P1-10 | 0.8229 / 0.8085 | 0.8628 / 0.8587 | 0.8969 / 0.8796 |
| P2-1 | 0.9698 / 0.9355 | 0.9816 / 0.9479 | 0.9858 / 0.9583 |
| P2-5 | 0.9322 / 0.9071 | 0.9400 / 0.9062 | 0.9699 / 0.9305 |
| P2-10 | 0.8501 / 0.8280 | 0.8873 / 0.8723 | 0.9163 / 0.8947 |
| P3-10 | 0.8224 / 0.8108 | 0.8642 / 0.8541 | 0.9046 / 0.8840 |
| P4-10 | 0.8496 / 0.8370 | 0.8891 / 0.8778 | 0.9189 / 0.8936 |

#### Train

| 模式 | T1：AP / F1 | T2：AP / F1 | T3：AP / F1 |
| --- | --- | --- | --- |
| P1-1 | 0.9708 / 0.9051 | 0.9745 / 0.9220 | 0.9938 / 0.9577 |
| P1-5 | 0.9115 / 0.8774 | 0.9336 / 0.8950 | 0.9728 / 0.9261 |
| P1-10 | 0.9002 / 0.8456 | 0.9100 / 0.8835 | 0.9644 / 0.9171 |
| P2-1 | 0.9747 / 0.9094 | 0.9695 / 0.9187 | 0.9942 / 0.9520 |
| P2-5 | 0.9170 / 0.8826 | 0.9339 / 0.8893 | 0.9724 / 0.9303 |
| P2-10 | 0.9129 / 0.8567 | 0.9252 / 0.8816 | 0.9660 / 0.9151 |
| P3-10 | 0.9199 / 0.8803 | 0.9212 / 0.8901 | 0.9733 / 0.9256 |
| P4-10 | 0.9304 / 0.8912 | 0.9387 / 0.9137 | 0.9758 / 0.9321 |

### 如何解讀

- LR 的 test AP 在八組內都最高；F1 多數也較高，但 P2-1、P4-10 是 IF 稍高。這是單次觀察，未做顯著性推論。
- 同一模式的三個 tier 使用相同 validation/test IDs；P1-10 與 P3-10、P2-10 與 P4-10 的 outer validation/test IDs 也已確認一致。
- 不同採樣比例的 test 族群與 fraud 比例不同，不應將跨比例的 AP/F1 差異直接視為方法改善。
- P2/P4 的 IQR 在切分前估計並過濾資料，不屬於完全只看 training 的預處理；可能影響評估分布。
- Train 指標是在該模式最終完整 training（兩類）上計算，T1/T2 實際 fit 仍只用正常列。
- Validation F1 同時用於挑門檻，可能偏樂觀；test 已按凍結門檻評估。
- 原本 FINAL_TEST 的固定 raw split 比較保持不變，這組屬於另一個探索協定。

### 檔案

- 程式：`run_new_exp1_tiers.py`；執行 `uv run --frozen python run_new_exp1_tiers.py`。
- 完整指標：[new_exp1_tiers_metrics.csv](new_exp1_tiers_metrics.csv)。
- 切分筆數：[new_exp1_tiers_split_counts.csv](new_exp1_tiers_split_counts.csv)。
- 設定：[new_exp1_tiers_protocol.json](new_exp1_tiers_protocol.json)。
- 模型、scaler、split IDs 與逐筆預測：`outputs/new_exp1_tiers/seed_42/<mode>/`；程式拒絕覆寫同名輸出目錄。

### T2：P1-10 下的 C0–C4 contamination ablation

固定 `data_preparation_exp1_updated_v2.py` 的 P1-10 outer 60/20/20、split seed 42；不使用 inner CV。模型／污染子集 seeds 為 42、43、44，共 15 次訓練。

#### 設定

- 六特徵 V14、V10、V12、V4、V11、V17；200 trees、max_samples=256、n_jobs=1。
- 固定 2,952 筆 normal 訓練資料，C0–C4 加入 0、24、73、147、295 筆 fraud；各 seed 的 fraud 子集巢狀包含。
- 直接以 sklearn IF 訓練所有選入列，刻意保留污染 fraud。sklearn contamination=auto；此參數不等於實際污染比例。
- train 評估使用完整固定 P1-10 train（2,952 normal / 295 fraud），不是各 C 模式的 fit 子集。
- validation 固定 984 normal / 98 fraud；test 固定 984 normal / 99 fraud。
- 沿用 P1-10 的 train-only Time/Amount scaler；IF 所選六個 V 特徵不受該 scaler 影響。
- 列排序統一 normal 後 fraud，各自依原始 ID；因此 C0 與先前 runner 的 P1-10 T2 不保證完全相同，差異以本次 C0 為準。
- 各模型依 validation 最大 F1 選門檻，凍結後評估 train/test。AP 為 sklearn average precision。

#### 三 seeds 結果

數值為跨 seeds mean ± sample std；非信賴區間。

##### train

| 模式 | 訓練 fraud | 實際污染率 | AP | F1 | AP 相對 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 0 | 0.000% | 0.9139 ± 0.0061 | 0.8838 ± 0.0131 | +0.0000 |
| C1 | 24 | 0.806% | 0.9171 ± 0.0014 | 0.8840 ± 0.0057 | +0.0032 |
| C2 | 73 | 2.413% | 0.9140 ± 0.0007 | 0.8766 ± 0.0029 | +0.0001 |
| C3 | 147 | 4.743% | 0.9085 ± 0.0005 | 0.8714 ± 0.0034 | -0.0054 |
| C4 | 295 | 9.085% | 0.8884 ± 0.0030 | 0.8519 ± 0.0033 | -0.0255 |

##### val

| 模式 | 訓練 fraud | 實際污染率 | AP | F1 | AP 相對 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 0 | 0.000% | 0.8666 ± 0.0011 | 0.8572 ± 0.0062 | +0.0000 |
| C1 | 24 | 0.806% | 0.8703 ± 0.0040 | 0.8576 ± 0.0033 | +0.0038 |
| C2 | 73 | 2.413% | 0.8665 ± 0.0008 | 0.8556 ± 0.0027 | -0.0001 |
| C3 | 147 | 4.743% | 0.8625 ± 0.0009 | 0.8498 ± 0.0024 | -0.0040 |
| C4 | 295 | 9.085% | 0.8489 ± 0.0056 | 0.8324 ± 0.0045 | -0.0177 |

##### test

| 模式 | 訓練 fraud | 實際污染率 | AP | F1 | AP 相對 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 0 | 0.000% | 0.9163 ± 0.0021 | 0.8913 ± 0.0071 | +0.0000 |
| C1 | 24 | 0.806% | 0.9173 ± 0.0014 | 0.8957 ± 0.0030 | +0.0010 |
| C2 | 73 | 2.413% | 0.9167 ± 0.0029 | 0.8913 ± 0.0106 | +0.0005 |
| C3 | 147 | 4.743% | 0.9118 ± 0.0012 | 0.8945 ± 0.0069 | -0.0044 |
| C4 | 295 | 9.085% | 0.9034 ± 0.0039 | 0.8806 ± 0.0061 | -0.0128 |

#### 解讀與限制

Test C1/C2 的 AP 接近 C0；C3/C4 較低，C4 平均 AP 減少 0.0128。尚未執行顯著性分析，不能把微小差異稱為顯著提升或下降。

這組 P1-10 的實際污染率最高約 9.09%，與原始資料 ablation 的低污染比例不同，不能只按 C 編號直接對照。污染時正常列固定、總訓練筆數隨 fraud 增加，因此也包含訓練總量改變的效果。

此 test 來自先對全資料 undersampling 再切分，fraud 比例約 9.14%，不是原始交易分布。僅在本次固定 test 內比較各 C 模式；不可據此宣稱比原始正式 test 表現更佳。本 test 先前已評估過，屬追加探索，不是新的獨立確認。

#### 重現與檔案

```powershell
uv run --frozen python run_new_exp1_if_contamination.py
```

預設輸出目錄已存在時會拒絕覆寫。模型、fit row IDs、固定 split IDs、各分區 predictions 在 `outputs/new_exp1_if_contamination/`。可分享的 metrics、summary、protocol、split counts 在本目錄 `new_exp1_if_contamination_*`。

驗證：整合測試通過，檢查 fraud 真正進入 fit、子集巢狀包含、test IDs/labels 固定及門檻跨分區固定。

## 10. 已完成範圍與交接注意事項

- 已完成兩套正式切分、三個 tiers、多 seeds、T3 採樣、IF ablations、主比較 test、時間成本與顯著性分析。
- 正式 test 使用保存模型與 validation 門檻，沒有依 test 重新訓練或調門檻。
- 未做 test 錯誤案例／解釋性研究，這部分不在目前約定的實驗分工範圍內。
- 不同 experiment 的 train/val/test 族群不同，不把 exp1、exp2 混成同一套公平比較。
- 原資料的 PCA 特徵 V1–V28 是資料集已提供的，不是這個專案重新 fit 的 PCA。
- 補算的 train 指標不改變模型或正式 test 結果。T3 仍使用 balanced；已還原的分支實驗不列入本文件。

## 11. 數據與程式索引

| 內容 | 檔案 |
| --- | --- |
| 主比較 train 逐 run | [train_metrics.csv](train_metrics.csv) |
| Train／validation／test 統一彙整 | [train_val_test_summary.csv](train_val_test_summary.csv) |
| 主比較 validation 逐 run | [p0_exp1_metrics.csv](p0_exp1_metrics.csv)、[p0_exp2_metrics.csv](p0_exp2_metrics.csv) |
| 正式 test 逐 run／彙整 | [final_test_metrics.csv](final_test_metrics.csv)、[final_test_summary.csv](final_test_summary.csv) |
| 凍結設定／顯著性 | [final_test_protocol.json](final_test_protocol.json)、[final_test_significance.csv](final_test_significance.csv) |
| 主比較 seed 統計 | [P0_SEEDS.md](P0_SEEDS.md)、[p0_within_fold_seed_summary.csv](p0_within_fold_seed_summary.csv) |
| 採樣結果 | [T3_SAMPLING.md](T3_SAMPLING.md)、t3_sampling_exp1/exp2_metrics.csv |
| IF 診斷 | [if_diagnostics_metrics.csv](if_diagnostics_metrics.csv) |
| IF 污染 | [IF_ABLATIONS.md](IF_ABLATIONS.md)、if_contamination_exp1/exp2_metrics.csv |
| 新版 Exp1 三 tiers | [new_exp1_tiers_metrics.csv](new_exp1_tiers_metrics.csv)、[new_exp1_tiers_split_counts.csv](new_exp1_tiers_split_counts.csv)、[new_exp1_tiers_protocol.json](new_exp1_tiers_protocol.json) |
| 新 P1-10 T2 污染 | [new_exp1_if_contamination_metrics.csv](new_exp1_if_contamination_metrics.csv)、[new_exp1_if_contamination_summary.csv](new_exp1_if_contamination_summary.csv)、[new_exp1_if_contamination_protocol.json](new_exp1_if_contamination_protocol.json)、[new_exp1_if_contamination_split_counts.csv](new_exp1_if_contamination_split_counts.csv) |
| 新版 Exp1 執行程式 | ../run_new_exp1_tiers.py、../run_new_exp1_if_contamination.py |
| 訓練／正式評估程式 | ../run_experiment.py、../evaluate_final_test.py |
| 採樣／污染程式 | ../run_t3_sampling.py、../complete_t3_seeds.py、../run_if_contamination.py |

完整模型與逐筆預測保存在本機 `outputs/P0/`、`outputs/T3_sampling/`、`outputs/final_test/` 等目錄，未納入 Git。
上述 CSV 與本文件可供交接；只有 repository 並不包含本機模型檔。
