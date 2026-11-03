# 實驗進度摘要（2026-11-02）

2026-11-03：test 10% 的 T3 P0–P6 追加探索已完成，三個 seeds，
exp1 約 72/18/10、exp2 70/20/10；僅比較 validation，見 [T3_TEST10.md](T3_TEST10.md)。
validation 樣本隨切分改變，不與原始結果做配對顯著性比較；原正式 test 結果保留。

以下保留 validation 階段結果；正式 test 主結果、時間成本與顯著性分析已完成，見 [FINAL_TEST.md](FINAL_TEST.md)。
IF contamination 與特徵 ablation 已完成，詳見 [IF_ABLATIONS.md](IF_ABLATIONS.md)。
T3 P0–P6 已完成 exp1 五個 folds 與 exp2，詳見 [T3_SAMPLING.md](T3_SAMPLING.md)。
AP 為 Average Precision。T2/T3 的 F1 門檻由同一 validation 選出，並非獨立泛化估計。

## 目前 P0（T2 固定六特徵）

T1 固定 V14 雙側 3σ，μ/σ 僅用正常 training 估計；不重新計算 KS。
此表 T2 為固定 V14、V10、V12、V4、V11、V17 的 IF（200 trees、256 samples/tree）。
T3 為全 30 特徵 LR（C=1、balanced、lbfgs）。
T1/T2 只用正常 training；T3 使用兩類 training 標籤。

| 模型 | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| V14 3σ | 0.5970 ± 0.0340 | 0.1767 ± 0.0087 | 0.4841 ± 0.0000 | 0.1020 ± 0.0000 |
| IF（六特徵） | 0.6110 ± 0.0348 | 0.6573 ± 0.0209 | 0.4489 ± 0.1087 | 0.5388 ± 0.0631 |
| LR | 0.7477 ± 0.0559 | 0.8190 ± 0.0145 | 0.7727 ± 0.0000 | 0.7921 ± 0.0000 |

exp1：固定 holdout test，5 個 stratified folds、model seeds=42/43/44，先平均各 fold 的 seeds，再報告跨 fold mean±sample std。
exp2：chronological 60/20/20，model seeds=42/43/44，報告跨 seed mean±sample std。
來源：`p0_exp1_metrics.csv`、`p0_exp2_metrics.csv`。

## 歷史 IF 診斷（選擇六特徵的依據，非目前主模型結果）

7 種設定 × seeds 42/43/44 × exp1/exp2，共 42 次正常資料訓練。
exp1 此處僅使用 fold 0，不能將下表的跨 seed 變異與上表跨 fold 變異混用。
各變體均相對 baseline 改變一項設定；固定特徵集合採用組員既有 EDA，不重算 KS。

| 設定 | exp1 AP | exp2 AP |
| --- | --- | --- |
| 全 30 特徵、256 samples | 0.1335 ± 0.0146 | 0.0190 ± 0.0021 |
| 移除 Time | 0.1058 ± 0.0182 | 0.0202 ± 0.0021 |
| V1–V28 | 0.1454 ± 0.0312 | 0.0251 ± 0.0051 |
| V14-only | 0.6561 ± 0.0099 | 0.4655 ± 0.0406 |
| 固定六特徵 | 0.6554 ± 0.0543 | 0.4489 ± 0.1087 |
| 全特徵、1024 samples | 0.1352 ± 0.0245 | 0.0242 ± 0.0009 |
| 全特徵、4096 samples | 0.2190 ± 0.0131 | 0.0347 ± 0.0030 |

主要證據支持特徵組合為最重要因素：全特徵 IF 偏重其他稀有正常模式。
移除 Time 幾乎未改善 exp2；增加 samples 有改善，但遠小於縮減特徵。
原始 exp2 seed=42 的前 100 筆異常分數全為正常交易；其中 V20/V2/V23
落在正常 training 第 1–99 百分位範圍之外的比例分別為 96%/95%/89%。
這是關聯證據，尚未確認單一特徵的因果貢獻。
來源：`if_diagnostics_metrics.csv`、`if_exp2_tail_features.csv`。

## 已採用設定與後續

- T2 程式預設已改成 V14、V10、V12、V4、V11、V17；保留 V14-only ablation。
- 六特徵版在 exp2 的 seed 波動仍大，不能宣稱優於 V14-only。
- 六特徵版已完成 exp1 全部 5 folds 與 exp2 P0；舊全特徵主結果、模型及預測已替換。
- T3 runner 已完成 P0–P6 比較，維持相同 raw split/validation；採樣 seeds=42/43/44、split seed 固定 42、balanced、C=1，SMOTE ratio=0.01。
- IF contamination C0–C4 已完成 exp1 五個 folds、exp2，各三個 seeds；主比較仍使用 normal-only。
- 分工範圍到正式 test 評估與顯著性分析；主比較 seeds=42/43/44 已補齊，設定已凍結；正式 test 與配對 bootstrap 顯著性分析已完成。
- 55 項測試通過。環境使用 uv，Python 3.11，依賴由 uv.lock 鎖定。

完整模型與逐筆預測存於本機 `outputs/`，未納入 Git；本目錄保留小型結果表以供追蹤。

主比較多 seed 的詳細統計與 seed/fold 區分見 [P0_SEEDS.md](P0_SEEDS.md)。
