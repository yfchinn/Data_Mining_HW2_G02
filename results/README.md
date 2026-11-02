# 實驗進度摘要（2026-11-02）

所有以下結果均為 validation，尚未評估 test，也尚未執行 contamination 或 T3 P1–P6。
AP 為 Average Precision。T2/T3 的 F1 門檻由同一 validation 選出，並非獨立泛化估計。

## 初始 P0

T1 固定 V14 雙側 3σ，μ/σ 僅用正常 training 估計；不重新計算 KS。
此表 T2 為舊版全 30 特徵 IF（200 trees、256 samples/tree）。
T3 為全 30 特徵 LR（C=1、balanced、lbfgs）。
T1/T2 只用正常 training；T3 使用兩類 training 標籤。

| 模型 | exp1 AP | exp1 F1 | exp2 AP | exp2 F1 |
| --- | --- | --- | --- | --- |
| V14 3σ | 0.5970 ± 0.0340 | 0.1767 ± 0.0087 | 0.4841 | 0.1020 |
| IF（全特徵） | 0.1235 ± 0.0359 | 0.2518 ± 0.0531 | 0.0166 | 0.0447 |
| LR | 0.7477 ± 0.0559 | 0.8190 ± 0.0145 | 0.7727 | 0.7921 |

exp1：固定 holdout test，5 個 stratified folds、model seed=42，報告跨 fold mean±sample std。
exp2：chronological 60/20/20，單次 model seed=42。
來源：`p0_exp1_metrics.csv`、`p0_exp2_metrics.csv`。

## IF 診斷

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
- 初始 P0 表仍為全特徵 T2；新版完整 5-fold P0 比較尚未重跑。
- 下一步擴充 T3 runner，固定 LR 設定比較 P0–P6，維持相同 raw split/validation。
- Test 保留到設定凍結後評估；目前不做 contamination。
- 52 項測試通過。環境使用 uv，Python 3.11，依賴由 uv.lock 鎖定。

完整模型與逐筆預測存於本機 `outputs/`，未納入 Git；本目錄保留小型結果表以供追蹤。
