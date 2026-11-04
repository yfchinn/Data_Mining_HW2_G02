# Data_Mining_HW2_G02

## 新版 Exp1：預先採樣 P1–P4 的三個 tiers

`uv run --frozen python run_new_exp1_tiers.py` 直接使用
`data_preparation_exp1_updated_v2.py` 的 outer 60/20/20 接口，seed42，
對八組 P1–P4 模式執行三個 tiers 的 train/validation/test；不含 inner CV。
結果見 `results/EXPERIMENT_SUMMARY_ZH.md`，模型與逐筆預測存於
`outputs/new_exp1_tiers/seed_42/`。這是獨立探索協定，原正式 test 結果保留。

## 環境（uv）

專案使用 Python 3.11，依賴由 `pyproject.toml` 與 `uv.lock` 管理。
鎖定 NumPy 1.x，以相容現有 `scikit-learn-extra` K-Medoids 實作。

```powershell
uv sync --frozen
uv run --frozen pytest -q
```

在 VS Code 選擇 `.venv/Scripts/python.exe` 作為 interpreter。
新增依賴使用 `uv add <package>`（測試工具用 `uv add --dev <package>`），
並一併提交 `pyproject.toml`、`uv.lock`。`requirements.txt` 僅為相容匯出，勿手動維護：

```powershell
uv export --frozen --no-dev --format requirements-txt --output-file requirements.txt
```

資料集使用 Git LFS；`data/creditcard.csv` 必須是完整 CSV，而非 LFS pointer。
若尚未下載，執行 `git lfs pull`。

## 共用資料接口

訓練程式統一從 `src.data_preparation` import，原有兩支實驗腳本仍保留。

```python
from src.data_preparation import get_prepared_data, iter_prepared_folds

# Exp1：固定 20% test，其餘 80% 做 stratified 5-fold。
for fold, data in iter_prepared_folds(experiment="exp1", mode="P0"):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    # 只以 validation 選模型與門檻。

# Exp2：chronological 60/20/20，只有 fold_index=0。
data = get_prepared_data(experiment="exp2", mode="P0")

# T1/T2 normal-only：scaler 與模型都只使用正常訓練資料。
normal_data = get_prepared_data(
    experiment="exp2", mode="C0", fit_normal_only=True,
)
assert normal_data["y_train"].eq(0).all()
```

共同參數：`experiment`、`mode`、`csv_path`、`random_state`、`smote_ratio`、
`fit_normal_only`。`get_prepared_data` 另接受 `fold_index`。

- exp1 切分參數：`n_splits=5`、`test_ratio=0.2`、`random_state=42`。
- exp2 切分參數：`train_ratio=0.6`、`val_ratio=0.2`；iterator 只產生 `(0, data)`。
- 共用入口的 `smote_ratio=0.01` 指 fraud/normal，兩個 experiment 相同。
  舊 exp2 腳本直接執行時仍維持原本的 `0.1`。
- `fit_normal_only=True` 只改變 scaler 的 fit population；P0 仍保留兩類訓練資料。
  使用 P0 訓練 normal-only 模型時，還須自行篩選 `y_train == 0`；C0 已完成此篩選。

### 統一回傳格式

| Key | 型別與用途 |
| --- | --- |
| `X_train`, `X_val`, `X_test` | float DataFrame，欄位固定為 Time、V1–V28、Amount，不含標籤或 IDs |
| `y_train`, `y_val`, `y_test` | Series，0=normal、1=fraud，與 X 的 index／順序對齊 |
| `train_row_ids`, `val_row_ids`, `test_row_ids` | ndarray，對應原始 CSV 的零起算列號；SMOTE 合成列 ID 為 `None` |
| `feature_columns` | 特徵欄位順序 |
| `scaler` | training-fitted RobustScaler，僅轉換 Time、Amount，沒有 log1p |
| `metadata` | experiment、split、fold、n_splits、mode、random_state、scaling、scaler_population、scaler_fit_rows，以及各模式資訊 |

每次採樣、IQR 過濾、污染實驗都保留相同的 validation/test 與其 IDs。
同一 experiment 使用相同切分參數，才是共用 raw split；exp1 與 exp2 是不同實驗，不應混成同一張公平比較表。
`X_test` 可被取得不代表可以用來選模型：設定與門檻固定後才做最終測試評估。

### Modes

使用 `available_modes()` 取得全部模式：

- `P0`：不採樣、不移除離群值。
- `P1-1/5/10`：random undersampling，normal:fraud 為 1/5/10:1。
- `P2-1/5/10`：normal-only V10 IQR 移除後 random undersampling。
- `P3-1/5/10`：K-Medoids undersampling。
- `P4-1/5/10`：normal-only V10 IQR 移除後 K-Medoids。
- `P5`：SMOTE；`P6`：normal-only V10 IQR 移除後 SMOTE。
- `C0`–`C4`：保留全部正常 training，以巢狀 fraud 子集改變訓練污染量。

C 模式的預設 fraud 筆數為 `floor(L * [0, 1/12, 1/4, 1/2, 1])`，
其中 `L = min(360, 當前 fold 可用 fraud 筆數)`，避免原本 exp1 要求 360 筆導致失敗。
當可用量為 360 時，仍是 0/30/90/180/360。極小資料可能有重複污染量。
這些是筆數設定，不是固定 1%/5%/10% 污染率；實際值讀取
`metadata["fraud_count"]`、`metadata["contamination_pct"]`。
C1–C4 僅供污染敏感度研究；normal-only 主比較使用 C0。

## 驗證

測試涵蓋兩套實驗的所有 modes、共用 test holdout、時間順序、合成列 ID、
row ID／標籤對齊，以及只使用 training 資料擬合 scaler。

## 三個模型與 P0 執行

```powershell
# Exp2：時間切分，先在 validation 跑通三個 tiers。
uv run --frozen python run_experiment.py --experiment exp2 --mode P0

# Exp1：預設跑完全部 5 folds；只跑一個 fold 可加 --folds 0。
uv run --frozen python run_experiment.py --experiment exp1 --mode P0

# 多個模型 seeds，共用同一 raw split。
uv run --frozen python run_experiment.py --experiment exp2 --seeds 42 43 44
```

| Tier | 模型檔案 | P0 訓練與分數 |
| --- | --- | --- |
| T1 | `src/models/three_sigma.py` | 固定 V14，僅正常 training 估計平均與標準差（ddof=0）；`abs((V14-mean)/std)` |
| T2 | `src/models/isolation_forest.py` | 僅正常 training；200 trees、max_samples=256，分數為負的 sklearn score_samples |
| T3 | `src/models/logistic_regression.py` | 兩類 training 標籤；C=1、class_weight=balanced、lbfgs、max_iter=2000，分數為 fraud 機率 |

V14 是團隊指定的固定特徵，不重新計算 KS，也不重新挑選 feature。
T2 固定使用 V14、V10、V12、V4、V11、V17；T3 使用全部 30 個特徵。
T2 的 V14-only ablation 使用 `IsolationForestDetector(features=("V14",))`，
`features=None` 僅供重現歷史全特徵診斷。目前 P0 的模型、預測與結果已統一為六特徵 T2。T1/T2 的 Time、Amount scaler 僅 fit 正常 training，
T3 的 scaler fit 全部 training；兩者仍使用相同的 raw split、原始 row IDs 與驗證標籤。
模型本身也會根據 training labels 選擇允許的訓練母體。

三個模型提供相同的 `fit(X, y)`、`score_samples(X)`、`predict(X)` 接口。
分數一律越大越異常，`predict` 回傳 0/1。
T1 固定 `score >= 3`；T2/T3 以 validation 最大 F1 選門檻，若同分則選較高門檻，
並使用 `score >= threshold` 判定。Isolation Forest 的 contamination="auto" 僅為
sklearn 內部設定；本流程不使用其 predict 或內建 offset 決策。

目前是固定超參數的初始 P0 實驗，不含超參數搜尋。
預設只計算 validation 指標；在 validation 上選出的 F1 不是獨立泛化估計。
所有設計固定後，可加 `--evaluate-test` 評估 test；請先凍結設定，避免反覆以 test 調參。

結果預設放在 `outputs/P0/<experiment>/`（可用 `--output-dir` 指定獨立目錄）：

- `metrics.csv`：每個 fold/seed/tier 的 AP（Average Precision，非梯形 PR-AUC）、
  ROC-AUC、F1、precision、recall、FPR、混淆矩陣、門檻與時間。
- `summary.csv`：執行的 fold/seed 組合之 mean、sample std、count；單次執行的 std 留白。
  folds 與 seeds 的描述統計不代表獨立的顯著性檢定。
- `run.json`：資料 SHA256、設定、版本、硬體資訊與執行狀態。
- `fold_<n>/split_ids.npz`：共用切分的原始 row IDs。
- `fold_<n>/*.joblib`：model、scaler、feature_columns 與 metadata；`threshold_` 存在模型內。
- `fold_<n>/*_val_predictions.csv`：row_id、y_true、score、threshold、y_pred；
  指定 `--evaluate-test` 才產生 test predictions。

重新載入時先按 `feature_columns` 取 raw features，以 bundle 的 scaler 轉換
Time、Amount，再呼叫模型。若要精確重現 CSV 中落在門檻上的比較，使用
`pd.read_csv(..., float_precision="round_trip")`。
同一輸出路徑重跑會覆寫同名檔案，正式實驗應指定新的 `--output-dir`。
時間為模型 fit 與批次 score，未含前處理或 IO；每筆延遲是批次攤提值。
T1 與 lbfgs LR 為確定性方法，改 seed 不一定產生差異。

## T3 P0–P6 採樣比較

```powershell
uv run --frozen python run_t3_sampling.py --experiment exp2
uv run --frozen python run_t3_sampling.py --experiment exp1
```

每套實驗執行 15 組：P0、P1–P4 各 1:1/5:1/10:1 normal:fraud，
以及 P5/P6。固定 C=1、class_weight=balanced、seed=42，SMOTE fraud/normal
預設 0.01（可用 `--smote-ratio` 指定）。exp1 跑全部五個 folds。
同一 fold 共用採樣前的 training-fitted scaler 與 validation，僅 training 做處理。
P3/P4 沿用組員的 candidate-based K-Medoids 近似方法。
各模式分別以 validation 選 F1 門檻，結果存於 `outputs/T3_sampling/<experiment>/`。
輸出包括 metrics、summary、split IDs、模型及 validation predictions；不評估 test。
原始單 seed 結果已補齊三個採樣 seeds；validation 最佳 F1
也有門檻選擇的樂觀偏差；不得直接視為最終 test 效能。

補齊採樣 seeds 43/44 並合併既有 42：

```powershell
uv run --frozen python complete_t3_seeds.py
```

`run_t3_sampling.py --seed` 控制採樣與模型，`--split-seed` 控制 raw split（預設固定 42）。
補跑模型分開存放在 `seed_43/`、`seed_44/`。合併結果涵蓋每個 fold、每個模式的三個 seeds。
exp1 先平均各 fold 的 seeds，再計算跨 fold mean/std；exp2 計算跨 seed mean/std。
各 fold 內的 seed mean/std 另存 results 的 `*_within_fold_seeds.csv`。
validation 門檻選擇的樂觀偏差仍存在，不能直接視為最終 test 效能。

## IF contamination ablation

新 pre-split P1-10 的 T2 C0–C4（三 seeds）結果見
[`results/EXPERIMENT_SUMMARY_ZH.md`](results/EXPERIMENT_SUMMARY_ZH.md)。
重現：`uv run --frozen python run_new_exp1_if_contamination.py`。
此探索使用固定 P1-10 validation/test，與下列原始資料污染實驗分開保存。

```powershell
uv run --frozen python run_if_contamination.py --experiment exp2
uv run --frozen python run_if_contamination.py --experiment exp1
```

固定六特徵、200 trees、256 samples/tree、正常訓練樣本及 normal-only scaler，
以 seeds 42/43/44 比較 C0–C4。fraud 子集依 seed 選擇並保持巢狀包含。
資料排序固定為正常列（依原始 ID 排序）後接 fraud 列，避免各 C 模式獨立打亂正常列。
此排序與原本 exp1 P0 順序不同，因此差異一律對本實驗自己的 C0 計算。
刻意直接 fit 所有選入樣本，避免主模型的 normal-only 篩選把污染 fraud 再次移除。
這是污染敏感度研究；正式三 tier 主比較的 IF 仍為 normal-only。
僅評估 validation，結果存於 `outputs/IF_contamination/<experiment>/`。
實際污染百分比由資料計算，C4 也只有約 0.17–0.21%，不是 10%。

## 正式 test 評估

正式結果見 `results/FINAL_TEST.md`。`evaluate_final_test.py` 載入已保存的 P0
模型及 validation 門檻，評估兩套實驗的三個 tiers，並執行 1000 次配對、
按類別分層的 transaction bootstrap，比較 IF 與 LR 的平均 AP 差異。
不重新訓練模型、不選 test 門檻；設定在評分前寫入 frozen_protocol.json。
輸出已存於 `outputs/final_test/`，程式會拒絕覆寫既有目錄。
主要結果、混淆矩陣、訓練／推論時間與 CI 保留在 results 的 final_test 檔案中。
