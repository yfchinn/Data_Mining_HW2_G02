# Data_Mining_HW2_G02

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
原始全特徵版本使用 `features=None`。既有 P0 結果是更改前的全特徵版本，
六特徵版本應指定新的輸出目錄重跑，避免覆蓋原始比較結果。T1/T2 的 Time、Amount scaler 僅 fit 正常 training，
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
