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
