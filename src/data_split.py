
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# 1. 設定檔案路徑
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = "/Users/linda/Downloads/creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 讀取資料
# ==========================================

df = pd.read_csv(DATA_PATH)

print("===== Dataset Information =====")
print("資料形狀:", df.shape)
print("欄位名稱:", df.columns.tolist())
print("缺失值總數:", df.isna().sum().sum())
print("重複資料列數:", df.duplicated().sum())

# 檢查資料是否包含必要欄位
feature_cols = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)
required_cols = feature_cols + ["Class"]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(f"缺少必要欄位: {missing_cols}")

# 檢查是否有缺失值
if df[required_cols].isna().any().any():
    raise ValueError("資料含有缺失值，請先檢查。")

# 檢查標籤是否只有 0 和 1
if not df["Class"].isin([0, 1]).all():
    raise ValueError("Class 必須只有 0（正常）與 1（詐欺）。")

# ==========================================
# 3. 檢查整體類別分布
# ==========================================

print("\n===== Overall Class Distribution =====")

class_counts = df["Class"].value_counts().sort_index()

print("正常交易:", class_counts.get(0, 0))
print("詐欺交易:", class_counts.get(1, 0))
print("詐欺比例:", f"{df['Class'].mean() * 100:.4f}%")

print("\nTime 範圍:")
print("最小值:", df["Time"].min())
print("最大值:", df["Time"].max())

# ==========================================
# 4. Chronological Split
# ==========================================

# 建立原始資料列 ID，方便後續追蹤
df = df.copy()
df["sample_id"] = np.arange(len(df))

# 按照 Time 由小到大排序
# mergesort 為穩定排序，相同 Time 保留原始順序
df = df.sort_values(
    by="Time",
    kind="mergesort"
).reset_index(drop=True)

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# ==========================================
# 5. 檢查切分結果
# ==========================================

def print_split_info(name, data):
    normal_count = (data["Class"] == 0).sum()
    fraud_count = (data["Class"] == 1).sum()
    fraud_ratio = data["Class"].mean() * 100

    print(f"\n===== {name} =====")
    print("資料筆數:", len(data))
    print("正常交易:", normal_count)
    print("詐欺交易:", fraud_count)
    print("詐欺比例:", f"{fraud_ratio:.4f}%")
    print("Time 範圍:",
          data["Time"].min(),
          "至",
          data["Time"].max())


print_split_info("Train", train_df)
print_split_info("Validation", val_df)
print_split_info("Test", test_df)

# 檢查三個集合是否有交集
train_ids = set(train_df["sample_id"])
val_ids = set(val_df["sample_id"])
test_ids = set(test_df["sample_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

# 確保所有資料都被分配
assert len(train_df) + len(val_df) + len(test_df) == n

# 確保時間順序正確
assert train_df["Time"].max() <= val_df["Time"].min()
assert val_df["Time"].max() <= test_df["Time"].min()

# 確保每個集合都包含正常與詐欺交易
for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    if data["Class"].nunique() != 2:
        raise ValueError(
            f"{name} 不包含完整的正常與詐欺類別，"
            "需要重新檢查切分方案。"
        )

# ==========================================
# 6. 儲存共用切分索引
# ==========================================

split_index = pd.concat([
    train_df[["sample_id", "Time", "Class"]].assign(
        split="train"
    ),
    val_df[["sample_id", "Time", "Class"]].assign(
        split="validation"
    ),
    test_df[["sample_id", "Time", "Class"]].assign(
        split="test"
    )
], ignore_index=True)

split_index.to_csv(
    OUTPUT_DIR / "split_indices.csv",
    index=False
)

# 儲存切分後的完整資料
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\n===== Split Completed =====")
print("已儲存至:", OUTPUT_DIR)
print("train.csv")
print("validation.csv")
print("test.csv")
print("split_indices.csv")
