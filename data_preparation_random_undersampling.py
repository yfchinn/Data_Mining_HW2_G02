"""Data Mining HW2 - Version 2: Random UnderSampling before 60/20/20 split.

Exploratory balanced-dataset experiment: evaluation sets are also balanced.
Class 0 = Normal; Class 1 = Fraud.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLUMNS = ["Time", "Amount"]
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "creditcard.csv"


def load_data(csv_path):
    """讀取資料並保留原始 row_id，方便檢查是否有重複樣本。"""
    df = pd.read_csv(csv_path)
    required = FEATURES + ["Class"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Missing values detected")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    df = df.copy()
    df["row_id"] = np.arange(len(df))
    return df


def random_undersample_before_split(df, random_state=RANDOM_STATE):
    """先分正常/詐欺，再將正常資料隨機欠採樣至詐欺資料筆數 (1:1)。"""
    normal_df = df.loc[df["Class"] == 0].copy()
    fraud_df = df.loc[df["Class"] == 1].copy()
    if fraud_df.empty or len(normal_df) < len(fraud_df):
        raise ValueError("Fraud must be nonempty and normal count >= fraud count")

    sampled_normal = normal_df.sample(n=len(fraud_df), replace=False,
                                      random_state=random_state)
    balanced_df = pd.concat([sampled_normal, fraud_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    assert balanced_df["row_id"].is_unique
    return balanced_df


def split_602020(df, random_state=RANDOM_STATE):
    """Stratified 60/20/20 split，比例以欠採樣後的資料為基準。"""
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df["Class"],
        shuffle=True, random_state=random_state
    )
    val_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["Class"],
        shuffle=True, random_state=random_state
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def scale_features(train_df, val_df, test_df):
    """RobustScaler 只用 Train 擬合；Val/Test 只能 transform。"""
    scaler = RobustScaler()
    scaler.fit(train_df[SCALE_COLUMNS])
    result = {"scaler": scaler, "feature_columns": FEATURES.copy()}
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        X = frame[FEATURES].astype(float).copy()
        X.loc[:, SCALE_COLUMNS] = scaler.transform(frame[SCALE_COLUMNS])
        result[f"X_{name}"] = X.reset_index(drop=True)
        result[f"y_{name}"] = frame["Class"].reset_index(drop=True)
        result[f"{name}_row_ids"] = frame["row_id"].to_numpy().copy()
    return result


def get_prepared_data(csv_path=DEFAULT_CSV, random_state=RANDOM_STATE):
    """提供模型訓練者使用的 API，回傳 X_train/y_train/X_val/y_val/X_test/y_test。"""
    original = load_data(csv_path)
    balanced = random_undersample_before_split(original, random_state)
    train_df, val_df, test_df = split_602020(balanced, random_state)
    data = scale_features(train_df, val_df, test_df)
    ids = [set(data[f"{name}_row_ids"]) for name in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    data["metadata"] = {
        "version": "Version 2 (Random UnderSampling)",
        "sampling": "random_undersampling_before_split",
        "normal_to_fraud_ratio": "1:1",
        "split": "stratified_60_20_20",
        "random_state": random_state,
        "original_counts": original["Class"].value_counts().to_dict(),
        "balanced_counts": balanced["Class"].value_counts().to_dict(),
    }
    return data


def main(csv_path):
    data = get_prepared_data(csv_path)
    print("\n=== Version 2 (Random UnderSampling) ===")
    original = data["metadata"]["original_counts"]
    balanced = data["metadata"]["balanced_counts"]
    print(f"Original | Normal={original.get(0, 0)} | Fraud={original.get(1, 0)}")
    print(f"Balanced | Normal={balanced.get(0, 0)} | Fraud={balanced.get(1, 0)}")
    print("\n--- Stratified Split 6:2:2 ---")
    for part, label in (("train", "Train (60%)"), ("val", "Validation (20%)"), ("test", "Test (20%)")):
        y = data[f"y_{part}"]
        print(f"{label:18s} | Normal={(y == 0).sum():4d} | Fraud={(y == 1).sum():4d} | Total={len(y):4d}")
    print("\nData split verified: no overlapping original row IDs.")
    print("RobustScaler fitted on Train only (Time, Amount).")
    print("No CSV or NPZ files saved.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random UnderSampling 1:1 before 60/20/20 split")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    main(args.csv)
