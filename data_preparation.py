
"""Data Mining HW2 - Credit Card Fraud Detection

Version 1.1
1. Chronological Split (60% / 20% / 20%)
2. RobustScaler for Time and Amount
3. Training Contamination Ablation (C0-C4)
4. Keep Validation and Test sets unchanged

Class 0 = Normal
Class 1 = Fraud
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ============================================================
# Step 1. Load Dataset
# ============================================================

def load_data(csv_path="data/creditcard.csv"):
    """Read and validate the original dataset."""

    df = pd.read_csv(csv_path)

    required_columns = (
        ["Time", "Amount", "Class"]
        + [f"V{i}" for i in range(1, 29)]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    # Save original row ID for leakage checking.
    df = df.copy()
    df["row_id"] = np.arange(len(df))

    return df


# ============================================================
# Step 2. Chronological Split
# ============================================================

def chronological_split(
    df,
    train_ratio=0.6,
    val_ratio=0.2
):
    """Split transactions by chronological order."""

    if not (0 < train_ratio < 1):
        raise ValueError("Invalid train_ratio.")

    if not (0 < val_ratio < 1):
        raise ValueError("Invalid val_ratio.")

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1."
        )

    # Stable sorting preserves original order for tied Time.
    df = df.sort_values(
        "Time",
        kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ============================================================
# Step 3. Feature Scaling
# ============================================================

def scale_features(
    train_df,
    val_df,
    test_df
):
    """Fit RobustScaler on Train only.

    Scale Time and Amount.
    Keep V1-V28 unchanged.
    """

    feature_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    scale_columns = ["Time", "Amount"]

    scaler = RobustScaler()

    # Fit only on Training Set.
    scaler.fit(train_df[scale_columns])

    def transform(df):
        X = df[feature_columns].copy()

        X.loc[:, scale_columns] = scaler.transform(
            df[scale_columns]
        )

        y = df["Class"].copy()

        return X, y

    X_train, y_train = transform(train_df)
    X_val, y_val = transform(val_df)
    X_test, y_test = transform(test_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }


# ============================================================
# Step 4. Prepare Original Dataset
# ============================================================

def prepare_data(
    csv_path="data/creditcard.csv",
    train_ratio=0.6,
    val_ratio=0.2
):
    """Prepare the original chronological dataset.

    No sampling or outlier removal is applied.
    """

    df = load_data(csv_path)

    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    data = scale_features(
        train_df,
        val_df,
        test_df
    )

    # Keep original row IDs.
    data["train_row_ids"] = train_df["row_id"].to_numpy()
    data["val_row_ids"] = val_df["row_id"].to_numpy()
    data["test_row_ids"] = test_df["row_id"].to_numpy()

    data["metadata"] = {
        "split_method": "chronological",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "scaling": "RobustScaler",
        "scaled_features": ["Time", "Amount"],
        "sampling": "none",
        "outlier_removal": "none",
    }

    # Verify that row IDs do not overlap.
    train_ids = set(data["train_row_ids"])
    val_ids = set(data["val_row_ids"])
    test_ids = set(data["test_row_ids"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return data


# ============================================================
# Step 5. Training Contamination Ablation
# ============================================================

def prepare_contamination_data(
    data,
    fraud_counts=(0, 30, 90, 180, 360),
    random_state=42
):
    """Create C0-C4 for Isolation Forest.

    - Keep all normal transactions in Training Set.
    - Gradually add fraud transactions.
    - Use only fraud transactions from original Train.
    - Keep Validation and Test unchanged.
    - Use nested fraud subsets for controlled comparison.
    """

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Normal and fraud row positions.
    normal_positions = np.flatnonzero(
        y_train.to_numpy() == 0
    )

    fraud_positions = np.flatnonzero(
        y_train.to_numpy() == 1
    )

    normal_count = len(normal_positions)
    available_fraud = len(fraud_positions)

    # Validate requested fraud counts.
    fraud_counts = tuple(fraud_counts)

    if not fraud_counts:
        raise ValueError("fraud_counts cannot be empty.")

    if len(set(fraud_counts)) != len(fraud_counts):
        raise ValueError("fraud_counts must be unique.")

    if any(
        count < 0 or count > available_fraud
        for count in fraud_counts
    ):
        raise ValueError(
            f"Fraud counts must be between "
            f"0 and {available_fraud}."
        )

    # Shuffle fraud positions only once.
    # C1 fraud subset is contained in C2, C3, C4.
    rng = np.random.default_rng(random_state)

    shuffled_fraud_positions = rng.permutation(
        fraud_positions
    )

    experiments = {}

    for i, fraud_count in enumerate(fraud_counts):

        # Select the first N fraud transactions.
        selected_fraud = shuffled_fraud_positions[
            :fraud_count
        ]

        # Keep all normal transactions.
        selected_positions = np.concatenate([
            normal_positions,
            selected_fraud
        ])

        # Shuffle selected training rows.
        experiment_rng = np.random.default_rng(
            random_state + i
        )

        selected_positions = experiment_rng.permutation(
            selected_positions
        )

        X_exp = X_train.iloc[
            selected_positions
        ].copy().reset_index(drop=True)

        y_exp = y_train.iloc[
            selected_positions
        ].copy().reset_index(drop=True)

        total = len(y_exp)

        contamination_pct = (
            fraud_count / total * 100
        )

        name = f"C{i}"

        experiments[name] = {
            "X_train": X_exp,
            "y_train": y_exp,

            # Validation and Test remain unchanged.
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],

            "metadata": {
                "experiment": name,
                "normal_count": normal_count,
                "fraud_count": fraud_count,
                "total_count": total,
                "contamination_pct": contamination_pct,
                "random_state": random_state,
                "training_mode": "isolation_forest",
            }
        }

    return experiments


# ============================================================
# Step 6. Print Dataset Summary
# ============================================================

def print_split_summary(data):
    """Print original Train / Validation / Test statistics."""

    print("\n=== Chronological Split Summary ===")

    for name in ["train", "val", "test"]:

        y = data[f"y_{name}"]

        normal = int((y == 0).sum())
        fraud = int((y == 1).sum())
        total = len(y)

        fraud_pct = fraud / total * 100

        print(
            f"{name:5s} | "
            f"n={total:6d} | "
            f"normal={normal:6d} | "
            f"fraud={fraud:4d} | "
            f"fraud%={fraud_pct:.4f}"
        )

    print(
        "\nFeatures:",
        len(data["feature_columns"])
    )

    print("Row ID overlap: none")


def print_contamination_summary(experiments):
    """Print C0-C4 training contamination results."""

    print("\n=== Training Contamination Ablation ===")

    for name, exp in experiments.items():

        info = exp["metadata"]

        print(
            f"{name} | "
            f"Normal={info['normal_count']:6d} | "
            f"Fraud={info['fraud_count']:3d} | "
            f"Total={info['total_count']:6d} | "
            f"Contamination={info['contamination_pct']:.4f}%"
        )


# ============================================================
# Step 7. Main
# ============================================================

if __name__ == "__main__":

    # Update path if creditcard.csv is stored elsewhere.
    csv_path = Path(__file__).resolve().parent / "data/creditcard.csv"

    # Original Time-based Split.
    data = prepare_data(csv_path)

    print_split_summary(data)

    # Generate Isolation Forest contamination experiments.
    experiments = prepare_contamination_data(
        data,
        fraud_counts=(0, 30, 90, 180, 360),
        random_state=42
    )

    print_contamination_summary(experiments)

    # Example: access C2 training data.
    X_train_C2 = experiments["C2"]["X_train"]
    y_train_C2 = experiments["C2"]["y_train"]

    print("\n=== Example: C2 ===")
    print("X_train shape:", X_train_C2.shape)
    print("y_train shape:", y_train_C2.shape)
    print("Class distribution:")
    print(y_train_C2.value_counts().sort_index())
