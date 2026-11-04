
"""Data Mining HW2 - Data Preparation V2.

Class 0 = Normal
Class 1 = Fraud

Features:
- Chronological 60/20/20 split
- Train-only RobustScaler for Time and Amount
- Normal-only IQR removal
- Random UnderSampling (1:1, 5:1, 10:1)
- Isolation Forest contamination ablation C0-C4
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn_extra.cluster import KMedoids

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Function stable_cumsum is deprecated.*",
    category=FutureWarning
)

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# ==========================================================
# Experiment Configuration
# ==========================================================

# IQR 使用的特徵，可修改成 V10、V12、V14 等
IQR_FEATURE = "V10"

# IQR 離群值判斷係數
IQR_FACTOR = 1.5

# 固定隨機種子
RANDOM_STATE = 42

# SMOTE ratio = Fraud / Normal
MOTE_RATIO = 0.1  # Normal : Fraud = 10:1

# ==========================================================
# 1. Load original data
# ==========================================================

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    required = FEATURES + ["Class"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must be 0 or 1.")

    df = df.copy()
    df["row_id"] = np.arange(len(df))

    return df


# ==========================================================
# 2. Chronological split
# ==========================================================

def chronological_split(
    df,
    train_ratio=0.6,
    val_ratio=0.2
):
    if not (
        0 < train_ratio < 1
        and 0 < val_ratio < 1
        and train_ratio + val_ratio < 1
    ):
        raise ValueError("Invalid split ratios.")

    df = df.sort_values(
        "Time",
        kind="mergesort"
    )

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )


# ==========================================================
# 3. Train-only RobustScaler
# ==========================================================

def scale_features(train_df, val_df, test_df, fit_normal_only=False):

    scale_cols = ["Time", "Amount"]

    scaler = RobustScaler()

    # Fit using Training Set only.
    fit_frame = train_df.loc[train_df["Class"].eq(0)] if fit_normal_only else train_df
    if fit_frame.empty:
        raise ValueError("No training rows available for scaler fitting.")
    scaler.fit(fit_frame[scale_cols])

    result = {
        "scaler": scaler,
        "feature_columns": FEATURES.copy(),
        "metadata": {"scaler_population": "normal_train" if fit_normal_only else "all_train",
                     "scaler_fit_rows": len(fit_frame)}
    }

    for name, frame in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df)
    ]:
        X = frame[FEATURES].copy().astype(float)

        X.loc[:, scale_cols] = scaler.transform(
            frame[scale_cols]
        )

        result[f"X_{name}"] = X.reset_index(drop=True)

        result[f"y_{name}"] = (
            frame["Class"]
            .reset_index(drop=True)
        )

        result[f"{name}_row_ids"] = (
            frame["row_id"].to_numpy().copy()
        )

    return result


# ==========================================================
# 4. Original chronological dataset (P0)
# ==========================================================

def prepare_data(
    csv_path="data/creditcard.csv",
    train_ratio=0.6,
    val_ratio=0.2,
    fit_normal_only=False
):
    df = load_data(csv_path)

    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio,
        val_ratio
    )

    data = scale_features(
        train_df,
        val_df,
        test_df,
        fit_normal_only=fit_normal_only
    )

    train_ids = set(data["train_row_ids"])
    val_ids = set(data["val_row_ids"])
    test_ids = set(data["test_row_ids"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    data["metadata"] = {
        **data["metadata"],
        "split": "chronological",
        "scaling": "RobustScaler",
        "sampling": "none"
    }

    return data


# ==========================================================
# 5. Normal-only IQR outlier removal
# ==========================================================

def remove_normal_outliers(
    X_train,
    y_train,
    row_ids,
    feature="V14",
    factor=1.5
):
    """Remove normal-class outliers only."""

    if feature not in X_train.columns:
        raise ValueError(f"Unknown feature: {feature}")

    if factor <= 0:
        raise ValueError("IQR factor must be positive.")

    labels = y_train.to_numpy()
    values = X_train[feature].to_numpy()

    normal_mask = labels == 0

    # Calculate quartiles using normal training data.
    normal_values = values[normal_mask]

    q1, q3 = np.quantile(
        normal_values,
        [0.25, 0.75]
    )

    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    # Keep all fraud samples.
    # Remove only normal samples outside IQR bounds.
    keep_mask = (
        ~normal_mask
        | ((values >= lower) & (values <= upper))
    )

    removed_normal = int(
        (normal_mask & ~keep_mask).sum()
    )

    positions = np.flatnonzero(keep_mask)

    X_new = (
        X_train.iloc[positions]
        .reset_index(drop=True)
    )

    y_new = (
        y_train.iloc[positions]
        .reset_index(drop=True)
    )

    ids_new = np.asarray(row_ids)[positions].copy()

    info = {
        "feature": feature,
        "q1": float(q1),
        "q3": float(q3),
        "lower": float(lower),
        "upper": float(upper),
        "normal_removed": removed_normal,
        "fraud_removed": 0
    }

    return X_new, y_new, ids_new, info


# ==========================================================
# 6. Random UnderSampling
# ==========================================================

def random_undersampling(
    X_train,
    y_train,
    row_ids,
    ratio=1,
    random_state=42
):
    """ratio = Normal / Fraud after sampling."""

    if ratio not in (1, 5, 10):
        raise ValueError("Ratio must be 1, 5, or 10.")

    labels = y_train.to_numpy()

    normal_positions = np.flatnonzero(labels == 0)
    fraud_positions = np.flatnonzero(labels == 1)

    fraud_count = len(fraud_positions)

    if fraud_count == 0:
        raise ValueError("No fraud samples in Training Set.")

    target_normal = ratio * fraud_count

    if target_normal > len(normal_positions):
        raise ValueError(
            "Not enough normal samples for this ratio."
        )

    rng = np.random.default_rng(random_state)

    # Select normal samples without replacement.
    selected_normal = rng.choice(
        normal_positions,
        size=target_normal,
        replace=False
    )

    # Keep every fraud sample.
    selected_positions = np.concatenate([
        selected_normal,
        fraud_positions
    ])

    # Shuffle training rows.
    selected_positions = rng.permutation(
        selected_positions
    )

    X_new = (
        X_train.iloc[selected_positions]
        .reset_index(drop=True)
    )

    y_new = (
        y_train.iloc[selected_positions]
        .reset_index(drop=True)
    )

    ids_new = np.asarray(row_ids)[
        selected_positions
    ].copy()

    return X_new, y_new, ids_new


# ==========================================================
# 7. SMOTE Oversampling
# ==========================================================

def smote_oversampling(
    X_train,
    y_train,
    row_ids,
    sampling_ratio=MOTE_RATIO,
    k_neighbors=5,
    random_state=42
):
    """
    Apply SMOTE to Training Set only.

    sampling_ratio:
        Target Fraud / Normal ratio after SMOTE.
        1.0 means Normal : Fraud = 1:1.

    Returns:
        X_new, y_new, ids_new, info
    """

    if not (0 < sampling_ratio <= 1):
        raise ValueError(
            "SMOTE sampling_ratio must be in (0, 1]."
        )

    if k_neighbors < 1:
        raise ValueError(
            "k_neighbors must be at least 1."
        )

    original_counts = y_train.value_counts()

    normal_count = int(original_counts.get(0, 0))
    fraud_count = int(original_counts.get(1, 0))

    if fraud_count <= k_neighbors:
        raise ValueError(
            "Not enough fraud samples for SMOTE."
        )

    target_fraud = int(
        sampling_ratio * normal_count
    )

    if target_fraud <= fraud_count:
        raise ValueError(
            "SMOTE target must exceed original fraud count."
        )

    smote = SMOTE(
        sampling_strategy={1: target_fraud},
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    # Convert results to DataFrame / Series
    X_new = pd.DataFrame(
        X_resampled,
        columns=X_train.columns
    ).reset_index(drop=True)

    y_new = pd.Series(
        np.asarray(y_resampled),
        name="Class"
    ).reset_index(drop=True)

    # SMOTE retains original rows and appends synthetic rows.
    # Synthetic rows have no original row_id.
    n_original = len(X_train)
    n_synthetic = len(X_new) - n_original

    ids_new = np.empty(len(X_new), dtype=object)

    ids_new[:n_original] = np.asarray(row_ids)
    ids_new[n_original:] = None

    info = {
        "method": "SMOTE",
        "sampling_ratio": sampling_ratio,
        "k_neighbors": k_neighbors,
        "original_normal": normal_count,
        "original_fraud": fraud_count,
        "synthetic_fraud": n_synthetic,
        "final_normal": int((y_new == 0).sum()),
        "final_fraud": int((y_new == 1).sum())
    }

    return X_new, y_new, ids_new, info


# ==========================================================
# K-Medoids UnderSampling (candidate-based)
# ==========================================================

def kmedoids_undersampling(
    X_train,
    y_train,
    row_ids,
    ratio=1,
    random_state=42,
    candidate_multiplier=1.25,
    max_candidates=5000,
    max_iter=20
):
    """
    Candidate-based K-Medoids UnderSampling.

    ratio:
        Target Normal : Fraud = 1:1, 5:1, or 10:1.

    Only normal samples participate in clustering.
    All fraud samples are retained.

    Candidate-based approximation:
        First randomly select a normal candidate pool.
        Then run KMedoids on that pool.
    """

    if ratio not in (1, 5, 10):
        raise ValueError("Ratio must be 1, 5, or 10.")

    if candidate_multiplier <= 1:
        raise ValueError(
            "candidate_multiplier must be > 1."
        )

    labels = y_train.to_numpy()

    normal_positions = np.flatnonzero(labels == 0)
    fraud_positions = np.flatnonzero(labels == 1)

    n_fraud = len(fraud_positions)
    n_normal = len(normal_positions)

    if n_fraud == 0:
        raise ValueError("No fraud samples found.")

    n_medoids = n_fraud * ratio

    if n_medoids > n_normal:
        raise ValueError(
            "Not enough normal samples for requested ratio."
        )

    # Number of normal candidates.
    candidate_size = min(
        n_normal,
        max_candidates,
        max(
            n_medoids + 1,
            int(np.ceil(
                n_medoids * candidate_multiplier
            ))
        )
    )

    if candidate_size <= n_medoids:
        raise ValueError(
            "Candidate pool must be larger than n_medoids."
        )

    rng = np.random.default_rng(random_state)

    # Randomly select candidate normal transactions.
    candidate_positions = rng.choice(
        normal_positions,
        size=candidate_size,
        replace=False
    )

    X_candidates = X_train.iloc[
        candidate_positions
    ].to_numpy(dtype=float)

    # K-Medoids on the candidate pool.
    model = KMedoids(
        n_clusters=n_medoids,
        metric="euclidean",
        method="alternate",
        init="k-medoids++",
        max_iter=max_iter,
        random_state=random_state
    )

    model.fit(X_candidates)

    # Medoids are actual candidate observations.
    medoid_local_positions = model.medoid_indices_

    selected_normal = candidate_positions[
        medoid_local_positions
    ]

    # Retain all original fraud transactions.
    selected_positions = np.concatenate([
        selected_normal,
        fraud_positions
    ])

    selected_positions = rng.permutation(
        selected_positions
    )

    X_new = (
        X_train.iloc[selected_positions]
        .reset_index(drop=True)
    )

    y_new = (
        y_train.iloc[selected_positions]
        .reset_index(drop=True)
    )

    ids_new = np.asarray(row_ids)[
        selected_positions
    ].copy()

    # Validate selected sample counts.
    assert int((y_new == 0).sum()) == n_medoids
    assert int((y_new == 1).sum()) == n_fraud

    info = {
        "method": "candidate_kmedoids",
        "ratio": ratio,
        "candidate_size": candidate_size,
        "candidate_multiplier": candidate_multiplier,
        "n_medoids": n_medoids,
        "max_iter": max_iter,
        "random_state": random_state,
        "normal_before": n_normal,
        "normal_after": n_medoids,
        "fraud_retained": n_fraud
    }

    return X_new, y_new, ids_new, info


# ==========================================================
# Unified sampling interface: P0/P1/P2/P5/P6
# ==========================================================

def prepare_sampling_data(
    data,
    sampling="none",
    sampling_ratio=None,
    outlier_removal="none",
    iqr_feature=IQR_FEATURE,
    iqr_factor=IQR_FACTOR,
    random_state=RANDOM_STATE,
    smote_k_neighbors=5,
    kmedoids_candidate_multiplier=1.25,
    kmedoids_max_candidates=5000,
    kmedoids_max_iter=20
):
    """
    Supported configurations:

    P0:
        sampling="none"
        outlier_removal="none"

    P1:
        sampling="random"
        outlier_removal="none"
        sampling_ratio=1, 5, or 10

    P2:
        sampling="random"
        outlier_removal="normal_iqr"
        sampling_ratio=1, 5, or 10

    P5:
        sampling="smote"
        outlier_removal="none"
        sampling_ratio=1.0

    P6:
        sampling="smote"
        outlier_removal="normal_iqr"
        sampling_ratio=1.0
    """

    if sampling not in ("none", "random", "smote", "kmedoids"):
        raise ValueError("Unsupported sampling method.")

    if outlier_removal not in ("none", "normal_iqr"):
        raise ValueError("Invalid outlier removal.")

    if sampling == "none":
        if outlier_removal != "none":
            raise ValueError(
                "P0 does not use outlier removal."
            )

    if sampling == "random":
        if sampling_ratio not in (1, 5, 10):
            raise ValueError(
                "Random UnderSampling ratio must be 1, 5, or 10."
            )

    
    if sampling == "smote":
        if sampling_ratio is None or not (0 < sampling_ratio <= 1):
            raise ValueError(
                "SMOTE ratio must be between 0 and 1."
            )

    if sampling == "kmedoids":
        if sampling_ratio not in (1, 5, 10):
            raise ValueError(
                "K-Medoids ratio must be 1, 5, or 10."
            )

    # Copy original Training Set.
    X = data["X_train"].copy()
    y = data["y_train"].copy()
    ids = data["train_row_ids"].copy()

    iqr_info = None
    smote_info = None
    kmedoids_info = None

    # ------------------------------------------------------
    # Step A. Normal-only IQR
    # ------------------------------------------------------

    if outlier_removal == "normal_iqr":

        X, y, ids, iqr_info = remove_normal_outliers(
            X,
            y,
            ids,
            feature=iqr_feature,
            factor=iqr_factor
        )

    # ------------------------------------------------------
    # Step B. Sampling
    # ------------------------------------------------------

    if sampling == "random":

        X, y, ids = random_undersampling(
            X,
            y,
            ids,
            ratio=sampling_ratio,
            random_state=random_state
        )

    elif sampling == "smote":

        X, y, ids, smote_info = smote_oversampling(
            X,
            y,
            ids,
            sampling_ratio=sampling_ratio,
            k_neighbors=smote_k_neighbors,
            random_state=random_state
        )
    elif sampling == "kmedoids":

        X, y, ids, kmedoids_info = kmedoids_undersampling(
            X,
            y,
            ids,
            ratio=sampling_ratio,
            random_state=random_state,
            candidate_multiplier=kmedoids_candidate_multiplier,
            max_candidates=kmedoids_max_candidates,
            max_iter=kmedoids_max_iter
        )

    # ------------------------------------------------------
    # Step C. Data leakage checks
    # ------------------------------------------------------

    # Ignore synthetic rows (row_id=None).
    original_ids = {
        int(row_id)
        for row_id in ids
        if row_id is not None
    }

    assert original_ids.isdisjoint(
        set(data["val_row_ids"])
    )

    assert original_ids.isdisjoint(
        set(data["test_row_ids"])
    )

    # Validation and Test remain unchanged.
    return {
        **data,
        "X_train": X,
        "y_train": y,
        "train_row_ids": ids,

        "X_val": data["X_val"],
        "y_val": data["y_val"],

        "X_test": data["X_test"],
        "y_test": data["y_test"],

        "scaler": data["scaler"],

        "metadata": {
            **data["metadata"],
            "sampling": sampling,
            "sampling_ratio": sampling_ratio,
            "outlier_removal": outlier_removal,
            "iqr": iqr_info,
            "smote": smote_info,
            "random_state": random_state,
            "kmedoids": kmedoids_info,
        }
    }

# ==========================================================
# 8. Isolation Forest contamination ablation (C0-C4)
# ==========================================================

def prepare_contamination_data(
    data,
    fraud_counts=None,
    random_state=42
):
    X_train = data["X_train"]
    y_train = data["y_train"]

    labels = y_train.to_numpy()

    normal_positions = np.flatnonzero(labels == 0)
    fraud_positions = np.flatnonzero(labels == 1)

    if fraud_counts is None:
        limit = min(360, len(fraud_positions))
        counts = tuple(int(limit * fraction) for fraction in (0, 1/12, 1/4, 1/2, 1))
    else:
        counts = tuple(fraud_counts)

    if (
        any(
            c < 0 or c > len(fraud_positions)
            for c in counts
        )
    ):
        raise ValueError("Invalid fraud counts.")

    rng = np.random.default_rng(random_state)

    # One permutation ensures nested fraud subsets.
    fraud_order = rng.permutation(fraud_positions)

    experiments = {}

    for i, fraud_count in enumerate(counts):

        selected = np.concatenate([
            normal_positions,
            fraud_order[:fraud_count]
        ])

        selected = np.random.default_rng(
            random_state + i
        ).permutation(selected)

        total = len(selected)

        experiments[f"C{i}"] = {
            **data,
            "X_train": (
                X_train.iloc[selected]
                .reset_index(drop=True)
            ),
            "y_train": (
                y_train.iloc[selected]
                .reset_index(drop=True)
            ),
            "train_row_ids": (
                data["train_row_ids"][selected].copy()
            ),

            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],

            "metadata": {
                **data["metadata"],
                "mode": f"C{i}",
                "normal_count": len(normal_positions),
                "fraud_count": fraud_count,
                "total_count": total,
                "contamination_pct": (
                    fraud_count / total * 100
                )
            }
        }

    return experiments


# ==========================================================
# 9. Main: verify all experiments
# ==========================================================

if __name__ == "__main__":

    csv_path = (
        Path(__file__).resolve().parent
        / "data"
        / "creditcard.csv"
    )

    data = prepare_data(csv_path)

    print("\n=== Chronological Split ===")

    for name in ("train", "val", "test"):
        y = data[f"y_{name}"]

        print(
            f"{name:5s} | "
            f"Normal={(y == 0).sum():6d} | "
            f"Fraud={(y == 1).sum():3d} | "
            f"Total={len(y):6d}"
        )

    print("\n=== C0-C4 Contamination ===")

    contamination = prepare_contamination_data(data)

    for name, exp in contamination.items():
        m = exp["metadata"]

        print(
            f"{name} | "
            f"Normal={m['normal_count']} | "
            f"Fraud={m['fraud_count']} | "
            f"Contamination={m['contamination_pct']:.4f}%"
        )

    print("\n=== Version 2: P0 / P1 / P2 ===")

    configs = [
        ("P0", "none", "none", None)
    ]

    for ratio in (1, 5, 10):
        configs.append(
            (f"P1-{ratio}", "random", "none", ratio)
        )

    for ratio in (1, 5, 10):
        configs.append(
            (f"P2-{ratio}", "random", "normal_iqr", ratio)
        )

    for name, sampling, outlier, ratio in configs:

        exp = prepare_sampling_data(
            data,
            sampling=sampling,
            sampling_ratio=ratio,
            outlier_removal=outlier,
            iqr_feature=IQR_FEATURE,
            iqr_factor=IQR_FACTOR,
            random_state=RANDOM_STATE
        )

        y = exp["y_train"]
        iqr_info = exp["metadata"]["iqr"]

        removed = (
            iqr_info["normal_removed"]
            if iqr_info is not None
            else 0
        )

        print(
            f"{name:6s} | "
            f"Normal={(y == 0).sum():5d} | "
            f"Fraud={(y == 1).sum():3d} | "
            f"Total={len(y):5d} | "
            f"IQR Feature: {IQR_FEATURE} | "
            f"IQR Factor: {IQR_FACTOR} | "
            f"IQR removed={removed}"
        )

    # ======================================================
    # Version 3: SMOTE Experiments
    # ======================================================

    print("\n=== Version 3: P5 / P6 ===")

    
    smote_configs = [
        ("P5", "none"),
        ("P6", "normal_iqr")
    ]

    for name, outlier_method in smote_configs:

        exp = prepare_sampling_data(
            data,
            sampling="smote",
            sampling_ratio=MOTE_RATIO,
            outlier_removal=outlier_method,
            iqr_feature=IQR_FEATURE,
            iqr_factor=IQR_FACTOR,
            random_state=RANDOM_STATE,
            smote_k_neighbors=5
        )

        y = exp["y_train"]

        smote_info = exp["metadata"]["smote"]
        iqr_info = exp["metadata"]["iqr"]

        removed = (
            iqr_info["normal_removed"]
            if iqr_info is not None
            else 0
        )

        print(
            f"{name:3s} | "
            f"Normal={(y == 0).sum():6d} | "
            f"Fraud={(y == 1).sum():6d} | "
            f"Total={len(y):6d} | "
            f"Synthetic={smote_info['synthetic_fraud']:6d} | "
            f"IQR removed={removed}"
        )

        # Verify that Validation and Test are unchanged.
        assert exp["X_val"].equals(data["X_val"])
        assert exp["y_val"].equals(data["y_val"])
        assert exp["X_test"].equals(data["X_test"])
        assert exp["y_test"].equals(data["y_test"])

  
    # ======================================================
    # Version 4: K-Medoids Experiments
    # ======================================================

    print("\n=== Version 4: P3 / P4 ===")

    kmedoids_configs = []

    for ratio in (1, 5, 10):
        kmedoids_configs.append(
            (f"P3-{ratio}", "none", ratio)
        )

    for ratio in (1, 5, 10):
        kmedoids_configs.append(
            (f"P4-{ratio}", "normal_iqr", ratio)
        )

    for name, outlier_method, ratio in kmedoids_configs:

        print(f"Running {name} ...", flush=True)

        exp = prepare_sampling_data(
            data,
            sampling="kmedoids",
            sampling_ratio=ratio,
            outlier_removal=outlier_method,
            iqr_feature=IQR_FEATURE,
            iqr_factor=IQR_FACTOR,
            random_state=RANDOM_STATE,
            kmedoids_candidate_multiplier=1.25,
            kmedoids_max_candidates=5000,
            kmedoids_max_iter=20
        )

        y = exp["y_train"]
        info = exp["metadata"]["kmedoids"]
        iqr_info = exp["metadata"]["iqr"]

        removed = (
            iqr_info["normal_removed"]
            if iqr_info is not None
            else 0
        )

        print(
            f"{name:6s} | "
            f"Normal={(y == 0).sum():5d} | "
            f"Fraud={(y == 1).sum():3d} | "
            f"Total={len(y):5d} | "
            f"Candidates={info['candidate_size']} | "
            f"IQR removed={removed}",
            flush=True
        )

        # All selected original rows belong to Training Set.
        assert set(exp["train_row_ids"]).issubset(
            set(data["train_row_ids"])
        )

        # Validation / Test remain unchanged.
        assert exp["X_val"].equals(data["X_val"])
        assert exp["y_val"].equals(data["y_val"])
        assert exp["X_test"].equals(data["X_test"])
        assert exp["y_test"].equals(data["y_test"])

    print("\nVersion 4 validation passed.")


