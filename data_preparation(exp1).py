
"""Data Mining HW2 - Experiment 1: Stratified 5-fold CV.

Class 0 = Normal
Class 1 = Fraud

Features:
- Stratified 20% holdout test, then 5-fold CV on remaining 80%
- Train-only RobustScaler for Time and Amount
- Normal-only IQR removal
- Random UnderSampling (1:1, 5:1, 10:1)
- Isolation Forest contamination ablation C0-C4
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
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
MOTE_RATIO = 0.01  # Normal : Fraud = 100:1

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
# 2. Stratified holdout test + StratifiedKFold on development data
# ==========================================================

def make_stratified_folds(df, test_ratio=0.20, n_splits=5,
                          random_state=42):
    """Yield (fold_index, train_df, val_df, test_df).

    Test set is held out ONCE and is never included in CV.
    Each fold has ~64/16/20% of original data (train/val/test).
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    dev_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["Class"],
        random_state=random_state, shuffle=True
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(
        cv.split(dev_df[FEATURES], dev_df["Class"])
    ):
        yield (fold, dev_df.iloc[train_idx].copy(),
               dev_df.iloc[val_idx].copy(), test_df.copy())


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
# 4. Fold-specific preparation (P0)
# ==========================================================

def prepare_data_folds(csv_path="data/creditcard.csv", test_ratio=0.20,
                       n_splits=5, random_state=42, fit_normal_only=False):
    """Yield (fold_index, data), with a scaler fitted only on fold train."""
    df = load_data(csv_path)
    for fold, train_df, val_df, test_df in make_stratified_folds(
        df, test_ratio=test_ratio, n_splits=n_splits,
        random_state=random_state
    ):
        data = scale_features(train_df, val_df, test_df, fit_normal_only=fit_normal_only)
        train_ids = set(data["train_row_ids"])
        val_ids = set(data["val_row_ids"])
        test_ids = set(data["test_row_ids"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        data["metadata"] = {
            **data["metadata"],
            "split": "stratified_5fold_with_holdout_test",
            "fold": fold, "n_splits": n_splits,
            "scaling": "train_fold_only_RobustScaler",
            "sampling": "none", "random_state": random_state
        }
        yield fold, data


def prepare_data(csv_path="data/creditcard.csv", fold_index=0,
                 test_ratio=0.20, n_splits=5, random_state=42, fit_normal_only=False):
    """Convenience: return one fold. For evaluation, use ALL folds."""
    if not 0 <= fold_index < n_splits:
        raise ValueError("fold_index out of range")
    for fold, data in prepare_data_folds(
        csv_path, test_ratio=test_ratio, n_splits=n_splits,
        random_state=random_state, fit_normal_only=fit_normal_only
    ):
        if fold == fold_index:
            return data
    raise RuntimeError("Requested fold not found")


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
# 9. Execute P0-P6 for each stratified fold
# ==========================================================

def sampling_configurations(smote_ratio=MOTE_RATIO):
    """15 configurations: P0(1), P1-P4(3 each), P5-P6(1 each)."""
    configs = [("P0", "none", "none", None)]
    for prefix, sampling, outlier in (
        ("P1", "random", "none"),
        ("P2", "random", "normal_iqr"),
        ("P3", "kmedoids", "none"),
        ("P4", "kmedoids", "normal_iqr"),
    ):
        for ratio in (1, 5, 10):
            configs.append((f"{prefix}-{ratio}", sampling, outlier, ratio))
    configs += [
        ("P5", "smote", "none", smote_ratio),
        ("P6", "smote", "normal_iqr", smote_ratio),
    ]
    # Present experiments in the same Version 2 / 3 / 4 order as exp2.
    return sorted(configs, key=lambda c: (0 if c[0].startswith(("P0", "P1", "P2"))
                                           else 1 if c[0].startswith(("P5", "P6")) else 2))


def run_sampling_experiments(csv_path, output_dir="outputs/exp1",
                             selected_groups=None, smote_ratio=MOTE_RATIO,
                             save_datasets=False, n_splits=5):
    """Run fold-specific preprocessing and print results without saving CSV.

    Set save_datasets=True only when NPZ exports are needed.
    """
    if save_datasets:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    configs = sampling_configurations(smote_ratio)
    if selected_groups is not None:
        configs = [c for c in configs if c[0].split("-")[0] in selected_groups]
    if not configs:
        raise ValueError("No configurations selected")

    records = []
    # Version 為外層，Fold 為內層；每個 Fold 的資料仍獨立處理。
    versions = [
        ("Version 2 (Random UnderSampling)", ("P0", "P1", "P2")),
        ("Version 3 (SMOTE OverSampling)", ("P5", "P6")),
        ("Version 4 (K-Medoids UnderSampling)", ("P3", "P4")),
    ]
    for version_title, prefixes in versions:
        version_configs = [c for c in configs if c[0].split("-")[0] in prefixes]
        if not version_configs:
            continue
        print(f"\n=== {version_title} ===", flush=True)
        for fold, data in prepare_data_folds(csv_path, n_splits=n_splits):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---", flush=True)
            y_train, y_val, y_test = (data[f"y_{part}"] for part in ("train", "val", "test"))
            for label, y_part in (("train", y_train), ("val", y_val), ("test", y_test)):
                print(f"{label:5s} | Normal={(y_part == 0).sum():6d} | Fraud={(y_part == 1).sum():3d} | Total={len(y_part):6d}")
            val_ids = set(data["val_row_ids"])
            test_ids = set(data["test_row_ids"])
            for name, sampling, outlier, ratio in version_configs:
                try:
                    exp = prepare_sampling_data(
                        data, sampling=sampling, sampling_ratio=ratio,
                        outlier_removal=outlier, iqr_feature=IQR_FEATURE,
                        iqr_factor=IQR_FACTOR, random_state=RANDOM_STATE,
                    )
                    ids = exp["train_row_ids"]
                    original_ids = {int(i) for i in ids if i is not None}
                    assert original_ids.isdisjoint(val_ids | test_ids)
                    for split in ("val", "test"):
                        assert exp[f"X_{split}"].equals(data[f"X_{split}"])
                        assert exp[f"y_{split}"].equals(data[f"y_{split}"])
                    y = exp["y_train"]
                    metadata = exp["metadata"]
                    iqr = metadata.get("iqr") or {}
                    smote = metadata.get("smote") or {}
                    kmedoids = metadata.get("kmedoids") or {}
                    record = dict(
                        fold=fold + 1, experiment=name, sampling=sampling,
                        sampling_ratio=ratio, outlier_removal=outlier,
                        train_normal=int((y == 0).sum()),
                        train_fraud=int((y == 1).sum()),
                        train_total=len(y),
                        train_fraud_pct=float(y.mean() * 100),
                        val_normal=int((data["y_val"] == 0).sum()),
                        val_fraud=int((data["y_val"] == 1).sum()),
                        test_normal=int((data["y_test"] == 0).sum()),
                        test_fraud=int((data["y_test"] == 1).sum()),
                        iqr_removed=iqr.get("normal_removed", 0),
                        synthetic_fraud=smote.get("synthetic_fraud", 0),
                        kmedoids_candidates=kmedoids.get("candidate_size", ""),
                        status="success", error="",
                    )
                    if save_datasets:
                        target = output_dir / f"fold{fold+1}_{name}.npz"
                        np.savez_compressed(
                            target, X_train=exp["X_train"].to_numpy(),
                            y_train=y.to_numpy(),
                            train_row_ids=np.asarray(ids, dtype=object),
                            X_val=data["X_val"].to_numpy(),
                            y_val=data["y_val"].to_numpy(),
                            X_test=data["X_test"].to_numpy(),
                            y_test=data["y_test"].to_numpy(),
                            feature_names=np.array(FEATURES),
                        )
                    detail = (f" | Synthetic={record['synthetic_fraud']:5d}" if sampling == "smote"
                              else f" | Candidates={record['kmedoids_candidates']}" if sampling == "kmedoids"
                              else "")
                    print(f"{name:6s} | Normal={record['train_normal']:6d} | "
                          f"Fraud={record['train_fraud']:4d} | Total={record['train_total']:6d}"
                          f"{detail} | IQR removed={record['iqr_removed']}", flush=True)
                except (ValueError, AssertionError, MemoryError) as exc:
                    record = dict(fold=fold+1, experiment=name,
                                  sampling=sampling, sampling_ratio=ratio,
                                  outlier_removal=outlier, status="failed",
                                  error=str(exc))
                    print(f"{name:6s} | FAILED: {exc}", flush=True)
                records.append(record)
    summary = pd.DataFrame(records)
    success = int((summary["status"] == "success").sum())
    print(f"Completed: {success}/{len(summary)} configurations successful.")
    if success < len(summary):
        print(summary.loc[summary["status"] != "success", ["fold", "experiment", "error"]].to_string(index=False))
    return summary


# ==========================================================
# Public API for model-training teammates
# ==========================================================

def available_modes():
    """Return all supported P0-P6 modes and contamination modes."""
    return [item[0] for item in sampling_configurations()] + [f"C{i}" for i in range(5)]


def _apply_mode(base_data, mode, *, smote_ratio=MOTE_RATIO,
                random_state=RANDOM_STATE):
    """Apply one named preprocessing mode to an already prepared fold."""
    mode = str(mode).upper().strip()
    if mode in {f"C{i}" for i in range(5)}:
        return prepare_contamination_data(base_data, random_state=random_state)[mode]
    config = {name: (sampling, outlier, ratio)
              for name, sampling, outlier, ratio
              in sampling_configurations(smote_ratio)}
    if mode not in config:
        raise ValueError(f"Unknown mode {mode!r}; choose from {available_modes()}")
    sampling, outlier, ratio = config[mode]
    result = prepare_sampling_data(
        base_data, sampling=sampling, sampling_ratio=ratio,
        outlier_removal=outlier, random_state=random_state,
    )
    result["metadata"]["mode"] = mode
    return result


def get_prepared_data(mode="P0", fold_index=0,
                      csv_path=None, n_splits=5,
                      test_ratio=0.20, random_state=RANDOM_STATE,
                      smote_ratio=MOTE_RATIO):
    """One-call API: return processed train/val/test for one zero-based fold.

    Example: get_prepared_data(mode="P2-5", fold_index=0).
    No files are written. Validation and test are never resampled.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "data" / "creditcard.csv"
    base = prepare_data(csv_path=csv_path, fold_index=fold_index,
                        test_ratio=test_ratio, n_splits=n_splits,
                        random_state=random_state)
    return _apply_mode(base, mode, smote_ratio=smote_ratio,
                       random_state=random_state)


def iter_prepared_folds(mode="P0", csv_path=None, n_splits=5,
                        test_ratio=0.20, random_state=RANDOM_STATE,
                        smote_ratio=MOTE_RATIO):
    """Yield (zero-based fold index, processed data) for cross-validation.

    Each fold has its own training-fitted scaler and preprocessing.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "data" / "creditcard.csv"
    for fold, base in prepare_data_folds(
        csv_path=csv_path, test_ratio=test_ratio,
        n_splits=n_splits, random_state=random_state
    ):
        yield fold, _apply_mode(base, mode, smote_ratio=smote_ratio,
                                random_state=random_state)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp1: 5-fold P0-P6 preparation")
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parent / "data" / "creditcard.csv"))
    parser.add_argument("--output-dir", default="outputs/exp1")
    parser.add_argument("--groups", nargs="+", choices=[f"P{i}" for i in range(7)],
                        help="Select groups, e.g. --groups P0 P1 P2")
    parser.add_argument("--smote-ratio", type=float, default=MOTE_RATIO,
                        help="Target fraud/normal ratio; 0.01 = 100:1")
    parser.add_argument("--save-datasets", action="store_true",
                        help="Also save compressed train/val/test arrays per configuration")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run_sampling_experiments(
        args.csv, output_dir=args.output_dir,
        selected_groups=args.groups, smote_ratio=args.smote_ratio,
        save_datasets=args.save_datasets, n_splits=args.folds,
    )
