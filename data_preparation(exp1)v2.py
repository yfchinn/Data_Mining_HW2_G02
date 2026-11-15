
"""Data Mining HW2 - Experiment 1: fixed holdouts + train-only 5-fold CV.

Class 0 = Normal
Class 1 = Fraud

Features:
- P1-P4: fixed validation/test counts first, train-only preprocessing, then train-only 5-fold CV
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
MOTE_RATIO = 1.0  # P5/P6: Normal : Fraud = 1:1 (train-only)

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
# 8. Exp1: fixed holdouts BEFORE training-only sampling (P1-P4)
# ==========================================================
# Table 5 counts are exact for the OUTER training set. Validation and test
# are selected first, never filtered, and are shared by paired P1/P2 modes.
# Across ratios, the held-out populations differ as specified by Table 5;
# therefore cross-ratio metrics are NOT common-test comparisons.
# Within 5-fold CV, IQR/RUS/K-Medoids are refitted on each CV-training fold.

CONFIGS = {
    "P0": ("none", "none", None),
    **{f"P1-{r}": ("random", "none", r) for r in (1, 5, 10)},
    **{f"P2-{r}": ("random", "normal_iqr", r) for r in (1, 5, 10)},
    **{f"P3-{r}": ("kmedoids", "none", r) for r in (10,)},
    **{f"P4-{r}": ("kmedoids", "normal_iqr", r) for r in (10,)},
    "P5": ("smote", "none", 1.0),
    "P6": ("smote", "normal_iqr", 1.0),
}
VERSIONS = (
    ("Version 1 (Original Data)", ("P0",)),
    ("Version 2 (Random UnderSampling)", ("P1", "P2")),
    ("Version 3 (SMOTE OverSampling)", ("P5", "P6")),
    ("Version 4 (Random UnderSampling + K-Medoids)", ("P3", "P4")),
)

def sampling_configurations(smote_ratio=MOTE_RATIO):
    return [(name, *config) for name, config in CONFIGS.items()]


def _split_622(frame, seed=RANDOM_STATE):
    """One stratified 60/20/20 split; both classes retain ~60/20/20."""
    train, remaining = train_test_split(
        frame, train_size=0.60, stratify=frame["Class"],
        random_state=seed, shuffle=True)
    val, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["Class"],
        random_state=seed, shuffle=True)
    return (part.reset_index(drop=True) for part in (train, val, test))


def _iqr_filter_normal(frame, feature=IQR_FEATURE, factor=IQR_FACTOR):
    normal = frame["Class"].eq(0)
    q1, q3 = np.quantile(frame.loc[normal, feature], [0.25, 0.75])
    low, high = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    keep = ~normal | frame[feature].between(low, high)
    return frame.loc[keep].copy().reset_index(drop=True), int((~keep).sum())


def _random_normal_subset(frame, ratio, seed=RANDOM_STATE):
    normal = frame.loc[frame["Class"].eq(0)]
    fraud = frame.loc[frame["Class"].eq(1)]
    n = ratio * len(fraud)
    if n > len(normal):
        raise ValueError(f"Need {n} normal rows; only {len(normal)} available")
    chosen = normal.sample(n=n, random_state=seed, replace=False)
    return pd.concat([chosen, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _counts(frame):
    labels = frame["Class"] if "Class" in frame else frame
    n = int((labels == 0).sum())
    f = int((labels == 1).sum())
    return f"Normal={n:6d} | Fraud={f:4d} | Total={n+f:6d}"


def _kmedoids_on_frame(frame, seed=RANDOM_STATE):
    """Select normal medoids to reach N:F=5:1, retaining all fraud.

    If input is already 5:1, selecting fewer normal rows would violate
    the specified final ratio. In that case this is an explicit no-op.
    """
    normal_count = int(frame["Class"].eq(0).sum())
    fraud_count = int(frame["Class"].eq(1).sum())
    target = 5 * fraud_count
    if normal_count < target and target - normal_count > 5:
        raise ValueError(f"K-Medoids 5:1 impossible: {normal_count} normal, {fraud_count} fraud")
    if abs(normal_count - target) <= 5:
        return frame.copy().reset_index(drop=True), {"method": "skipped_already_5_to_1", "normal_before": normal_count, "normal_after": normal_count}
    X = frame[FEATURES].reset_index(drop=True)
    y = frame["Class"].reset_index(drop=True)
    ids = frame["row_id"].to_numpy()
    # Fit scaling on this training portion only, before distance-based K-Medoids.
    scaler = RobustScaler().fit(X.loc[y.eq(0), ["Time", "Amount"]])
    X = X.copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    X2, y2, ids2, info = kmedoids_undersampling(
        X, y, ids, ratio=5, random_state=seed)
    # Use original, unscaled rows; downstream scaler is fitted independently.
    indexed = frame.set_index("row_id", drop=False)
    return indexed.loc[ids2].reset_index(drop=True), info


# Exact outer split sizes requested in Table 5 (Normal, Fraud).
TABLE5_HOLDOUTS = {
    1: {"val": (98, 99), "test": (99, 98)},
    5: {"val": (492, 98), "test": (492, 99)},
    10: {"val": (984, 98), "test": (984, 99)},
}
# Table 5 reports 1,476 rather than 5*295=1,475 for P1-5/P2-5.
TABLE5_TRAIN_NORMAL = {1: 295, 5: 1476, 10: 2952}


def _fixed_table5_split(original, ratio, seed=RANDOM_STATE):
    """Choose exact untouched holdouts first; retain remaining rows for train.

    P1/P2 at a given ratio (and P3/P4 at ratio 10) use the same
    held-out row IDs, independently of the preprocessing method.
    """
    counts = TABLE5_HOLDOUTS[ratio]
    normal = original.loc[original["Class"].eq(0)]
    fraud = original.loc[original["Class"].eq(1)]
    if len(fraud) != 492:
        raise ValueError("Exact Table 5 counts require 492 fraud rows.")
    rng_n = np.random.default_rng(seed)
    rng_f = np.random.default_rng(seed + 1000)
    normal_order = rng_n.permutation(normal.index.to_numpy())
    fraud_order = rng_f.permutation(fraud.index.to_numpy())
    nv, fv = counts["val"]
    nt, ft = counts["test"]
    val_ids = np.r_[normal_order[:nv], fraud_order[:fv]]
    test_ids = np.r_[normal_order[nv:nv+nt], fraud_order[fv:fv+ft]]
    held_out = set(val_ids) | set(test_ids)
    train = original.loc[~original.index.isin(held_out)].copy()
    val = original.loc[val_ids].copy()
    test = original.loc[test_ids].copy()
    assert len(held_out) == len(val) + len(test)
    assert int(train["Class"].eq(1).sum()) == 295
    return (x.reset_index(drop=True) for x in (train, val, test))


def _make_mode_frames(original, mode, seed=RANDOM_STATE):
    """Return raw outer train pool and untouched outer validation/test.

    Training preprocessing is intentionally delayed to _prepare_scaled,
    including for IQR, so no validation/test or CV-val rows are used.
    """
    if mode.startswith(("P1-", "P2-", "P3-", "P4-")):
        ratio = CONFIGS[mode][2]
        train, val, test = _fixed_table5_split(original, ratio, seed)
    else:
        train, val, test = _split_622(original, seed)
    return train, val, test, 0


def _prepare_scaled(train, val, test, mode, fold=None, seed=RANDOM_STATE):
    sampling, outlier, ratio = CONFIGS[mode]
    k_info = None
    smote_info = None
    removed = 0
    train = train.copy().reset_index(drop=True)
    if outlier == "normal_iqr":
        train, removed = _iqr_filter_normal(train)
    if mode.startswith(("P1-", "P2-", "P3-", "P4-")):
        if sampling == "random":
            if fold is None:
                target = TABLE5_TRAIN_NORMAL[ratio]
                fraud_count = int(train["Class"].eq(1).sum())
                if fraud_count != 295:
                    raise AssertionError("Unexpected outer training fraud count")
                # Explicit outer count matches the supplied Table 5 exactly.
                normal = train.loc[train["Class"].eq(0)].sample(
                    n=target, random_state=seed, replace=False)
                fraud = train.loc[train["Class"].eq(1)]
                train = pd.concat([normal, fraud]).sample(
                    frac=1, random_state=seed).reset_index(drop=True)
            else:
                train = _random_normal_subset(train, ratio, seed)
        elif sampling == "kmedoids":
            # Stage 1: train-only RUS to 10:1; stage 2: train-only
            # candidate K-Medoids to 5:1 (1,475/295 in outer train).
            train = _random_normal_subset(train, 10, seed)
            train, k_info = _kmedoids_on_frame(train, seed)
    data = scale_features(train, val, test)
    if sampling == "smote":
        X, y, ids, smote_info = smote_oversampling(
            data["X_train"], data["y_train"], data["train_row_ids"],
            sampling_ratio=1.0, random_state=seed)
        data.update(X_train=X, y_train=y, train_row_ids=ids)
    ids = {int(x) for x in data["train_row_ids"] if x is not None}
    assert ids.isdisjoint(set(data["val_row_ids"]))
    assert ids.isdisjoint(set(data["test_row_ids"]))
    assert set(data["val_row_ids"]).isdisjoint(set(data["test_row_ids"]))
    data["metadata"].update(
        mode=mode, fold=fold, kmedoids=k_info, smote=smote_info,
        iqr_removed=removed,
        split="table5_fixed_holdouts_then_train_only_preprocessing"
        if mode.startswith(("P1-", "P2-", "P3-", "P4-"))
        else "original_split_then_train_only_preprocessing")
    return data


def get_prepared_data(mode="P0", fold_index=None, csv_path=None,
                      n_splits=5, random_state=RANDOM_STATE, **kwargs):
    """Return outer train/val/test, or CV train/val plus untouched outer test.

    fold_index is zero-based. For folds, outer validation is not used;
    preprocessing is fitted only on the corresponding CV training portion.
    """
    if mode not in CONFIGS:
        raise ValueError(f"Unknown mode {mode!r}")
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "data" / "creditcard.csv"
    original = load_data(csv_path)
    train, val, test, _ = _make_mode_frames(original, mode, random_state)
    if fold_index is None:
        data = _prepare_scaled(train, val, test, mode, seed=random_state)
        data["metadata"].update(scope="outer_split")
        return data
    if not 0 <= fold_index < n_splits:
        raise ValueError("fold_index out of range")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    for i, (tr, va) in enumerate(cv.split(train[FEATURES], train["Class"])):
        if i == fold_index:
            data = _prepare_scaled(train.iloc[tr].copy(), train.iloc[va].copy(),
                                   test, mode, fold=i + 1, seed=random_state + i)
            data["metadata"].update(
                scope="cv_fold", outer_val_row_ids=val["row_id"].to_numpy())
            return data
    raise RuntimeError("Fold not found")


def iter_prepared_folds(mode="P0", csv_path=None, n_splits=5, **kwargs):
    for fold in range(n_splits):
        yield fold, get_prepared_data(mode, fold, csv_path, n_splits, **kwargs)


def available_modes():
    return list(CONFIGS) + [f"C{i}" for i in range(5)]


def run_sampling_experiments(csv_path, selected_groups=None, n_splits=5,
                             output_dir="outputs/exp1", save_datasets=False,
                             smote_ratio=MOTE_RATIO):
    """Print outer splits and five inner CV folds, organized by version/mode."""
    original = load_data(csv_path)
    records = []
    if save_datasets:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    for title, prefixes in VERSIONS:
        modes = [m for m in CONFIGS if m.split("-")[0] in prefixes
                 and (selected_groups is None or m.split("-")[0] in selected_groups)]
        if not modes:
            continue
        print(f"\n========== {title} ==========", flush=True)
        for mode in modes:
            print(f"\n--- {mode} ---", flush=True)
            train, val, test, removed = _make_mode_frames(original, mode)
            for label, part in (("Train pool (before prep)", train),
                                ("Validation (untouched)", val),
                                ("Test (untouched)", test)):
                print(f"{label:25s} | {_counts(part)}")
            outer = _prepare_scaled(train, val, test, mode)
            print(f"Train after processing    | {_counts(outer['y_train'])}")
            print(f"IQR removed (Normal only): {outer['metadata']['iqr_removed']}")
            if mode.startswith(("P1-", "P2-", "P3-", "P4-")):
                ratio = CONFIGS[mode][2]
                exp_n = 1475 if mode.startswith(("P3-", "P4-")) else TABLE5_TRAIN_NORMAL[ratio]
                assert int(outer["y_train"].eq(0).sum()) == exp_n
                assert int(outer["y_train"].eq(1).sum()) == 295
                assert _counts(val) == _counts(outer["y_val"])
                assert _counts(test) == _counts(outer["y_test"])
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            for i, (tr, va) in enumerate(cv.split(train[FEATURES], train["Class"])):
                # Critical: K-Medoids / SMOTE and RobustScaler are fit only on CV train.
                prepared = _prepare_scaled(train.iloc[tr].copy(), train.iloc[va].copy(),
                                           test, mode, fold=i + 1, seed=RANDOM_STATE + i)
                info = prepared["metadata"].get("kmedoids") or {}
                note = f" | KMedoids={info.get('method', 'candidate_kmedoids')}" if info else ""
                print(f"  Fold {i+1}/{n_splits} | CV train: {_counts(prepared['y_train'])} | CV val: {_counts(prepared['y_val'])}{note}", flush=True)
                records.append({"mode": mode, "fold": i + 1,
                                "train_normal": int((prepared["y_train"] == 0).sum()),
                                "train_fraud": int((prepared["y_train"] == 1).sum()),
                                "cv_val_normal": int((prepared["y_val"] == 0).sum()),
                                "cv_val_fraud": int((prepared["y_val"] == 1).sum()),
                                "outer_val_normal": int(val["Class"].eq(0).sum()),
                                "outer_val_fraud": int(val["Class"].eq(1).sum()),
                                "test_normal": int(test["Class"].eq(0).sum()),
                                "test_fraud": int(test["Class"].eq(1).sum()),
                                "iqr_removed": prepared["metadata"]["iqr_removed"]})
                if save_datasets:
                    np.savez_compressed(output_dir / f"{mode}_fold{i+1}.npz",
                                        X_train=prepared["X_train"].to_numpy(),
                                        y_train=prepared["y_train"].to_numpy(),
                                        X_val=prepared["X_val"].to_numpy(),
                                        y_val=prepared["y_val"].to_numpy(),
                                        X_test=prepared["X_test"].to_numpy(),
                                        y_test=prepared["y_test"].to_numpy())
    print(f"\nCompleted: {len(records)} CV fold preparations", flush=True)
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp1: fixed Table 5 holdouts + train-only preprocessing and CV")
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parent / "data" / "creditcard.csv"))
    parser.add_argument("--groups", nargs="+", choices=[f"P{i}" for i in range(7)], default=["P0", "P1", "P2", "P3", "P4"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--save-datasets", action="store_true")
    parser.add_argument("--output-dir", default="outputs/exp1")
    args = parser.parse_args()
    run_sampling_experiments(args.csv, selected_groups=args.groups,
                             n_splits=args.folds, save_datasets=args.save_datasets,
                             output_dir=args.output_dir)
