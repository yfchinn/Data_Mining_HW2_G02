"""Train-only preprocessing for the ULB credit-card fraud dataset.

Import it after the team's shared split has been established::

    train = df.loc[train_indices]
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(df.loc[val_indices])
    processor = CreditCardPreprocessor()
    X_train_scaled = processor.fit_transform(X_train)
    X_val_scaled = processor.transform(X_val)
    plot_feature_distributions(X_train, X_train_scaled, y_train,
                               "outputs/train_feature_distributions.png")

For normal-only models, fit the processor on X_train.loc[y_train.eq(0)]
instead. The model's allowed training population also controls scaler fitting.
The plotting population may include both classes from the training partition.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted


FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def _validate_features(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named feature columns.")
    if not X.columns.is_unique or set(X.columns) != set(FEATURE_COLUMNS):
        raise ValueError("X must contain exactly Time, V1-V28 and Amount; exclude Class/IDs.")
    if X.empty:
        raise ValueError("X must contain at least one row.")
    X = X.loc[:, FEATURE_COLUMNS]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All features must be numeric.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or infinite values.")
    return X


def _validate_labels(y, index):
    if not isinstance(y, pd.Series) or not y.index.equals(index):
        raise ValueError("y must be a Series with the same index and row order as X.")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Labels must be 0 (normal) or 1 (fraud), without missing values.")
    return y


def split_features_target(frame):
    """Separate an already selected partition; retain its original row indices."""
    if "Class" not in frame or not frame.columns.is_unique:
        raise ValueError("Input must have unique columns and include Class.")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    X = _validate_features(frame.loc[:, FEATURE_COLUMNS]).copy()
    y = _validate_labels(frame["Class"], X.index).astype("int64").copy()
    return X, y


class CreditCardPreprocessor(TransformerMixin, BaseEstimator):
    """Robust-scale Time and log1p(Amount); leave V1-V28 unchanged.

    fit/fit_transform accept training features ONLY. transform accepts any
    partition and reuses training medians/IQRs, without refitting. No clipping
    is performed: validation/test values may exceed the training range.
    Returns DataFrames in FEATURE_COLUMNS order, with original indices.
    """

    def fit(self, X, y=None):
        X = _validate_features(X)
        values = self._scaler_input(X)
        self.scaler_ = RobustScaler().fit(values)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    @staticmethod
    def _scaler_input(X):
        if X["Amount"].lt(0).any():
            raise ValueError("Amount must be nonnegative; negative values need investigation.")
        values = X.loc[:, ["Time", "Amount"]].astype(float).copy()
        values["Amount"] = np.log1p(values["Amount"])
        return values

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        X = _validate_features(X)
        result = X.astype(float).copy()
        result[["Time", "Amount"]] = self.scaler_.transform(self._scaler_input(X))
        return result


def filter_training_rows(X_train, y_train, keep_mask):
    """Apply a future, caller-supplied outlier policy to TRAINING rows only.

    keep_mask is an index-aligned boolean Series: True keeps a row. Derive
    this mask/rule from training data only. This function estimates no
    thresholds and deletes nothing unless explicitly called with a mask.
    Never use it to remove difficult validation/test cases. An extreme value
    alone is not evidence of corrupt data in anomaly detection.
    """
    X_train = _validate_features(X_train)
    y_train = _validate_labels(y_train, X_train.index)
    if (not isinstance(keep_mask, pd.Series)
            or not keep_mask.index.equals(X_train.index)
            or not pd.api.types.is_bool_dtype(keep_mask.dtype)
            or keep_mask.isna().any()):
        raise ValueError("keep_mask must be a non-null boolean Series aligned with X_train.")
    if not keep_mask.any():
        raise ValueError("The outlier policy would remove every training row.")
    return X_train.loc[keep_mask].copy(), y_train.loc[keep_mask].copy()


def plot_feature_distributions(X_before, X_after, y_train, output_path,
                               bins=80, dpi=150):
    """Save ONE PNG: Time/Amount before-after pairs, then processed V1-V28.

    Pass training-partition rows only. Each class histogram is normalized
    separately to density, so the majority class does not hide the minority.
    Within each panel both classes share bin edges. All values are included;
    no clipping, sampling or outlier removal is used for the visualization.
    """
    X_before = _validate_features(X_before)
    X_after = _validate_features(X_after)
    if not X_before.index.equals(X_after.index):
        raise ValueError("Before/after frames must have identical row indices and order.")
    y_train = _validate_labels(y_train, X_before.index)
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2.")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png.")
    panels = [
        ("Time", X_before, "Original"),
        ("Time", X_after, "Processed: RobustScaler"),
        ("Amount", X_before, "Original"),
        ("Amount", X_after, "Processed: log1p + RobustScaler"),
    ]
    panels.extend((f"V{i}", X_after, "Processed (unchanged)") for i in range(1, 29))
    fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(16, 36))
    try:
        for panel_index, (feature, frame, description) in enumerate(panels):
            ax = axes.flat[panel_index]
            edges = np.histogram_bin_edges(frame[feature].to_numpy(), bins=bins)
            for label, name, color in ((0, "Normal", "#2563eb"),
                                        (1, "Fraud", "#dc2626")):
                values = frame.loc[y_train.eq(label), feature].to_numpy()
                if values.size:
                    ax.hist(values, bins=edges, density=True, histtype="step",
                            linewidth=1.2, color=color,
                            label=f"{name} (n={values.size:,})")
            ax.set_title(f"{feature} | {description}", fontsize=10)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel("Class density", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2)
            if panel_index < 2:
                ax.legend(fontsize=8)
        fig.suptitle("Training distributions: Time/Amount before-after; V1-V28 processed only\n"
                     "Each class normalized separately; full range retained; no rows removed",
                     fontsize=14, y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.993))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)
    return output_path


def main():
    """Optional CLI: the supplied CSV MUST already be the training partition."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True,
                        help="Already split training CSV, including Class; never the full dataset.")
    parser.add_argument("--fit-normal-only", action="store_true",
                        help="Fit scaler using only normal training rows (one-class/AE models).")
    parser.add_argument("--output", default="outputs/train_feature_distributions.png",
                        help="Single output PNG path; no processed CSV or scaler is saved.")
    args = parser.parse_args()
    X_train, y_train = split_features_target(pd.read_csv(args.train_csv))
    fit_data = X_train.loc[y_train.eq(0)] if args.fit_normal_only else X_train
    processor = CreditCardPreprocessor().fit(fit_data)
    transformed = processor.transform(X_train)
    path = plot_feature_distributions(X_train, transformed, y_train, args.output)
    print(f"Saved {path.resolve()}; plotted {len(X_train):,} training rows; "
          f"scaler fitted on {len(fit_data):,} rows. No rows removed.")


if __name__ == "__main__":
    main()
