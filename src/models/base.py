"""Shared input validation and alert semantics for the three tiers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


def validate_features(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("X must be a DataFrame with unique named feature columns.")
    if "Class" in X.columns or any(str(c).endswith("_id") for c in X.columns):
        raise ValueError("Labels and row IDs must not be model features.")
    check_array(X, dtype=float)
    return X


def validate_labels(X, y):
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y must have aligned indices and row order.")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain one binary label (0/1) per row.")
    return labels


class AnomalyDetector(BaseEstimator):
    def _fit_features(self, X, y):
        X = validate_features(X)
        labels = validate_labels(X, y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return X, labels

    def _score_features(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = validate_features(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Scoring features must match training columns and order.")
        return X

    def predict(self, X, threshold=None):
        """Return 1 for an alert, including scores exactly equal to the threshold."""
        if threshold is None:
            check_is_fitted(self, "threshold_")
            threshold = self.threshold_
        if threshold is None:
            raise ValueError("Select a threshold on validation before predicting.")
        return (self.score_samples(X) >= threshold).astype(np.int64)
