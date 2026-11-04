"""Tier 1: fixed V14, two-sided 3-sigma baseline; no feature selection."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import AnomalyDetector


class ThreeSigmaDetector(AnomalyDetector):
    def __init__(self, feature="V14", threshold=3.0):
        self.feature = feature
        self.threshold = threshold

    def fit(self, X, y):
        X, labels = self._fit_features(X, y)
        if self.feature not in X:
            raise ValueError(f"Missing feature {self.feature!r}")
        normal = X.loc[labels == 0, self.feature].to_numpy(dtype=float)
        if len(normal) < 2:
            raise ValueError("3-sigma needs at least two normal training rows.")
        self.mean_ = float(np.mean(normal))
        self.std_ = float(np.std(normal, ddof=0))
        if self.std_ == 0:
            raise ValueError("3-sigma is undefined for a constant training feature.")
        self.n_training_rows_ = len(normal)
        self.threshold_ = float(self.threshold)
        return self

    def score_samples(self, X):
        check_is_fitted(self, ["mean_", "std_"])
        X = self._score_features(X)
        return np.abs((X[self.feature].to_numpy(dtype=float) - self.mean_) / self.std_)
