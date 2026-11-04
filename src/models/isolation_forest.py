"""Tier 2: normal-only Isolation Forest with a separate validation threshold."""

from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    def __init__(self, n_estimators=200, max_samples=256, random_state=42, n_jobs=1,
                 features=("V14", "V10", "V12", "V4", "V11", "V17")):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.features = features

    def _select_features(self, X):
        if self.features is None:
            return X
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        return X.loc[:, list(self.features)]

    def fit(self, X, y):
        X, labels = self._fit_features(self._select_features(X), y)
        normal = X.loc[labels == 0]
        if normal.empty:
            raise ValueError("Isolation Forest needs normal training rows.")
        self.estimator_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples,
            contamination="auto", random_state=self.random_state, n_jobs=self.n_jobs,
        ).fit(normal)
        self.n_training_rows_ = len(normal)
        self.threshold_ = None
        return self

    def score_samples(self, X):
        check_is_fitted(self, "estimator_")
        # sklearn assigns LOWER scores to anomalies; reverse once for all metrics.
        return -self.estimator_.score_samples(self._score_features(self._select_features(X)))
