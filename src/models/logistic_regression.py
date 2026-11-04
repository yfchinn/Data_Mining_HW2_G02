"""Tier 3: supervised, class-balanced logistic regression reference."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from .base import AnomalyDetector


class LogisticRegressionDetector(AnomalyDetector):
    def __init__(self, C=1.0, class_weight="balanced", max_iter=2000, random_state=42):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, labels = self._fit_features(X, y)
        if len(np.unique(labels)) != 2:
            raise ValueError("Logistic regression needs both normal and fraud training labels.")
        self.estimator_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
            random_state=self.random_state, solver="lbfgs",
        ).fit(X, labels)
        if self.estimator_.n_iter_.max() >= self.max_iter:
            raise RuntimeError("Logistic regression did not converge; check its training configuration.")
        self.n_training_rows_ = len(X)
        self.threshold_ = 0.5
        return self

    def score_samples(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(self._score_features(X))[:, 1]
