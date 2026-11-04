"""All detectors return scores where larger means more anomalous (fraud=1)."""

from .three_sigma import ThreeSigmaDetector
from .isolation_forest import IsolationForestDetector
from .logistic_regression import LogisticRegressionDetector

__all__ = ["ThreeSigmaDetector", "IsolationForestDetector", "LogisticRegressionDetector"]
