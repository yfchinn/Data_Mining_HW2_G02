import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from run_experiment import run_experiment
from src.evaluation import evaluate_scores, select_f1_threshold
from src.models import ThreeSigmaDetector, IsolationForestDetector, LogisticRegressionDetector


@pytest.fixture
def training_data():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(400, 30)), columns=["Time", *[f"V{i}" for i in range(1, 29)], "Amount"])
    X["Time"] = np.arange(len(X), dtype=float)
    X["Amount"] = rng.lognormal(size=len(X))
    y = pd.Series((np.arange(len(X)) % 10 == 0).astype(int), name="Class")
    X.loc[y.eq(1), "V14"] = -8.0
    return X, y


@pytest.mark.parametrize("model", [ThreeSigmaDetector(), IsolationForestDetector(n_estimators=10)])
def test_normal_only_fit_ignores_fraud_values(training_data, model):
    X, y = training_data
    model.fit(X, y)
    first_scores = model.score_samples(X)
    changed = X.copy()
    changed.loc[y.eq(1)] = 1e9
    model.fit(changed, y)
    np.testing.assert_array_equal(first_scores, model.score_samples(X))
    assert model.n_training_rows_ == int(y.eq(0).sum())
    assert first_scores[y.eq(1)].mean() > first_scores[y.eq(0)].mean()


def test_three_sigma_uses_v14_and_fixed_three_boundary():
    X = pd.DataFrame({"V14": [-1.0, 1.0, 1e6], "Amount": [100, 200, 300]})
    model = ThreeSigmaDetector().fit(X, pd.Series([0, 0, 1]))
    assert model.mean_ == 0 and model.std_ == 1
    probe = pd.DataFrame({"V14": [-3, 3, 2.999, 0], "Amount": [1e9] * 4})
    np.testing.assert_array_equal(model.predict(probe), [1, 1, 0, 0])


def test_supervised_model_and_inference_contract(training_data):
    X, y = training_data
    model = LogisticRegressionDetector()
    with pytest.raises(NotFittedError):
        model.score_samples(X)
    model.fit(X, y)
    scores = model.score_samples(X)
    assert model.n_training_rows_ == len(X)
    assert ((0 <= scores) & (scores <= 1)).all()
    assert scores[y.eq(1)].mean() > scores[y.eq(0)].mean()
    with pytest.raises(ValueError, match="columns"):
        model.score_samples(X[X.columns[::-1]])
    with pytest.raises(ValueError, match="aligned"):
        model.fit(X, y.iloc[::-1])
    with pytest.raises(ValueError, match="both normal"):
        model.fit(X, y * 0)


def test_threshold_selection_ties_and_metrics():
    # At 0.1 F1=2/3; at 0.3 F1=2/3. Prefer the higher threshold.
    y, scores = [1, 0, 0, 1], [0.1, 0.2, 0.2, 0.3]
    threshold = select_f1_threshold(y, scores)
    assert threshold == 0.3
    metrics = evaluate_scores(y, scores, threshold)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 1, 1)
    assert metrics["f1"] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="Both normal"):
        select_f1_threshold([0, 0], [0.1, 0.2])


@pytest.mark.parametrize("experiment", ["exp1", "exp2"])
def test_p0_runner_artifacts_and_validation_only(training_data, tmp_path, experiment):
    X, y = training_data
    csv = tmp_path / "data.csv"
    X.assign(Class=y).to_csv(csv, index=False)
    output = tmp_path / experiment
    metrics = run_experiment(experiment, csv, output, folds=[0], seeds=[42, 43],
                             n_splits=3, n_estimators=10, max_samples=64)
    assert len(metrics) == 6
    assert set(metrics["partition"]) == {"val"}
    assert set(metrics["tier"]) == {"tier1_three_sigma", "tier2_isolation_forest", "tier3_logistic_regression"}
    assert metrics.loc[metrics.tier != "tier3_logistic_regression", "training_frauds"].eq(0).all()
    assert metrics.loc[metrics.tier == "tier3_logistic_regression", "training_frauds"].gt(0).all()
    assert not list(output.rglob("*_test_predictions.csv"))
    config = json.loads((output / "run.json").read_text())
    assert config["status"] == "complete" and config["evaluate_test"] is False
    expected_ids = None
    for artifact in (output / "fold_0").glob("*.joblib"):
        bundle = joblib.load(artifact)
        prediction = pd.read_csv(artifact.with_name(artifact.stem + "_val_predictions.csv"),
                                 float_precision="round_trip")
        if expected_ids is not None:
            np.testing.assert_array_equal(expected_ids, prediction.row_id)
        expected_ids = prediction.row_id.to_numpy()
        assert prediction.threshold.eq(bundle["model"].threshold_).all()
        np.testing.assert_array_equal(prediction.y_pred, (prediction.score >= prediction.threshold).astype(int))
        scaler = bundle["scaler"]
        row_ids = np.load(output / "fold_0" / "split_ids.npz")["train_row_ids"]
        raw_train = X.iloc[row_ids]
        if bundle["metadata"]["scaler_population"] == "normal_train":
            raw_train = raw_train.loc[y.iloc[row_ids].eq(0)]
        np.testing.assert_allclose(scaler.center_, raw_train[["Time", "Amount"]].median())


def test_optional_test_evaluation_does_not_change_threshold(training_data, tmp_path):
    X, y = training_data
    csv = tmp_path / "data.csv"
    X.assign(Class=y).to_csv(csv, index=False)
    kwargs = dict(experiment="exp2", csv_path=csv, n_estimators=10, max_samples=64)
    first = run_experiment(**kwargs, output_dir=tmp_path / "val_only")
    # Change only chronological test features: training/validation must stay identical.
    X.loc[320:, "V14"] = 1000
    X.assign(Class=y).to_csv(csv, index=False)
    second = run_experiment(**kwargs, output_dir=tmp_path / "with_test", evaluate_test=True)
    second_val = second.loc[second.partition.eq("val")]
    np.testing.assert_array_equal(first.threshold, second_val.threshold)
    np.testing.assert_array_equal(first.average_precision, second_val.average_precision)
    assert len(second.loc[second.partition.eq("test")]) == 3
