import numpy as np
import pandas as pd
import pytest

from src.data_preparation import available_modes, get_prepared_data, iter_prepared_folds


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(1200, 28)), columns=[f"V{i}" for i in range(1, 29)])
    frame.insert(0, "Time", np.arange(len(frame), dtype=float))
    frame["Amount"] = rng.lognormal(3, 1, len(frame))
    frame["Class"] = (np.arange(len(frame)) % 20 == 0).astype(int)
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("experiment", ["exp1", "exp2"])
@pytest.mark.parametrize("mode", available_modes())
def test_modes_preserve_partitions_and_row_identity(csv_path, experiment, mode):
    kwargs = dict(experiment=experiment, csv_path=csv_path, n_splits=3, smote_ratio=0.1)
    base = get_prepared_data(**kwargs)
    result = get_prepared_data(mode=mode, **kwargs)
    assert set(result) == set(base)
    raw = pd.read_csv(csv_path)
    train_ids = {int(i) for i in result["train_row_ids"] if i is not None}
    assert train_ids <= set(base["train_row_ids"])
    for part in ("train", "val", "test"):
        X, y, ids = (result[f"X_{part}"], result[f"y_{part}"], result[f"{part}_row_ids"])
        assert len(X) == len(y) == len(ids)
        assert X.index.equals(y.index)
        assert list(X.columns) == result["feature_columns"]
        assert np.isfinite(X.to_numpy()).all()
        real_positions = [i for i, row_id in enumerate(ids) if row_id is not None]
        real_ids = [int(ids[i]) for i in real_positions]
        np.testing.assert_array_equal(y.iloc[real_positions], raw.iloc[real_ids]["Class"])
        if part != "train":
            pd.testing.assert_frame_equal(X, base[f"X_{part}"])
            pd.testing.assert_series_equal(y, base[f"y_{part}"])
            np.testing.assert_array_equal(ids, base[f"{part}_row_ids"])
            assert train_ids.isdisjoint(ids)
    assert result["metadata"]["mode"] == mode
    assert result["metadata"]["experiment"] == experiment
    assert result["metadata"]["split"] == base["metadata"]["split"]
    if mode == "C0":
        assert result["y_train"].eq(0).all()
    if mode in ("P5", "P6"):
        assert any(row_id is None for row_id in result["train_row_ids"])


def test_fold_split_and_shared_holdout(csv_path):
    folds = list(iter_prepared_folds(csv_path=csv_path, n_splits=3))
    assert [i for i, _ in folds] == [0, 1, 2]
    validation_ids = set()
    test_ids = set(folds[0][1]["test_row_ids"])
    for _, data in folds:
        ids = set(data["val_row_ids"])
        assert validation_ids.isdisjoint(ids)
        validation_ids.update(ids)
        assert test_ids == set(data["test_row_ids"])
    assert len(validation_ids | test_ids) == 1200
    chronological = list(iter_prepared_folds("exp2", csv_path=csv_path))
    assert len(chronological) == 1
    data = chronological[0][1]
    assert max(data["train_row_ids"]) < min(data["val_row_ids"])
    assert max(data["val_row_ids"]) < min(data["test_row_ids"])


@pytest.mark.parametrize("experiment", ["exp1", "exp2"])
def test_scaler_population_and_heldout_leakage(csv_path, tmp_path, experiment):
    kwargs = dict(experiment=experiment, csv_path=csv_path, fit_normal_only=True)
    base = get_prepared_data(**kwargs)
    raw = pd.read_csv(csv_path)
    train = raw.iloc[base["train_row_ids"]]
    normal = train.loc[train.Class.eq(0), ["Time", "Amount"]]
    np.testing.assert_allclose(base["scaler"].center_, normal.median().to_numpy())
    assert base["metadata"]["scaler_fit_rows"] == len(normal)
    assert base["metadata"]["scaler_population"] == "normal_train"
    heldout = np.concatenate([base["val_row_ids"], base["test_row_ids"]])
    raw.loc[heldout, "Amount"] += 1e8
    changed = tmp_path / "changed.csv"
    raw.to_csv(changed, index=False)
    kwargs["csv_path"] = changed
    result = get_prepared_data(**kwargs)
    pd.testing.assert_frame_equal(base["X_train"], result["X_train"])
    np.testing.assert_array_equal(base["scaler"].center_, result["scaler"].center_)
    np.testing.assert_array_equal(base["scaler"].scale_, result["scaler"].scale_)


def test_invalid_requests(csv_path):
    with pytest.raises(ValueError, match="experiment"):
        get_prepared_data("unknown", csv_path=csv_path)
    with pytest.raises(ValueError, match="fold_index"):
        get_prepared_data("exp2", fold_index=1, csv_path=csv_path)
    with pytest.raises(ValueError, match="Unknown mode"):
        get_prepared_data(mode="P99", csv_path=csv_path)
