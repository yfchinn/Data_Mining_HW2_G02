"""Common data API for the team's stratified and chronological experiments.

Features are ordered Time, V1-V28, Amount. Class and row IDs are never features.
All P/C modes preserve the raw validation/test partitions and their row IDs.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def _module(experiment):
    if experiment not in ("exp1", "exp2"):
        raise ValueError("experiment must be 'exp1' or 'exp2'")
    spec = importlib.util.spec_from_file_location(
        f"_hw2_{experiment}", ROOT / f"data_preparation({experiment}).py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def available_modes():
    """Return P0, P1-1 ... P4-10, P5, P6, and C0-C4."""
    return _module("exp1").available_modes()


def _apply_mode(module, base, mode, random_state, smote_ratio):
    mode = str(mode).upper().strip()
    if mode in {f"C{i}" for i in range(5)}:
        result = module.prepare_contamination_data(base, random_state=random_state)[mode]
    else:
        configs = {name: (sampling, outlier, ratio) for name, sampling, outlier, ratio
                   in _module("exp1").sampling_configurations(smote_ratio)}
        if mode not in configs:
            raise ValueError(f"Unknown mode {mode!r}; choose from {available_modes()}")
        sampling, outlier, ratio = configs[mode]
        result = module.prepare_sampling_data(
            base, sampling=sampling, sampling_ratio=ratio,
            outlier_removal=outlier, random_state=random_state,
        )
    result["metadata"]["mode"] = mode
    return result


def iter_prepared_folds(experiment="exp1", mode="P0", csv_path=None, *,
                        n_splits=5, test_ratio=0.2, random_state=42,
                        train_ratio=0.6, val_ratio=0.2, smote_ratio=0.01,
                        fit_normal_only=False):
    """Yield (fold_index, data); exp2 has exactly one chronological fold.

    exp1 uses n_splits/test_ratio/random_state for its shared split.
    exp2 uses train_ratio/val_ratio; random_state controls sampling only.
    smote_ratio is fraud/normal and defaults to 0.01 for BOTH experiments.
    fit_normal_only fits the scaler on normal training rows, but retains all
    training rows in P0. The model caller must select y_train == 0 for fitting
    a normal-only model, or use C0 to receive only normal training rows.
    No files are written. A returned test partition is not permission to tune on it.
    """
    module = _module(experiment)
    csv_path = ROOT / "data" / "creditcard.csv" if csv_path is None else csv_path
    if experiment == "exp1":
        bases = module.prepare_data_folds(
            csv_path, test_ratio=test_ratio, n_splits=n_splits,
            random_state=random_state, fit_normal_only=fit_normal_only,
        )
    else:
        bases = [(0, module.prepare_data(
            csv_path, train_ratio=train_ratio, val_ratio=val_ratio,
            fit_normal_only=fit_normal_only,
        ))]
    for fold, base in bases:
        base["metadata"].update(
            experiment=experiment, fold=fold,
            n_splits=n_splits if experiment == "exp1" else 1,
            random_state=random_state,
            scaling="RobustScaler(Time, Amount); V1-V28 unchanged",
        )
        yield fold, _apply_mode(module, base, mode, random_state, smote_ratio)


def get_prepared_data(experiment="exp1", mode="P0", fold_index=0,
                      csv_path=None, **kwargs):
    """Return one fold with the same schema for either experiment."""
    count = kwargs.get("n_splits", 5) if experiment == "exp1" else 1
    if not isinstance(fold_index, int) or not 0 <= fold_index < count:
        raise ValueError(f"fold_index must be in [0, {count})")
    for fold, data in iter_prepared_folds(experiment, mode, csv_path, **kwargs):
        if fold == fold_index:
            return data
    raise RuntimeError("Requested fold not found")
