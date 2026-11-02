"""Run three tiers on the shared P0 split; validation only by default."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.data_preparation import ROOT, iter_prepared_folds
from src.evaluation import evaluate_scores, select_f1_threshold
from src.models import ThreeSigmaDetector, IsolationForestDetector, LogisticRegressionDetector


def run_experiment(experiment="exp2", csv_path=None, output_dir=None, *,
                   folds=None, seeds=(42,), split_seed=42, n_splits=5,
                   n_estimators=200, max_samples=256, logistic_c=1.0,
                   evaluate_test=False):
    """Train fixed configurations; T2/T3 thresholds maximize validation F1.

    Model seeds do not change raw splits. T1 is deterministic; lbfgs logistic
    regression is also deterministic, so repeating seeds need not add variation.
    Timing covers model fit / one batch score call, excluding preprocessing and IO.
    """
    if experiment not in ("exp1", "exp2"):
        raise ValueError("experiment must be exp1 or exp2")
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds:
        raise ValueError("At least one model seed is required.")
    count = n_splits if experiment == "exp1" else 1
    selected_folds = set(range(count)) if folds is None else set(folds)
    if not selected_folds or any(f < 0 or f >= count for f in selected_folds):
        raise ValueError(f"folds must select indices in [0, {count})")
    csv_path = (ROOT / "data" / "creditcard.csv" if csv_path is None else Path(csv_path)).resolve()
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "outputs" / "P0" / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("rb") as source:
        dataset_hash = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "experiment": experiment, "mode": "P0", "folds": sorted(selected_folds),
        "seeds": list(seeds), "split_seed": split_seed, "n_splits": count,
        "csv_path": str(csv_path), "sha256": dataset_hash,
        "evaluate_test": evaluate_test,
        "threshold_rule": "T1 fixed score >= 3; T2/T3 maximize validation F1; higher threshold wins ties",
        "feature_selection": "V14 fixed by team; KS is not recomputed",
        "n_estimators": n_estimators, "max_samples": max_samples, "logistic_c": logistic_c,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(),
        "libraries": {name: importlib.metadata.version(name) for name in
                      ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "scikit-learn-extra")},
        "timing": "fit and batched score only; excludes preprocessing/IO; ms_per_sample is batch amortized",
        "status": "running",
    }
    manifest = output_dir / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    kwargs = dict(experiment=experiment, mode="P0", csv_path=csv_path,
                  n_splits=n_splits, random_state=split_seed)
    supervised_folds = iter_prepared_folds(**kwargs)
    normal_folds = iter_prepared_folds(**kwargs, fit_normal_only=True)
    records = []
    for (fold, full), (normal_fold, normal) in zip(supervised_folds, normal_folds, strict=True):
        assert fold == normal_fold
        if fold not in selected_folds:
            continue
        for partition in ("train", "val", "test"):
            np.testing.assert_array_equal(full[f"{partition}_row_ids"], normal[f"{partition}_row_ids"])
            pd.testing.assert_series_equal(full[f"y_{partition}"], normal[f"y_{partition}"])
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(fold_dir / "split_ids.npz", **{
            f"{p}_row_ids": full[f"{p}_row_ids"] for p in ("train", "val", "test")
        })
        for seed in seeds:
            models = [
                ("tier1_three_sigma", ThreeSigmaDetector(), normal),
                ("tier2_isolation_forest", IsolationForestDetector(
                    n_estimators=n_estimators, max_samples=max_samples, random_state=seed), normal),
                ("tier3_logistic_regression", LogisticRegressionDetector(C=logistic_c, random_state=seed), full),
            ]
            for tier, model, data in models:
                started = perf_counter()
                model.fit(data["X_train"], data["y_train"])
                training_seconds = perf_counter() - started
                started = perf_counter()
                validation_scores = model.score_samples(data["X_val"])
                validation_seconds = perf_counter() - started
                if tier != "tier1_three_sigma":
                    model.threshold_ = select_f1_threshold(data["y_val"], validation_scores)
                partitions = [("val", validation_scores, validation_seconds)]
                if evaluate_test:
                    started = perf_counter()
                    test_scores = model.score_samples(data["X_test"])
                    partitions.append(("test", test_scores, perf_counter() - started))
                stem = f"{tier}_seed_{seed}"
                joblib.dump({
                    "model": model, "scaler": data["scaler"],
                    "feature_columns": data["feature_columns"],
                    "metadata": {**data["metadata"], "seed": seed, "parameters": model.get_params()},
                }, fold_dir / f"{stem}.joblib")
                for partition, scores, elapsed in partitions:
                    y = data[f"y_{partition}"]
                    metrics = evaluate_scores(y, scores, model.threshold_)
                    record = {
                        "experiment": experiment, "mode": "P0", "fold": fold, "seed": seed,
                        "tier": tier, "partition": partition, "threshold": model.threshold_,
                        "threshold_strategy": "fixed_3sigma" if tier == "tier1_three_sigma" else "validation_best_f1",
                        "training_rows": model.n_training_rows_, "training_frauds":
                            int(data["y_train"].sum()) if tier == "tier3_logistic_regression" else 0,
                        "evaluation_rows": len(y), "evaluation_frauds": int(y.sum()),
                        "scaler_population": data["metadata"]["scaler_population"],
                        "training_seconds": training_seconds,
                        "score_ms_per_sample": elapsed * 1000 / len(y), **metrics,
                    }
                    records.append(record)
                    pd.DataFrame({
                        "row_id": data[f"{partition}_row_ids"], "y_true": y.to_numpy(),
                        "score": scores, "threshold": model.threshold_,
                        "y_pred": (scores >= model.threshold_).astype(int),
                    }).to_csv(fold_dir / f"{stem}_{partition}_predictions.csv", index=False)
                    print(f"{experiment} P0 fold={fold} seed={seed} {tier} {partition}: "
                          f"AP={metrics['average_precision']:.4f} F1={metrics['f1']:.4f} "
                          f"threshold={model.threshold_:.6g}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(output_dir / "metrics.csv", index=False)
    metric_columns = ["average_precision", "roc_auc", "f1", "precision", "recall", "fpr",
                      "training_seconds", "score_ms_per_sample"]
    summary = results.groupby(["experiment", "mode", "tier", "partition"])[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    # A single run has undefined sample std; leave blank, do not claim zero variation.
    summary.to_csv(output_dir / "summary.csv")
    metadata["status"] = "complete"
    metadata["result_rows"] = len(results)
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["exp1", "exp2"], default="exp2")
    parser.add_argument("--mode", choices=["P0"], default="P0")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", nargs="+", type=int, help="Zero-based folds; defaults to all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test only after freezing the experiment design")
    args = parser.parse_args()
    run_experiment(
        args.experiment, args.csv, args.output_dir, folds=args.folds, seeds=args.seeds,
        split_seed=args.split_seed, n_splits=args.n_splits,
        n_estimators=args.n_estimators, max_samples=args.max_samples,
        logistic_c=args.logistic_c, evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
