#!/usr/bin/env python3
"""Recover Exp1 Isolation Forest FP/FN cases across all folds and seeds.

Example:
python error_analysis.py \
  --data data/creditcard.csv \
  --predictions outputs/final_test/exp1 \
  --output error_case_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED_PATTERN = re.compile(r"tier2_isolation_forest_seed_(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/final_test/exp1"),
        help="Directory containing fold_*/tier2_isolation_forest_seed_*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("error_case_analysis"),
    )
    return parser.parse_args()


def load_predictions(prediction_root: Path) -> pd.DataFrame:
    files = sorted(prediction_root.glob("fold_*/tier2_isolation_forest_seed_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No tier2 predictions found under: {prediction_root}/fold_*/"
        )

    frames = []
    for path in files:
        match = SEED_PATTERN.search(path.name)
        if match is None:
            continue

        fold_match = re.search(r"fold_(\d+)", str(path.parent))
        if fold_match is None:
            raise ValueError(f"Cannot parse fold index from: {path}")

        fold = int(fold_match.group(1))
        seed = int(match.group(1))
        pred = pd.read_csv(path)

        required = {"row_id", "y_true", "score", "threshold", "y_pred"}
        missing = required.difference(pred.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        pred = pred[["row_id", "y_true", "score", "threshold", "y_pred"]].copy()
        pred["row_id"] = pd.to_numeric(pred["row_id"], errors="raise").astype(int)
        for col in ["y_true", "score", "threshold", "y_pred"]:
            pred[col] = pd.to_numeric(pred[col], errors="raise")

        pred["fold"] = fold
        pred["seed"] = seed
        pred["run_id"] = f"fold_{fold}_seed_{seed}"
        frames.append(pred)

    result = pd.concat(frames, ignore_index=True)

    duplicated = result.duplicated(["fold", "seed", "row_id"])
    if duplicated.any():
        example = result.loc[duplicated, ["fold", "seed", "row_id"]].head()
        raise ValueError(
            "Duplicate row_id within a fold/seed prediction file:\n"
            + example.to_string(index=False)
        )

    run_count = result[["fold", "seed"]].drop_duplicates().shape[0]
    print(f"Loaded {len(files)} prediction files ({run_count} fold-seed runs).")
    print("Runs:", sorted(result[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None)))
    return result


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    raw = pd.read_csv(args.data).reset_index(drop=True)
    required_features = [
        "Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"
    ]
    missing = [c for c in required_features if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    raw["row_id"] = np.arange(len(raw), dtype=int)

    print(f"Dataset size: {len(raw):,}")
    print(f"Fraud cases: {(raw['Class'] == 1).sum():,}")

    predictions = load_predictions(args.predictions)

    # Validate row IDs and label alignment before attaching raw features.
    if not predictions["row_id"].between(0, len(raw) - 1).all():
        bad = predictions.loc[
            ~predictions["row_id"].between(0, len(raw) - 1), "row_id"
        ].head()
        raise ValueError(f"row_id outside raw dataset range: {bad.tolist()}")

    raw_labels = raw.set_index("row_id")["Class"]
    expected_labels = predictions["row_id"].map(raw_labels).to_numpy()
    observed_labels = predictions["y_true"].to_numpy().astype(int)
    if not np.array_equal(expected_labels.astype(int), observed_labels):
        mismatch = predictions.loc[
            expected_labels.astype(int) != observed_labels,
            ["fold", "seed", "row_id", "y_true"],
        ].head(10)
        raise ValueError(
            "y_true does not match creditcard.csv Class for these rows:\n"
            + mismatch.to_string(index=False)
        )

    # Confirm the score direction and decision rule rather than assuming it.
    rule_ge = (
        (predictions["score"] >= predictions["threshold"]).astype(int)
        == predictions["y_pred"].astype(int)
    ).mean()
    rule_le = (
        (predictions["score"] <= predictions["threshold"]).astype(int)
        == predictions["y_pred"].astype(int)
    ).mean()

    print("\nPrediction rule check:")
    print(f"score >= threshold matches y_pred: {rule_ge:.4%}")
    print(f"score <= threshold matches y_pred: {rule_le:.4%}")

    if rule_ge >= 0.99 and rule_ge > rule_le:
        predictions["anomaly_margin"] = (
            predictions["score"] - predictions["threshold"]
        )
    elif rule_le >= 0.99:
        predictions["anomaly_margin"] = (
            predictions["threshold"] - predictions["score"]
        )
    else:
        raise ValueError(
            "Could not confirm score/threshold decision direction. "
            "Check the prediction-generation code."
        )

    predictions["abs_margin"] = predictions["anomaly_margin"].abs()

    predictions["error_type"] = "Correct"
    predictions.loc[
        (predictions["y_true"] == 0) & (predictions["y_pred"] == 1),
        "error_type",
    ] = "FP"
    predictions.loc[
        (predictions["y_true"] == 1) & (predictions["y_pred"] == 0),
        "error_type",
    ] = "FN"

    print("\nError counts by fold and seed:")
    counts = (
        predictions.groupby(["fold", "seed", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Correct", "FN", "FP"], fill_value=0)
    )
    print(counts.to_string())

    # Summarize each unique test transaction across all fold/seed evaluations.
    summary = (
        predictions.groupby("row_id")
        .agg(
            y_true=("y_true", "first"),
            n_evaluations=("run_id", "nunique"),
            n_FP=("error_type", lambda x: int((x == "FP").sum())),
            n_FN=("error_type", lambda x: int((x == "FN").sum())),
            n_correct=("error_type", lambda x: int((x == "Correct").sum())),
            mean_abs_margin=("abs_margin", "mean"),
            mean_anomaly_margin=("anomaly_margin", "mean"),
        )
        .reset_index()
    )

    def classify(row: pd.Series) -> str:
        if row["n_FP"] == row["n_evaluations"]:
            return "Consistent FP"
        if row["n_FN"] == row["n_evaluations"]:
            return "Consistent FN"
        if row["n_FP"] > 0 or row["n_FN"] > 0:
            return "Inconsistent error"
        return "Correct"

    summary["case_type"] = summary.apply(classify, axis=1)
    summary["error_eval_count"] = summary["n_FP"] + summary["n_FN"]

    candidates = summary.loc[summary["error_eval_count"] > 0].copy()
    candidates = candidates.sort_values(
        ["n_evaluations", "error_eval_count", "mean_abs_margin"],
        ascending=[False, False, True],
    )

    # Attach original feature values. Keep row_id as an ordinary column only.
    feature_table = raw[["row_id", *required_features]].copy()
    candidates_with_features = candidates.merge(
        feature_table, on="row_id", how="left", validate="one_to_one"
    )

    candidate_ids = candidates["row_id"].to_numpy()
    case_details = predictions.loc[
        predictions["row_id"].isin(candidate_ids)
    ].merge(
        feature_table, on="row_id", how="left", validate="many_to_one"
    )
    case_details = case_details.sort_values(["row_id", "fold", "seed"])

    stable_cases = candidates_with_features.loc[
        candidates_with_features["case_type"].isin(
            ["Consistent FP", "Consistent FN"]
        )
    ].copy()

    args.output.mkdir(parents=True, exist_ok=True)
    candidates_with_features.to_csv(
        args.output / "case_candidates_exp1.csv", index=False
    )
    case_details.to_csv(
        args.output / "case_details_exp1.csv", index=False
    )
    stable_cases.to_csv(
        args.output / "stable_error_cases_exp1.csv", index=False
    )

    n_runs = predictions[["fold", "seed"]].drop_duplicates().shape[0]
    print("\n====================================")
    print("Extraction completed.")
    print("====================================")
    print(f"Number of fold-seed runs: {n_runs}")
    print(f"Unique error candidates: {len(candidates_with_features)}")
    print(
        "Consistent FP:",
        int((stable_cases["case_type"] == "Consistent FP").sum()),
    )
    print(
        "Consistent FN:",
        int((stable_cases["case_type"] == "Consistent FN").sum()),
    )
    print("\nOutput files:")
    print(args.output / "case_candidates_exp1.csv")
    print(args.output / "case_details_exp1.csv")
    print(args.output / "stable_error_cases_exp1.csv")

    preview_cols = [
        "row_id", "case_type", "n_FP", "n_FN",
        "mean_abs_margin", "Time", "Amount", "Class",
    ]
    print("\nTop candidates:")
    print(candidates_with_features[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
