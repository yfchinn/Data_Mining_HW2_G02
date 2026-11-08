
"""
Credit Card Fraud Detection - Training-only Feature Analysis

Modules:
1. Feature Separation Analysis:
   KS statistic and Cohen's d for V1-V28.
2. Time & Amount Feature Engineering:
   Hour_of_Day and Amount_Log1p distribution analysis.
3. Correlation Difference Matrix:
   Compare normal/fraud correlations on training data.
4. Baseline Feasibility Check:
   Estimate normal statistics on training data and evaluate
   the resulting rule on validation data.

Important:
- Exp1: stratified 5-fold CV with a fixed 20% holdout test.
- Exp2: chronological 60/20/20 split.
- Modules 1-3 use training data only.
- Module 4 fits statistics on normal training data and evaluates
  on validation data.
- Test data is never used in this script.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp

from src.data_preparation import iter_prepared_folds


sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


# =============================================================================
# 0. Data loading and split recovery
# =============================================================================

def load_raw_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["Time"] + V_COLUMNS + ["Amount", "Class"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 65)
    print(f"Dataset loaded: {len(df):,} rows")
    print(f"Normal: {(df['Class'] == 0).sum():,}")
    print(f"Fraud : {(df['Class'] == 1).sum():,}")
    print("=" * 65)

    return df


def recover_raw_split(raw_df, row_ids):
    """
    Recover original, unscaled rows using the row IDs returned
    by src.data_preparation.

    The row IDs refer to zero-based positions in the original CSV.
    """
    ids = np.asarray(row_ids, dtype=int)

    split_df = raw_df.iloc[ids].copy().reset_index(drop=True)

    return split_df


def make_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Module 1: Feature Separation Analysis
# =============================================================================

def analyze_feature_separation(train_df, top_n=6, output_dir="figures",
                               fold_label="fold_0", make_plot=True):
    """
    Calculate KS statistic and Cohen's d using training data only.
    """

    print(f"\n[Module 1] Feature separation analysis - {fold_label}")

    normal_df = train_df[train_df["Class"] == 0]
    fraud_df = train_df[train_df["Class"] == 1]

    if len(normal_df) == 0 or len(fraud_df) == 0:
        raise ValueError("Both classes must exist in the training split.")

    stats_list = []

    for col in V_COLUMNS:
        s1 = normal_df[col].dropna()
        s2 = fraud_df[col].dropna()

        ks_stat, p_value = ks_2samp(s1, s2)

        n1, n2 = len(s1), len(s2)
        var1, var2 = s1.var(), s2.var()

        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2)
            / (n1 + n2 - 2)
        )

        cohens_d = (
            (s2.mean() - s1.mean()) / pooled_std
            if pooled_std > 0 else 0.0
        )

        stats_list.append({
            "Feature": col,
            "KS_Stat": ks_stat,
            "p_value": p_value,
            "Cohens_d": cohens_d,
            "Abs_d": abs(cohens_d),
        })

    stats_df = (
        pd.DataFrame(stats_list)
        .sort_values("KS_Stat", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 features ranked by KS statistic:")
    print(
        stats_df[
            ["Feature", "KS_Stat", "Cohens_d"]
        ].head(10).to_string(index=False)
    )

    top_features = stats_df["Feature"].head(top_n).tolist()

    if make_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for idx, col in enumerate(top_features):
            sns.kdeplot(
                data=normal_df[col],
                ax=axes[idx],
                color="dodgerblue",
                label="Normal (Class 0)",
                fill=True,
                alpha=0.3,
            )

            sns.kdeplot(
                data=fraud_df[col],
                ax=axes[idx],
                color="crimson",
                label="Fraud (Class 1)",
                fill=True,
                alpha=0.3,
            )

            ks_value = stats_df.loc[
                stats_df["Feature"] == col, "KS_Stat"
            ].iloc[0]

            axes[idx].set_title(
                f"{col} (KS = {ks_value:.3f})",
                fontweight="bold",
            )
            axes[idx].legend()

        for idx in range(len(top_features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"Top {top_n} Features - Training Data Only ({fold_label})",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"fig1_feature_separation_{fold_label}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    return stats_df, top_features


# =============================================================================
# Module 2: Time and Amount Feature Engineering
# =============================================================================

def engineer_time_and_amount(train_df, output_dir="figures",
                             fold_label="fold_0", make_plot=True):
    """
    Apply deterministic transformations to training rows only.

    Time and Amount are recovered in their original units from
    the original CSV before this function is called.
    """

    print(f"\n[Module 2] Time and Amount analysis - {fold_label}")

    df = train_df.copy()

    df["Hour_of_Day"] = (df["Time"] / 3600) % 24
    df["Amount_Log1p"] = np.log1p(df["Amount"])

    if make_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Hour of day
        for label, color, name in [
            (0, "dodgerblue", "Normal (Class 0)"),
            (1, "crimson", "Fraud (Class 1)"),
        ]:
            sns.kdeplot(
                data=df[df["Class"] == label]["Hour_of_Day"],
                ax=axes[0],
                color=color,
                label=name,
                fill=True,
                alpha=0.3,
                bw_adjust=0.8,
            )

        axes[0].set_title("Transaction Density by Hour")
        axes[0].set_xlabel("Hour of Day (0-23)")
        axes[0].legend()

        # Raw Amount
        for label, color, name in [
            (0, "dodgerblue", "Normal (Class 0)"),
            (1, "crimson", "Fraud (Class 1)"),
        ]:
            sns.kdeplot(
                data=df[df["Class"] == label]["Amount"],
                ax=axes[1],
                color=color,
                label=name,
                fill=True,
                alpha=0.3,
                clip=(0, 1000),
            )

        axes[1].set_title("Raw Amount Distribution")
        axes[1].set_xlabel("Raw Amount (clipped at 1000)")
        axes[1].legend()

        # Log1p Amount
        for label, color, name in [
            (0, "dodgerblue", "Normal (Class 0)"),
            (1, "crimson", "Fraud (Class 1)"),
        ]:
            sns.kdeplot(
                data=df[df["Class"] == label]["Amount_Log1p"],
                ax=axes[2],
                color=color,
                label=name,
                fill=True,
                alpha=0.3,
            )

        axes[2].set_title("Log1p Amount Distribution")
        axes[2].set_xlabel("log(Amount + 1)")
        axes[2].legend()

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"fig2_time_amount_{fold_label}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    return df


# =============================================================================
# Module 3: Correlation Difference Matrix
# =============================================================================

def analyze_correlation_difference(train_df, top_features,
                                   output_dir="figures",
                                   fold_label="fold_0",
                                   make_plot=True):
    """
    Compare normal/fraud correlation structures using training data only.
    """

    print(f"\n[Module 3] Correlation difference - {fold_label}")

    cols = list(top_features) + ["Amount"]

    normal_df = train_df[train_df["Class"] == 0]
    fraud_df = train_df[train_df["Class"] == 1]

    corr_normal = normal_df[cols].corr(method="pearson")
    corr_fraud = fraud_df[cols].corr(method="pearson")
    corr_diff = corr_fraud - corr_normal

    if make_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(
            corr_normal, ax=axes[0],
            cmap="coolwarm", vmin=-1, vmax=1,
            annot=True, fmt=".2f", cbar=False,
        )
        axes[0].set_title("Normal Correlation")

        sns.heatmap(
            corr_fraud, ax=axes[1],
            cmap="coolwarm", vmin=-1, vmax=1,
            annot=True, fmt=".2f", cbar=False,
        )
        axes[1].set_title("Fraud Correlation")

        sns.heatmap(
            corr_diff, ax=axes[2],
            cmap="bwr", vmin=-1, vmax=1,
            annot=True, fmt=".2f",
        )
        axes[2].set_title("Correlation Difference (Fraud - Normal)")

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"fig3_correlation_difference_{fold_label}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    return corr_normal, corr_fraud, corr_diff


# =============================================================================
# Module 4: Statistical Baseline Feasibility Check
# =============================================================================

def evaluate_t1_baseline_feasibility(train_df, val_df, top_features):
    """
    Fit normal-class mean/std using training data only.
    Evaluate the resulting fixed rule on validation data.

    This is a generic multi-feature Z-score diagnostic.
    It is NOT the official T1 model, which uses its fixed V14 rule.
    """

    print("\n[Module 4] Z-score baseline feasibility - validation evaluation")

    normal_train = train_df[train_df["Class"] == 0]

    means = normal_train[top_features].mean()
    stds = normal_train[top_features].std().replace(0, np.nan)

    z_scores = (
        (val_df[top_features] - means) / stds
    ).abs()

    z_scores = z_scores.fillna(0)

    threshold = 2.0
    outlier_matrix = z_scores > threshold

    results = []

    y_true = val_df["Class"].to_numpy()

    for min_outlier_count in [1, 2, 3, 5]:
        pred_fraud = (
            outlier_matrix.sum(axis=1).to_numpy()
            >= min_outlier_count
        )

        tp = int(((pred_fraud == 1) & (y_true == 1)).sum())
        fp = int(((pred_fraud == 1) & (y_true == 0)).sum())
        fn = int(((pred_fraud == 0) & (y_true == 1)).sum())
        tn = int(((pred_fraud == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0

        results.append({
            "Min_Outlier_Count": min_outlier_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
        })

        print(
            f"At least {min_outlier_count} feature(s) beyond "
            f"2 std | "
            f"Recall: {recall * 100:.2f}% | "
            f"Precision: {precision * 100:.2f}% | "
            f"FP: {fp:,} | "
            f"FPR: {fpr * 100:.2f}%"
        )

    return pd.DataFrame(results)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2"],
        default="exp1",
    )
    parser.add_argument(
        "--csv",
        default="./data/creditcard.csv",
    )
    parser.add_argument(
        "--mode",
        default="P0",
        help="Data preparation mode. Default: P0 (no sampling).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run one specific fold. Exp1: 0-4; Exp2: 0.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        "figures",
        f"feature_analysis_{args.experiment}_{args.mode}",
    )
    make_output_dir(output_dir)

    raw_df = load_raw_data(args.csv)

    all_stats = []
    all_baseline_results = []

    # Use the existing official split interface.
    prepared_folds = iter_prepared_folds(
        experiment=args.experiment,
        mode=args.mode,
        csv_path=args.csv,
    )

    for fold, data in prepared_folds:

        if args.fold is not None and fold != args.fold:
            continue

        fold_label = f"fold_{fold}"

        print("\n" + "=" * 65)
        print(
            f"Experiment: {args.experiment} | "
            f"Mode: {args.mode} | Fold: {fold}"
        )
        print("=" * 65)

        # Recover raw training rows only.
        train_df = recover_raw_split(
            raw_df,
            data["train_row_ids"],
        )

        # Recover validation rows for Module 4 only.
        val_df = recover_raw_split(
            raw_df,
            data["val_row_ids"],
        )

        # Sanity checks: labels must align with the preparation API.
        assert np.array_equal(
            train_df["Class"].to_numpy(),
            data["y_train"].to_numpy(),
        )
        assert np.array_equal(
            val_df["Class"].to_numpy(),
            data["y_val"].to_numpy(),
        )

        # Modules 1-3: training data only.
        # Plot only the first selected fold to avoid redundant figures.
        make_plot = (args.fold is not None) or (fold == 0)

        stats_df, top_features = analyze_feature_separation(
            train_df,
            output_dir=output_dir,
            fold_label=fold_label,
            make_plot=make_plot,
        )

        stats_df["Fold"] = fold
        all_stats.append(stats_df)

        engineer_time_and_amount(
            train_df,
            output_dir=output_dir,
            fold_label=fold_label,
            make_plot=make_plot,
        )

        analyze_correlation_difference(
            train_df,
            top_features,
            output_dir=output_dir,
            fold_label=fold_label,
            make_plot=make_plot,
        )

        # Module 4: fit on normal training data, evaluate on validation.
        baseline_df = evaluate_t1_baseline_feasibility(
            train_df,
            val_df,
            top_features,
        )
        baseline_df["Fold"] = fold
        all_baseline_results.append(baseline_df)

    if not all_stats:
        raise ValueError(
            "No folds were processed. Please check --fold."
        )

    # Save fold-wise KS statistics.
    fold_stats = pd.concat(all_stats, ignore_index=True)
    fold_stats.to_csv(
        os.path.join(output_dir, "feature_separation_by_fold.csv"),
        index=False,
    )

    # Exp1: summarize the stability of KS statistics across folds.
    # Exp2: this is simply the single chronological training split.
    summary = (
        fold_stats.groupby("Feature")
        .agg(
            KS_Mean=("KS_Stat", "mean"),
            KS_Std=("KS_Stat", "std"),
            Cohen_d_Mean=("Cohens_d", "mean"),
            Cohen_d_Std=("Cohens_d", "std"),
        )
        .sort_values("KS_Mean", ascending=False)
        .reset_index()
    )

    summary.to_csv(
        os.path.join(output_dir, "feature_separation_summary.csv"),
        index=False,
    )

    baseline_all = pd.concat(
        all_baseline_results,
        ignore_index=True,
    )
    baseline_all.to_csv(
        os.path.join(output_dir, "baseline_validation_by_fold.csv"),
        index=False,
    )

    baseline_summary = (
        baseline_all.groupby("Min_Outlier_Count")
        .agg(
            Precision_Mean=("Precision", "mean"),
            Recall_Mean=("Recall", "mean"),
            FPR_Mean=("FPR", "mean"),
            FP_Mean=("FP", "mean"),
        )
        .reset_index()
    )

    baseline_summary.to_csv(
        os.path.join(output_dir, "baseline_validation_summary.csv"),
        index=False,
    )

    print("\n" + "=" * 65)
    print("Training-only feature analysis completed.")
    print(f"Output directory: {output_dir}")

    print("\nTop features by mean training-fold KS:")
    print(summary.head(10).to_string(index=False))

    print("\nBaseline validation summary:")
    print(baseline_summary.to_string(index=False))

    print("\nTest set was not used.")
    print("=" * 65)


if __name__ == "__main__":
    main()
