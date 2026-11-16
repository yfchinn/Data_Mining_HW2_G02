# Data_Mining_HW2_G02

Credit-card fraud anomaly detection experiments using the ULB credit-card fraud dataset (`data/creditcard.csv`, 30 features: `Time`, `V1`–`V28`, `Amount`, and binary `Class`).

This repository evaluates three model tiers across multiple data-splitting protocols (`exp1`, `exp1 v2`, `exp2`), sampling/outlier-removal modes (`P0`–`P6`), and fraud contamination levels (`C0`–`C4`), with `RobustScaler` fitted on training `Time` and `Amount`.

## Models

- **T1 - Three-Sigma** (`src/models/three_sigma.py`): Normal-only thresholding on `V14` (`abs((V14 - mean) / std) >= 3`).
- **T2 - Isolation Forest** (`src/models/isolation_forest.py`): Normal-only model (200 trees, `max_samples=256`) using `V14`, `V10`, `V12`, `V4`, `V11`, and `V17`.
- **T3 - Logistic Regression** (`src/models/logistic_regression.py`): Supervised classifier (`C=1`, `class_weight="balanced"`, `lbfgs`, `max_iter=2000`) using all 30 features.

## Setup

Requires Python `>=3.11,<3.12`, [Git LFS](https://git-lfs.com/), and [`uv`](https://docs.astral.sh/uv/).

```bash
git lfs pull
uv sync --frozen
```

## Usage

Run the test suite:

```bash
uv run --frozen pytest -q
```

Run baseline `P0` experiments:

```bash
uv run --frozen python run_experiment.py --experiment exp2 --mode P0
uv run --frozen python run_experiment.py --experiment exp1 --mode P0
```

Run `exp1 v2` fixed-holdout tier and Isolation Forest contamination experiments:

```bash
uv run --frozen python run_new_exp1_tiers.py
uv run --frozen python run_new_exp1_if_contamination.py
```

## Repository Structure

- `src/` — Shared data preparation, splitting, evaluation (`src/evaluation.py`), and model implementations (`src/models/`).
- `data/` — `creditcard.csv` dataset (tracked with Git LFS).
- `results/` — Saved metrics (`.csv`, `.json`) and experiment reports (`.md`).
- `figures/` — Analysis plots.
- `tests/` — Unit and experiment tests (`pytest`).
