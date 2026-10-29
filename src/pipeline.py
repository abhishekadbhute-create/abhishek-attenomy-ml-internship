"""Leak-free preprocessing pipeline and split loading for Week 2."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import train_val_test_split

SEED = 42
TARGET = "is_anomaly"
NUMERIC_FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
]
CATEGORICAL_FEATURES = [
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step (not yet fitted).

    Numeric columns get median imputation then scaling; categorical columns
    get "UNKNOWN" imputation then one-hot encoding. Nothing is learned until
    the pipeline is fitted, so inside cross-validation each fold fits it on
    its own training rows only.

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Attach a model to the preprocessing step.

    Args:
        model: Any scikit-learn classifier.

    Returns:
        A Pipeline with steps "prep" and "clf".
    """
    return Pipeline([("prep", build_preprocessor()), ("clf", model)])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target.

    Categorical columns are cast to plain strings because scikit-learn
    imputers do not handle the pandas category dtype reliably.

    Args:
        df: A split of the cleaned data.

    Returns:
        A (features, target) tuple.
    """
    x = df[FEATURES].copy()
    x[CATEGORICAL_FEATURES] = x[CATEGORICAL_FEATURES].astype(str)
    return x, df[TARGET]


def load_splits(
    path: str = "data/clean/clean.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cleaned data and recreate the Week 1 split.

    The split is deterministic (70/15/15, seed 42, stratified on the
    target), so this returns the same rows as Week 1 every time.

    Args:
        path: Path to the cleaned parquet file.

    Returns:
        A (train, val, test) tuple of DataFrames.
    """
    df = pd.read_parquet(path)
    return train_val_test_split(df, (0.7, 0.15, 0.15), SEED, TARGET)
