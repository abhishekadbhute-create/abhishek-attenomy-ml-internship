"""Leak-free preprocessing pipeline and split loading for Week 2."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.features import FrequencyEncoder, QuantileBinner, add_features
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


def build_preprocessor(
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    freq_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    bin_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build the preprocessing step (not yet fitted).

    Numeric columns get median imputation then scaling; categorical columns
    get "UNKNOWN" imputation then one-hot encoding. Optional groups:
    frequency encoding, out-of-fold target encoding, and quantile bins.
    Nothing is learned until the pipeline is fitted, so inside
    cross-validation each fold fits every step on its own training rows only.

    Args:
        num_cols: Numeric columns to use. Defaults to NUMERIC_FEATURES.
        cat_cols: Categorical columns to use. Defaults to CATEGORICAL_FEATURES.
        freq_cols: Columns to frequency-encode (for example account_id).
        target_cols: Columns to target-encode with internal cross-fitting.
        bin_cols: Numeric columns to quantile-bin then one-hot encode.

    Returns:
        An unfitted ColumnTransformer.
    """
    if num_cols is None:
        num_cols = NUMERIC_FEATURES
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES

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
    transformers = [("num", numeric, num_cols), ("cat", categorical, cat_cols)]

    if freq_cols:
        freq = Pipeline([("freq", FrequencyEncoder()), ("scaler", StandardScaler())])
        transformers.append(("freq", freq, freq_cols))
    if target_cols:
        target = Pipeline(
            [
                (
                    "te",
                    TargetEncoder(
                        target_type="binary",
                        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("target", target, target_cols))
    if bin_cols:
        binned = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("binner", QuantileBinner(n_bins=5)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("bin", binned, bin_cols))
    return ColumnTransformer(transformers)


def build_pipeline(
    model: BaseEstimator,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    freq_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    bin_cols: list[str] | None = None,
) -> Pipeline:
    """Attach a model to the preprocessing step.

    Args:
        model: Any scikit-learn classifier.
        num_cols: Numeric columns to use. Defaults to NUMERIC_FEATURES.
        cat_cols: Categorical columns to use. Defaults to CATEGORICAL_FEATURES.
        freq_cols: Columns to frequency-encode.
        target_cols: Columns to target-encode (cross-fitted).
        bin_cols: Numeric columns to quantile-bin.

    Returns:
        A Pipeline with steps "prep" and "clf".
    """
    prep = build_preprocessor(num_cols, cat_cols, freq_cols, target_cols, bin_cols)
    return Pipeline([("prep", prep), ("clf", model)])


def split_xy(
    df: pd.DataFrame, extra: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target.

    Categorical columns, and any non-numeric extra columns such as
    account_id, are cast to plain strings because scikit-learn encoders do
    not handle the pandas category dtype reliably.

    Args:
        df: A split of the cleaned data.
        extra: Additional feature columns to keep (for example engineered
            features or account_id).

    Returns:
        A (features, target) tuple.
    """
    extra = extra or []
    x = df[FEATURES + extra].copy()
    text_cols = CATEGORICAL_FEATURES + [
        c for c in extra if not pd.api.types.is_numeric_dtype(x[c])
    ]
    x[text_cols] = x[text_cols].astype(str)
    return x, df[TARGET]


def load_splits(
    path: str = "data/clean/clean.parquet",
    with_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cleaned data and recreate the Week 1 split.

    The split is deterministic (70/15/15, seed 42, stratified on the
    target), so this returns the same rows as Week 1 every time.

    Args:
        path: Path to the cleaned parquet file.
        with_features: Add the stateless engineered features before splitting.

    Returns:
        A (train, val, test) tuple of DataFrames.
    """
    df = pd.read_parquet(path)
    if with_features:
        df = add_features(df)
    return train_val_test_split(df, (0.7, 0.15, 0.15), SEED, TARGET)
