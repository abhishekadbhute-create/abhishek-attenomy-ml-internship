"""Data handling utilities for the Attenomy ML practice repo."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ["transaction_id", "account_id", "timestamp", "is_anomaly"]
NUMERIC_COLUMNS = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
]


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV or JSON dataset with defensive checks and type casting.

    Args:
        path: Path to a .csv or .json file.

    Returns:
        A DataFrame with stripped strings, blanks filled as "UNKNOWN",
        numeric columns coerced, and timestamp parsed as UTC datetime.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, has an unsupported extension,
            or is missing required columns.

    Examples:
        >>> load_data("no_such_file.csv")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File not found: no_such_file.csv
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix == ".json":
        df = pd.read_json(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "none"]))
            df[col] = s.fillna("UNKNOWN")
    return df


def summarize_dataframe_health(df: pd.DataFrame) -> dict:
    """Summarize the structural health of a DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A dict with keys n_rows, n_cols, dtypes, null_pct (percent null per
        column), duplicate_rows, and memory_mb.

    Examples:
        >>> df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
        >>> h = summarize_dataframe_health(df)
        >>> h["n_rows"], h["n_cols"], h["duplicate_rows"]
        (3, 2, 1)
        >>> h["null_pct"]
        {'a': 33.33, 'b': 0.0}
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 4),
    }


def detect_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float = 1.5
) -> pd.Series:
    """Flag outliers in a numeric column using the IQR rule.

    Args:
        df: Input DataFrame.
        column: Numeric column to check.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Boolean Series, True where the value is an outlier.

    Raises:
        KeyError: If the column does not exist.

    Examples:
        >>> d = pd.DataFrame({"v": [1, 2, 3, 4, 100]})
        >>> detect_outliers_iqr(d, "v").tolist()
        [False, False, False, False, True]
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return (df[column] < q1 - multiplier * iqr) | (df[column] > q3 + multiplier * iqr)


def train_val_test_split(
    df: pd.DataFrame,
    ratios: tuple,
    seed: int,
    stratify_col: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test sets.

    Args:
        df: The DataFrame to split.
        ratios: (train, val, test) fractions that sum to 1.
        seed: Random seed for reproducibility.
        stratify_col: Optional column to stratify on (e.g. the target).

    Returns:
        A (train, val, test) tuple of DataFrames.

    Raises:
        ValueError: If ratios do not sum to 1.

    Examples:
        >>> df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
        >>> tr, va, te = train_val_test_split(
        ...     df, (0.7, 0.15, 0.15), seed=42, stratify_col="y"
        ... )
        >>> len(tr), len(va), len(te)
        (70, 15, 15)
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")

    n = len(df)
    n_train = round(n * ratios[0])
    n_test = round(n * ratios[2])

    strat = df[stratify_col] if stratify_col else None
    train, temp = train_test_split(
        df, train_size=n_train, random_state=seed, stratify=strat
    )
    strat_temp = temp[stratify_col] if stratify_col else None
    val, test = train_test_split(
        temp, test_size=n_test, random_state=seed, stratify=strat_temp
    )
    return train, val, test
