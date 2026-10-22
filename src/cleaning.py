"""Deterministic cleaning pipeline for the Attenomy telemetry dataset.

Only fixes that do not depend on dataset statistics live here (casing,
dedupe, invalid values, bad keys). Median imputation, scaling and encoding
are fit on the training split only, inside a sklearn Pipeline.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    NUMERIC_COLUMNS,
    detect_outliers_iqr,
    load_data,
    summarize_dataframe_health,
)

KEY_PATTERN = r"^TRX_\d+$"
TEXT_COLUMNS = ["interaction_type", "channel", "error_code", "client_tier"]
OUTLIER_COLUMNS = ["session_duration_sec", "payload_size_kb", "latency_ms"]


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip categorical text, keeping the UNKNOWN sentinel.

    A missing error_code means no error occurred, so it becomes NO_ERROR.
    """
    for col in [*TEXT_COLUMNS, "region"]:
        s = df[col].astype("string").str.strip().str.lower()
        if col != "region":
            s = s.str.replace(r"[\s\-]+", "_", regex=True)
        df[col] = s.mask(s == "unknown", "UNKNOWN")
    df["error_code"] = df["error_code"].replace("UNKNOWN", "NO_ERROR")
    return df


def drop_bad_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a corrupted transaction_id or an unparseable timestamp."""
    df["transaction_id"] = df["transaction_id"].str.upper()
    valid = df["transaction_id"].str.match(KEY_PATTERN).fillna(False)
    valid &= df["timestamp"].notna()
    return df[valid].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, then repeated transaction_ids (keep first)."""
    df = df.drop_duplicates()
    return df.drop_duplicates(subset="transaction_id", keep="first")


def invalidate_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set impossible values (negative durations, sizes, latencies) to NaN.

    The NaNs are imputed later with a median fit on the train split only.
    """
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].mask(df[col] < 0)
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 IQR outlier flags. Rows are kept, because outliers may be anomalies."""
    for col in OUTLIER_COLUMNS:
        df[f"{col}_is_outlier"] = detect_outliers_iqr(df, col).astype("int8")
    return df


def optimize_memory(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    """Downcast int64 to int32 and turn low-cardinality strings into category."""
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    for col in [*TEXT_COLUMNS, "region"]:
        if df[col].nunique() <= max_categories:
            df[col] = df[col].astype("category")
    return df


def clean(input_path: str, output_path: str) -> pd.DataFrame:
    """Run all deterministic cleaning steps and write a snappy parquet file."""
    df = load_data(input_path)
    before = summarize_dataframe_health(df)
    print(f"[1/5] Loaded {before['n_rows']:,} rows, {before['memory_mb']} MB")

    df = normalize_text(df)
    n = len(df)
    df = drop_bad_keys(df)
    print(f"[2/5] Dropped {n - len(df)} rows with bad keys or timestamps")

    n = len(df)
    df = remove_duplicates(df)
    print(f"[3/5] Removed {n - len(df)} duplicate rows")

    df = invalidate_bad_values(df)
    df = flag_outliers(df)
    print("[4/5] Invalid values set to NaN, outlier flags added")

    df = optimize_memory(df).reset_index(drop=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)

    after = summarize_dataframe_health(df)
    saved = 100 * (1 - after["memory_mb"] / before["memory_mb"])
    print(f"[5/5] Wrote {after['n_rows']:,} rows to {out}")
    print(
        f"      Memory {before['memory_mb']} MB -> {after['memory_mb']} MB "
        f"({saved:.0f}% saved)"
    )
    nulls = {c: p for c, p in after["null_pct"].items() if p > 0}
    print(f"      Remaining nulls (%): {nulls}")
    return df


def main() -> None:
    """Parse command-line arguments and run the cleaning pipeline."""
    parser = argparse.ArgumentParser(description="Clean the raw telemetry CSV.")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", default="data/clean/clean.parquet")
    args = parser.parse_args()
    clean(args.input, args.output)


if __name__ == "__main__":
    main()
