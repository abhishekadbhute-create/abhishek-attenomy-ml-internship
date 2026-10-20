"""Unit tests for src/utils.py: ingestion, health summary, outliers, splitting."""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    detect_outliers_iqr,
    load_data,
    summarize_dataframe_health,
    train_val_test_split,
)

CSV = (
    "transaction_id,account_id,timestamp,session_duration_sec,is_anomaly\n"
    "T1,A1,2026-09-06T09:42:36Z,10.5,0\n"
    "T2,  ,2026-09-07T09:42:36Z,abc,1\n"
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:04d}" for i in range(100)],
            "account_id": [f"A{i % 20:03d}" for i in range(100)],
            "timestamp": pd.date_range("2026-09-01", periods=100, freq="h"),
            "session_duration_sec": rng.exponential(scale=300, size=100),
            "device_os": rng.choice(["Android", "iOS", "Windows", "macOS"], 100),
            "is_anomaly": rng.choice([0, 1], p=[0.85, 0.15], size=100),
        }
    )


@pytest.fixture
def big_df():
    return pd.DataFrame({"x": range(1000), "y": [0] * 850 + [1] * 150})


# ---------- load_data ----------


def test_load_data_valid_csv(tmp_path, sample_df):
    f = tmp_path / "valid.csv"
    sample_df.to_csv(f, index=False)
    loaded = load_data(str(f))
    assert len(loaded) == 100
    assert list(loaded.columns) == list(sample_df.columns)


def test_load_data_empty_file_raises(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_bytes(b"")
    with pytest.raises(ValueError, match="File is empty"):
        load_data(str(f))


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "nope.csv"))


def test_load_data_missing_columns_raises(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(str(f))


def test_load_data_unsupported_extension_raises(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_data(str(f))


def test_load_data_cleans_strings_and_casts_numbers(tmp_path):
    f = tmp_path / "ok.csv"
    f.write_text(CSV)
    df = load_data(str(f))
    assert df.loc[1, "account_id"] == "UNKNOWN"
    assert pd.isna(df.loc[1, "session_duration_sec"])
    assert str(df["timestamp"].dtype).startswith("datetime64")


# ---------- summarize_dataframe_health ----------


def test_summarize_keys(sample_df):
    h = summarize_dataframe_health(sample_df)
    assert set(h) == {
        "n_rows",
        "n_cols",
        "dtypes",
        "null_pct",
        "duplicate_rows",
        "memory_mb",
    }
    assert h["n_rows"] == 100
    assert h["n_cols"] == 6
    assert h["duplicate_rows"] == 0


def test_summarize_null_pct_per_column():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    h = summarize_dataframe_health(df)
    assert h["null_pct"] == {"a": 50.0, "b": 0.0}


# ---------- detect_outliers_iqr ----------


def test_detect_outliers_iqr_flags_extreme_value(sample_df):
    sample_df.loc[0, "session_duration_sec"] = 999999.0
    outliers = detect_outliers_iqr(sample_df, "session_duration_sec")
    assert isinstance(outliers, pd.Series)
    assert bool(outliers.iloc[0]) is True


def test_detect_outliers_iqr_missing_column_raises(sample_df):
    with pytest.raises(KeyError):
        detect_outliers_iqr(sample_df, "no_such_column")


# ---------- train_val_test_split ----------


def test_split_lengths_sum_to_total(big_df):
    tr, va, te = train_val_test_split(big_df, (0.7, 0.15, 0.15), 42, "y")
    assert len(tr) + len(va) + len(te) == len(big_df)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)


def test_split_indices_disjoint(big_df):
    tr, va, te = train_val_test_split(big_df, (0.7, 0.15, 0.15), 42, "y")
    assert set(tr.index).isdisjoint(va.index)
    assert set(tr.index).isdisjoint(te.index)
    assert set(va.index).isdisjoint(te.index)


def test_split_stratified_and_reproducible(big_df):
    a = train_val_test_split(big_df, (0.7, 0.15, 0.15), 42, "y")
    b = train_val_test_split(big_df, (0.7, 0.15, 0.15), 42, "y")
    assert a[0].index.equals(b[0].index)
    assert a[2]["y"].mean() == pytest.approx(0.15, abs=0.01)


def test_split_without_stratify(big_df):
    tr, va, te = train_val_test_split(big_df, (0.7, 0.15, 0.15), 42)
    assert len(tr) + len(va) + len(te) == len(big_df)


def test_split_invalid_ratios_raises(big_df):
    with pytest.raises(ValueError, match="ratios must sum to 1"):
        train_val_test_split(big_df, (0.5, 0.1, 0.1), 42)
