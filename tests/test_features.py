"""Tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    HISTORY_FEATURES,
    STATELESS_FEATURES,
    add_account_history_features,
    add_features,
    add_ratio_features,
    add_time_features,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2026-09-01 10:00",
                    "2026-09-01 11:00",
                    "2026-09-02 12:00",
                    "2026-09-01 09:00",
                ],
                utc=True,
            ),
            "payload_size_kb": [10.0, 20.0, 5.0, 8.0],
            "session_duration_sec": [100.0, np.nan, 50.0, 40.0],
            "latency_ms": [100.0, 200.0, 300.0, 50.0],
            "retry_count": [0, 2, 1, 0],
        }
    )


def test_ratio_features_values_and_missing():
    out = add_ratio_features(_df())
    assert out.loc[0, "payload_per_sec"] == pytest.approx(0.1)
    assert np.isnan(out.loc[1, "payload_per_sec"])
    assert out.loc[0, "latency_per_kb"] == pytest.approx(10.0)
    assert out.loc[1, "retry_x_latency"] == 400
    assert not np.isinf(out[["payload_per_sec", "latency_per_kb"]]).any().any()


def test_time_features_are_cyclical():
    out = add_time_features(_df())
    assert out.loc[2, "hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[2, "hour_cos"] == pytest.approx(-1.0)
    assert out[["hour_sin", "hour_cos"]].abs().max().max() <= 1.0


def test_history_uses_only_earlier_calls():
    out = add_account_history_features(_df())
    assert out["acct_prior_calls"].tolist() == [0, 1, 2, 0]
    assert np.isnan(out.loc[0, "acct_prior_mean_latency"])
    assert out.loc[1, "acct_prior_mean_latency"] == pytest.approx(100.0)
    assert out.loc[2, "acct_prior_mean_latency"] == pytest.approx(150.0)
    assert out.loc[2, "acct_prior_retry_rate"] == pytest.approx(1.0)
    assert out.loc[1, "secs_since_prev_call"] == 3600
    assert out.loc[2, "secs_since_prev_call"] == 25 * 3600


def test_history_ignores_future_rows():
    base = add_account_history_features(_df())
    changed = _df()
    changed.loc[2, "latency_ms"] = 9999.0
    after = add_account_history_features(changed)
    cols = HISTORY_FEATURES
    pd.testing.assert_frame_equal(base.loc[[0, 1], cols], after.loc[[0, 1], cols])


def test_history_independent_of_row_order():
    df = _df()
    a = add_account_history_features(df)
    b = add_account_history_features(df.sample(frac=1, random_state=0))
    pd.testing.assert_frame_equal(a[HISTORY_FEATURES], b.loc[a.index, HISTORY_FEATURES])


def test_add_features_adds_all_and_keeps_input():
    df = _df()
    out = add_features(df)
    assert set(STATELESS_FEATURES) <= set(out.columns)
    assert not set(STATELESS_FEATURES) & set(df.columns)
