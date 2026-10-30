"""Tests for the stateful features (fit on training rows only)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import FrequencyEncoder, QuantileBinner
from src.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline


def _toy(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: rng.normal(100, 20, n) for c in NUMERIC_FEATURES}
    for c in CATEGORICAL_FEATURES:
        data[c] = rng.choice(["a", "b", "c"], n)
    data["account_id"] = rng.choice([f"acct{i}" for i in range(10)], n)
    return pd.DataFrame(data)


def test_frequency_encoder_uses_fit_data_and_zero_for_unseen():
    train = pd.DataFrame({"a": ["x", "x", "y", "z"]})
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"a": ["x", "y", "new"]}))
    assert out.ravel().tolist() == [0.5, 0.25, 0.0]


def test_quantile_binner_edges_come_from_fit_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    binner = QuantileBinner(n_bins=4).fit(train)
    edges = binner.edges_[0].copy()
    out = binner.transform(np.array([[-5.0], [50.0], [10_000.0]]))
    assert out.ravel().tolist() == [0, 2, 3]
    np.testing.assert_array_equal(binner.edges_[0], edges)


def test_full_pipeline_with_stateful_groups_handles_unseen_account():
    x = _toy()
    y = pd.Series([0, 1] * 50)
    pipe = build_pipeline(
        LogisticRegression(max_iter=300),
        freq_cols=["account_id"],
        target_cols=["account_id"],
        bin_cols=["latency_ms"],
    ).fit(x, y)
    new = x.iloc[:5].copy()
    new["account_id"] = "never-seen"
    assert pipe.predict(new).shape == (5,)
