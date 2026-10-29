"""Tests for src/pipeline.py."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
)


def _toy_x(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: rng.normal(100, 20, n) for c in NUMERIC_FEATURES}
    for c in CATEGORICAL_FEATURES:
        data[c] = rng.choice(["a", "b", "c"], n)
    return pd.DataFrame(data)


def test_pipeline_fits_and_predicts():
    x = _toy_x()
    y = pd.Series([0, 1] * 30)
    pipe = build_pipeline(LogisticRegression(max_iter=200)).fit(x, y)
    assert pipe.predict(x).shape == (60,)


def test_imputer_uses_training_median_only():
    x = _toy_x()
    x.loc[0, "session_duration_sec"] = np.nan
    train = x.iloc[:40]
    y = pd.Series([0, 1] * 30).iloc[:40]
    pipe = build_pipeline(LogisticRegression(max_iter=200)).fit(train, y)
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["imputer"]
    expected = train["session_duration_sec"].median()
    assert imputer.statistics_[0] == expected
    assert expected != x["session_duration_sec"].median()


def test_unseen_category_does_not_crash():
    x = _toy_x()
    y = pd.Series([0, 1] * 30)
    pipe = build_pipeline(LogisticRegression(max_iter=200)).fit(x, y)
    new = x.iloc[:5].copy()
    new["region"] = "never-seen"
    assert pipe.predict(new).shape == (5,)
