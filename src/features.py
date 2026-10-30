"""Stateless engineered features for Week 2.

Stateless means each feature is a pure function of the rows themselves: no
mean, median or encoding is learned from the data, so nothing can leak
between train and test. Stateful features (encoders, bins) live inside the
pipeline instead, so they are refit in every CV fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RATIO_FEATURES = ["payload_per_sec", "latency_per_kb", "retry_x_latency"]
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
HISTORY_FEATURES = [
    "acct_prior_calls",
    "acct_prior_mean_latency",
    "acct_prior_retry_rate",
    "secs_since_prev_call",
]
STATELESS_FEATURES = RATIO_FEATURES + TIME_FEATURES + HISTORY_FEATURES


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and interaction features.

    Rationale: automated or abusive traffic often moves unusual payloads in
    very short sessions, or retries a lot on slow calls. Raw columns hide
    this because only their combination is unusual.

    Args:
        df: Cleaned data with payload_size_kb, session_duration_sec,
            latency_ms and retry_count.

    Returns:
        A copy with payload_per_sec, latency_per_kb and retry_x_latency.
        Ratios are NaN where the denominator is missing or not positive.

    Examples:
        >>> d = pd.DataFrame({"payload_size_kb": [10.0], "session_duration_sec":
        ...     [100.0], "latency_ms": [50.0], "retry_count": [2]})
        >>> add_ratio_features(d)["payload_per_sec"].tolist()
        [0.1]
    """
    out = df.copy()
    duration = out["session_duration_sec"].where(out["session_duration_sec"] > 0)
    payload = out["payload_size_kb"].where(out["payload_size_kb"] > 0)
    out["payload_per_sec"] = out["payload_size_kb"] / duration
    out["latency_per_kb"] = out["latency_ms"] / payload
    out["retry_x_latency"] = out["retry_count"] * out["latency_ms"]
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour-of-day and day-of-week features.

    Rationale: hour 23 and hour 0 are neighbours, but as plain numbers they
    are 23 apart. Sine and cosine put them next to each other on a circle.

    Args:
        df: Data with a timezone-aware datetime column "timestamp".

    Returns:
        A copy with hour_sin, hour_cos, weekday_sin and weekday_cos.
    """
    out = df.copy()
    ts = out["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60
    weekday = ts.dt.weekday
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return out


def add_account_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-account history built only from EARLIER calls.

    Rationale: one call can look normal while the account's recent pattern
    does not. Each row sees only calls of the same account that come before
    it in time, never itself or later ones, and never any label.

    Args:
        df: Data with account_id, timestamp, latency_ms and retry_count.

    Returns:
        A copy with acct_prior_calls, acct_prior_mean_latency,
        acct_prior_retry_rate and secs_since_prev_call. They are NaN for an
        account's first call (except acct_prior_calls, which is 0).
    """
    out = df.copy()
    ordered = out.sort_values(["account_id", "timestamp"], kind="stable")
    grp = ordered.groupby("account_id", sort=False)
    prior = grp.cumcount()
    has_prior = prior.where(prior > 0)
    out["acct_prior_calls"] = prior
    out["acct_prior_mean_latency"] = (
        grp["latency_ms"].cumsum() - ordered["latency_ms"]
    ) / has_prior
    out["acct_prior_retry_rate"] = (
        grp["retry_count"].cumsum() - ordered["retry_count"]
    ) / has_prior
    out["secs_since_prev_call"] = grp["timestamp"].diff().dt.total_seconds()
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all stateless features.

    Args:
        df: Cleaned data.

    Returns:
        A copy of df with every column in STATELESS_FEATURES added.
    """
    return add_account_history_features(add_time_features(add_ratio_features(df)))


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by how common it was in the fit data.

    Rationale: an account that appears very rarely (or very often) can behave
    differently. Frequencies are learned in fit, so inside cross-validation
    they come from training rows only. Categories unseen in fit get 0.
    """

    def fit(self, x, y=None):
        """Learn category frequencies from the training rows."""
        frame = pd.DataFrame(x)
        self.columns_ = list(frame.columns)
        self.freq_ = {
            c: frame[c].value_counts(normalize=True).to_dict() for c in self.columns_
        }
        return self

    def transform(self, x):
        """Map categories to the frequencies learned in fit."""
        frame = pd.DataFrame(x)
        frame.columns = self.columns_
        return np.column_stack(
            [
                frame[c].map(self.freq_[c]).fillna(0.0).to_numpy(dtype=float)
                for c in self.columns_
            ]
        )

    def get_feature_names_out(self, input_features=None):
        """Return output column names."""
        names = self.columns_ if input_features is None else input_features
        return np.asarray([f"{c}_freq" for c in names], dtype=object)


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Bin each numeric column into quantile groups.

    Rationale: anomalies may depend on a range (for example very high
    latency) rather than on a straight-line effect. Bin edges are learned
    from the fit data only. Values beyond the edges fall in the first or
    last bin. Input must have no missing values (impute first).
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, x, y=None):
        """Learn the bin edges from the training rows."""
        arr = np.asarray(x, dtype=float)
        self.n_features_in_ = arr.shape[1]
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [
            np.unique(np.quantile(arr[:, j], qs)) for j in range(arr.shape[1])
        ]
        return self

    def transform(self, x):
        """Return the bin index of every value."""
        arr = np.asarray(x, dtype=float)
        return np.column_stack(
            [np.digitize(arr[:, j], e) for j, e in enumerate(self.edges_)]
        )

    def get_feature_names_out(self, input_features=None):
        """Return output column names."""
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)
